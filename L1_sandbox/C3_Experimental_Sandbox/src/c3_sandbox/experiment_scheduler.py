"""Durable serial E2 broker with explicit R1 admission for real replay.

The host and injected callbacks are trusted.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import sqlite3


class ProtocolError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _identity(value):
    sha = value.get("sha256") if isinstance(value, dict) else value
    if not isinstance(sha, str) or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", sha):
        raise ProtocolError("identity requires immutable SHA256")
    canonical(value)


@dataclass(frozen=True)
class BranchSpec:
    case_id: str
    task_identity: object
    line: str
    system_identity: object
    checkpoint_id: str
    checkpoint_identity: object
    probe_id: str
    probe_identity: object
    sampling: dict
    repeat: int
    runtime_identity: object
    grader_identity: object
    is_synthetic: bool
    phase: str

    def __post_init__(self):
        for name in ("case_id", "checkpoint_id", "probe_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ProtocolError(f"missing {name}")
        for name in ("task_identity", "system_identity", "checkpoint_identity", "probe_identity", "grader_identity"):
            _identity(getattr(self, name))
        runtime = self.runtime_identity
        if isinstance(runtime, dict) and "image_digest" in runtime:
            _identity(runtime["image_digest"])
            if not runtime["image_digest"].startswith("sha256:"):
                raise ProtocolError("runtime image must carry sha256 prefix")
            if any(not isinstance(runtime.get(k), dict) for k in ("config", "model_config", "tool_config", "environment")):
                raise ProtocolError("missing complete runtime identity")
        else:
            _identity(runtime)
        if self.line not in {"E2-S", "E2-F"} or self.phase not in {"exploration", "confirmation"}:
            raise ProtocolError("invalid line/phase")
        if type(self.repeat) is not int or self.repeat < 0 or type(self.is_synthetic) is not bool:
            raise ProtocolError("invalid repeat/synthetic identity")
        if not isinstance(self.sampling, dict) or not self.sampling:
            raise ProtocolError("sampling identity required")
        canonical(asdict(self))

    @property
    def key(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ExecutionOutcome:
    result: object
    physical_cost: dict


class MethodClient:
    """Opaque bearer capability; expose through an RPC boundary to hostile methods.

    Python introspection is not an isolation boundary. Never give a hostile host
    access to this object or the database; production adapters expose its verbs.
    """
    __slots__ = ("__dispatch", "__token")

    def __init__(self, dispatch, token):
        self.__dispatch, self.__token = dispatch, token

    def commit_prediction(self, case_id, round_index, probabilities, selected_branch):
        return self.__dispatch(self.__token, "commit", case_id, round_index, probabilities, selected_branch)

    def request_probe(self, case_id, round_index):
        return self.__dispatch(self.__token, "request", case_id, round_index)

    def charge_budget(self, receipt):
        return self.__dispatch(self.__token, "charge", receipt)

    def reveal_result(self, receipt):
        return self.__dispatch(self.__token, "reveal", receipt)

    def observations(self):
        return self.__dispatch(self.__token, "observations")

    def close_exploration(self):
        return self.__dispatch(self.__token, "close")


class ExperimentScheduler:
    def __init__(self, db_path, *, qualifier, probe_authorizer, executor, logical_price=1, physical_lock_path=None, real_admission=None):
        if type(logical_price) is not int or logical_price <= 0:
            raise ProtocolError("logical price must be a positive integer")
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(physical_lock_path) if physical_lock_path else self.path.parent / ".shared-e2-execution.lock"
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.qualifier, self.probe_authorizer, self.executor = qualifier, probe_authorizer, executor
        self.real_admission = real_admission
        self.price = logical_price
        with self._db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS methods(id TEXT PRIMARY KEY,token TEXT UNIQUE,budget INTEGER,spent INTEGER DEFAULT 0,reserved INTEGER DEFAULT 0,closed INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS plans(case_id TEXT PRIMARY KEY,contract TEXT);
            CREATE TABLE IF NOT EXISTS branches(key TEXT PRIMARY KEY,case_id TEXT,spec TEXT);
            CREATE TABLE IF NOT EXISTS predictions(method TEXT,case_id TEXT,round INTEGER,body TEXT,hash TEXT,PRIMARY KEY(method,case_id,round));
            CREATE TABLE IF NOT EXISTS receipts(token TEXT PRIMARY KEY,method TEXT,case_id TEXT,round INTEGER,branch TEXT,price INTEGER,charged INTEGER DEFAULT 0,revealed INTEGER DEFAULT 0,UNIQUE(method,case_id,round));
            CREATE TABLE IF NOT EXISTS outcomes(branch TEXT PRIMARY KEY,state TEXT,result TEXT);
            CREATE TABLE IF NOT EXISTS validations(id INTEGER PRIMARY KEY,branch TEXT,report TEXT);
            CREATE TABLE IF NOT EXISTS physical(id INTEGER PRIMARY KEY,branch TEXT,state TEXT,cost TEXT,error TEXT);
            CREATE TABLE IF NOT EXISTS observations(method TEXT,receipt TEXT UNIQUE,body TEXT);
            ''')
            old = db.execute("SELECT v FROM meta WHERE k='price'").fetchone()
            if old and int(old[0]) != self.price:
                raise ProtocolError("cannot change frozen price")
            db.execute("INSERT OR IGNORE INTO meta VALUES('price',?)", (str(self.price),))

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def _serial(self):
        with open(self.lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _started(db):
        return db.execute("SELECT 1 FROM meta WHERE k='started'").fetchone() is not None

    def enroll_plan(self, case_id, *, case_contract, branches):
        # Snapshot mutable nested inputs before validation and persistence.
        specs = [BranchSpec(**json.loads(canonical(asdict(b)))) for b in branches]
        contract = json.loads(canonical(case_contract))
        if not specs or any(b.case_id != case_id for b in specs):
            raise ProtocolError("case identity mismatch")
        if contract.get("line") not in {"E2-S", "E2-F"} or any(b.line != contract["line"] for b in specs):
            raise ProtocolError("case line must match every branch")
        if contract.get("baseline_probe_id") != "original":
            raise ProtocolError("baseline must be frozen original probe")
        panel_validator = getattr(self.probe_authorizer, "validate_panel", None)
        if not callable(panel_validator):
            raise ProtocolError("trusted frozen registry panel validator required")
        required = {contract.get("baseline_probe_id"), contract.get("control_probe_id")}
        if None in required or len(required) != 2:
            raise ProtocolError("distinct baseline and control required")
        seed_banks = {}
        for phase in ("exploration", "confirmation"):
            phase_specs = [b for b in specs if b.phase == phase]
            if not phase_specs:
                raise ProtocolError("both phases required")
            seeds = []
            for repeat in {b.repeat for b in phase_specs}:
                panel = [b for b in phase_specs if b.repeat == repeat]
                if not required <= {b.probe_id for b in panel}:
                    raise ProtocolError("baseline and control required per phase/repeat")
                if len({canonical(b.sampling) for b in panel}) != 1:
                    raise ProtocolError("sampling must match within phase/repeat")
                seed = panel[0].sampling.get("seed")
                if type(seed) is not int or seed < 0:
                    raise ProtocolError("explicit nonnegative integer seed required")
                seeds.append(seed)
            if len(set(seeds)) != len(seeds):
                raise ProtocolError("repeats must use distinct seeds")
            seed_banks[phase] = set(seeds)
        if seed_banks["exploration"] & seed_banks["confirmation"]:
            raise ProtocolError("exploration/confirmation seed banks must be disjoint")
        if len({canonical({k: v for k, v in b.sampling.items() if k != "seed"}) for b in specs}) != 1:
            raise ProtocolError("non-seed sampling settings must match across case")
        for field in ("checkpoint_id", "checkpoint_identity", "task_identity", "system_identity", "runtime_identity", "grader_identity", "is_synthetic"):
            if len({canonical(getattr(b, field)) for b in specs}) != 1:
                raise ProtocolError(f"same case must share checkpoint and verified {field}")
        if len({b.key for b in specs}) != len(specs):
            raise ProtocolError("duplicate branch")
        panel_validator(case_id, contract, specs)
        with self._db() as db:
            if self._started(db):
                raise ProtocolError("plans frozen after acquisition starts")
            db.execute("INSERT INTO plans VALUES(?,?)", (case_id, canonical(contract)))
            db.executemany("INSERT INTO branches VALUES(?,?,?)", [(b.key, case_id, canonical(asdict(b))) for b in specs])

    def register_method(self, method_id, budget):
        if not isinstance(method_id, str) or not method_id or type(budget) is not int or budget < 0:
            raise ProtocolError("invalid method/budget")
        token = secrets.token_urlsafe(32)
        with self._db() as db:
            if self._started(db):
                raise ProtocolError("methods frozen after acquisition starts")
            db.execute("INSERT INTO methods(id,token,budget) VALUES(?,?,?)", (method_id, digest(token), budget))
        return MethodClient(self._dispatch, token)

    def resume_method(self, bearer_token):
        # Root adapter may persist a bearer securely; no lookup by method ID.
        with self._db() as db:
            self._method(db, bearer_token)
        return MethodClient(self._dispatch, bearer_token)

    def _method(self, db, token):
        row = db.execute("SELECT * FROM methods WHERE token=?", (digest(token),)).fetchone()
        if row is None:
            raise ProtocolError("invalid capability")
        return row

    def _dispatch(self, token, verb, *args):
        if verb == "request":
            return self._request(token, *args)
        with self._db() as db:
            method = self._method(db, token)
            mid = method["id"]
            if verb == "commit":
                case, rnd, probs, selected = args
                if method["closed"] or type(rnd) is not int or rnd < 0:
                    raise ProtocolError("closed method/invalid round")
                universe = {r["key"] for r in db.execute("SELECT key,spec FROM branches WHERE case_id=?", (case,)) if json.loads(r["spec"])["phase"] == "exploration"}
                if not universe or set(probs) != universe or selected not in universe:
                    raise ProtocolError("complete candidate universe and selected branch required")
                if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
                    raise ProtocolError("invalid probabilities")
                previous = db.execute("SELECT round FROM predictions WHERE method=? AND case_id=? ORDER BY round DESC LIMIT 1", (mid, case)).fetchone()
                if rnd != (previous[0] + 1 if previous else 0):
                    raise ProtocolError("immutable sequential predictions")
                if previous and not db.execute("SELECT 1 FROM receipts WHERE method=? AND case_id=? AND round=? AND revealed=1", (mid, case, previous[0])).fetchone():
                    raise ProtocolError("previous result must be revealed")
                if db.execute("SELECT 1 FROM receipts WHERE method=? AND branch=?", (mid, selected)).fetchone():
                    raise ProtocolError("known branch cannot be purchased again")
                snapshot = {k: method[k] for k in ("budget", "spent", "reserved")}
                body = {"method_id": mid, "case_id": case, "round": rnd, "probabilities": probs, "selected_branch": selected, "budget": snapshot, "budget_hash": digest(snapshot)}
                encoded = canonical(body)
                db.execute("INSERT INTO predictions VALUES(?,?,?,?,?)", (mid, case, rnd, encoded, digest(body)))
                db.execute("INSERT OR IGNORE INTO meta VALUES('started','1')")
                return digest(body)
            if verb == "observations":
                return [json.loads(r[0]) for r in db.execute("SELECT body FROM observations WHERE method=? ORDER BY rowid", (mid,))]
            if verb == "close":
                if db.execute("SELECT 1 FROM predictions p LEFT JOIN receipts r ON p.method=r.method AND p.case_id=r.case_id AND p.round=r.round WHERE p.method=? AND (r.token IS NULL OR r.revealed=0)", (mid,)).fetchone():
                    raise ProtocolError("pending prediction/query")
                db.execute("UPDATE methods SET closed=1 WHERE id=?", (mid,))
                return None
            receipt = db.execute("SELECT * FROM receipts WHERE token=? AND method=?", (args[0], mid)).fetchone()
            if receipt is None:
                raise ProtocolError("receipt not owned")
            if verb == "charge":
                if not receipt["charged"]:
                    db.execute("UPDATE methods SET reserved=reserved-?,spent=spent+? WHERE id=?", (receipt["price"], receipt["price"], mid))
                    db.execute("UPDATE receipts SET charged=1 WHERE token=?", (receipt["token"],))
                return receipt["price"]
            if verb == "reveal":
                if not receipt["charged"]:
                    raise ProtocolError("charge before reveal")
                prediction = db.execute("SELECT body,hash FROM predictions WHERE method=? AND case_id=? AND round=?", (mid, receipt["case_id"], receipt["round"])).fetchone()
                if prediction is None or digest(json.loads(prediction[0])) != prediction[1] or json.loads(prediction[0])["selected_branch"] != receipt["branch"]:
                    raise ProtocolError("missing/corrupt commit")
                outcome = db.execute("SELECT * FROM outcomes WHERE branch=?", (receipt["branch"],)).fetchone()
                if outcome is None or outcome["state"] != "complete":
                    raise ProtocolError("result unavailable; physical attempt will not retry")
                body = {"case_id": receipt["case_id"], "round": receipt["round"], "branch": receipt["branch"], "result": json.loads(outcome["result"])}
                db.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?)", (mid, receipt["token"], canonical(body)))
                db.execute("UPDATE receipts SET revealed=1 WHERE token=?", (receipt["token"],))
                return body
            raise ProtocolError("unknown operation")

    def _validate(self, spec, contract):
        report = self.qualifier(spec.checkpoint_id)
        if hasattr(report, "__dataclass_fields__"):
            report = asdict(report)
        report = json.loads(canonical(report))
        with self._db() as db:
            db.execute("INSERT INTO validations(branch,report) VALUES(?,?)", (spec.key, canonical(report)))
        if report.get("eligible_for_intervention") is not True:
            raise ProtocolError("checkpoint ineligible")
        for key in ("checkpoint_identity", "system_identity", "runtime_identity", "is_synthetic"):
            if report.get(key) != getattr(spec, key):
                raise ProtocolError(f"verified {key} mismatch")
        if not spec.is_synthetic:
            from .replay import ReplayQualification
            from .r1.admission import R1AdmissionAuthority
            owner = getattr(self.qualifier, "__self__", None)
            if (type(self.real_admission) is not R1AdmissionAuthority or
                    type(owner) is not ReplayQualification or
                    getattr(self.qualifier, "__func__", None) is not ReplayQualification.qualify or
                    owner.real_admission is not self.real_admission):
                raise ProtocolError("real execution requires bound replay qualification and admission authority")
            if not self.real_admission.verify(spec.checkpoint_id, owner._record(spec.checkpoint_id)):
                raise ProtocolError("real execution admission receipt invalid")
        self.probe_authorizer(spec, contract)

    def _request(self, token, case, rnd):
        with self._serial():
            with self._db() as db:
                method = self._method(db, token)
                if method["closed"]:
                    raise ProtocolError("method closed")
                pred = db.execute("SELECT body,hash FROM predictions WHERE method=? AND case_id=? AND round=?", (method["id"], case, rnd)).fetchone()
                if pred is None:
                    raise ProtocolError("commit before query")
                if digest(json.loads(pred[0])) != pred[1]:
                    raise ProtocolError("prediction hash mismatch")
                key = json.loads(pred[0])["selected_branch"]
                spec = BranchSpec(**json.loads(db.execute("SELECT spec FROM branches WHERE key=?", (key,)).fetchone()[0]))
                contract = json.loads(db.execute("SELECT contract FROM plans WHERE case_id=?", (case,)).fetchone()[0])
            self._validate(spec, contract)  # Always revalidate, including cache/retry.
            with self._db() as db:
                method = self._method(db, token)
                old = db.execute("SELECT token FROM receipts WHERE method=? AND case_id=? AND round=?", (method["id"], case, rnd)).fetchone()
                if old:
                    return old[0]
                if method["spent"] + method["reserved"] + self.price > method["budget"]:
                    raise ProtocolError("insufficient logical budget")
                receipt = secrets.token_urlsafe(32)
                db.execute("INSERT INTO receipts(token,method,case_id,round,branch,price) VALUES(?,?,?,?,?,?)", (receipt, method["id"], case, rnd, key, self.price))
                db.execute("UPDATE methods SET reserved=reserved+? WHERE id=?", (self.price, method["id"]))
            self._execute(spec)
            return receipt

    def _execute(self, spec):
        with self._db() as db:
            existing = db.execute("SELECT state FROM outcomes WHERE branch=?", (spec.key,)).fetchone()
            if existing:
                if existing[0] != "complete":
                    raise ProtocolError("indeterminate/failed branch: no automatic retry")
                return
            db.execute("INSERT INTO outcomes VALUES(?,'running',NULL)", (spec.key,))
            cur = db.execute("INSERT INTO physical(branch,state) VALUES(?,'running')", (spec.key,))
            attempt = cur.lastrowid
        try:
            outcome = self.executor(spec)
            if not isinstance(outcome, ExecutionOutcome):
                raise ProtocolError("invalid executor outcome")
            cost = outcome.physical_cost
            if not isinstance(cost, dict) or set(cost) != {"input_tokens", "output_tokens", "gpu_seconds", "environment_seconds", "wall_seconds"}:
                raise ProtocolError("complete physical cost components required")
            for name, value in cost.items():
                if value is None:
                    continue
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ProtocolError("invalid physical cost")
                if name.endswith("tokens") and type(value) is not int:
                    raise ProtocolError("token counts must be integer or unknown")
            result = canonical(outcome.result)
        except BaseException as error:
            # A callback exception may have occurred after an external side effect.
            with self._db() as db:
                db.execute("UPDATE outcomes SET state='indeterminate' WHERE branch=?", (spec.key,))
                db.execute("UPDATE physical SET state='indeterminate',error=? WHERE id=?", (f"{type(error).__name__}: {error}", attempt))
            raise ProtocolError("physical execution indeterminate; do not retry") from error
        with self._db() as db:
            db.execute("UPDATE outcomes SET state='complete',result=? WHERE branch=?", (result, spec.key))
            db.execute("UPDATE physical SET state='complete',cost=? WHERE id=?", (canonical(outcome.physical_cost), attempt))

    def finalize_confirmation(self):
        with self._serial():
            with self._db() as db:
                if not self._started(db) or not db.execute("SELECT 1 FROM methods").fetchone() or db.execute("SELECT 1 FROM methods WHERE closed=0").fetchone():
                    raise ProtocolError("all frozen methods must explicitly close exploration")
                rows = list(db.execute("SELECT b.spec,p.contract FROM branches b JOIN plans p ON b.case_id=p.case_id ORDER BY b.key"))
            results = {}
            for row in rows:
                spec = BranchSpec(**json.loads(row[0]))
                if spec.phase != "confirmation":
                    continue
                self._validate(spec, json.loads(row[1]))
                self._execute(spec)
                with self._db() as db:
                    results[spec.key] = json.loads(db.execute("SELECT result FROM outcomes WHERE branch=?", (spec.key,)).fetchone()[0])
            with self._db() as db:
                db.execute("INSERT OR IGNORE INTO meta VALUES('confirmation_sealed','1')")
            return results

    def audit(self):
        """Privileged physical/logical/state audit, never a method verb."""
        with self._db() as db:
            return {table: [dict(row) for row in db.execute(f"SELECT * FROM {table}")] for table in ("methods", "predictions", "receipts", "validations", "physical", "outcomes", "observations")}
