#!/usr/bin/env python3
"""Read-only recursive inventory and side-by-side study-rule sensitivity audit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / 'L1_sandbox' / 'C3_Experimental_Sandbox'
L2_ROOT = REPO_ROOT / 'L2_Behavioral Event Localization' / 'L2_Sandbox'
sys.path.insert(0, str(L1_ROOT / 'src'))
from c3_sandbox.l1.evaluation import study  # noqa: E402

OUT_JSON = L2_ROOT / 'reports' / 'v3_r6_study_outcome_version_audit_20261017.json'
OUT_MD = L2_ROOT / 'reports' / 'V3_R6_STUDY_OUTCOME_VERSION_AUDIT_20261017.md'


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _summary(rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
    outcomes = [r['outcomes'][version] for r in rows if r['status'] == 'processed']
    scores = [float(o['study_outcome_y']) for o in outcomes
              if o.get('study_valid') and o.get('study_outcome_y') is not None]
    return dict(valid_count=len(scores), invalid_count=len(outcomes) - len(scores),
                valid_zero_count=sum(o.get('study_classification') == 'valid_zero' for o in outcomes),
                mean_score=sum(scores) / len(scores) if scores else None)


def build_audit(runs_root: Path) -> dict[str, Any]:
    """Inventory every manifest, retaining incomplete, malformed and invalid packages."""
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for manifest_path in sorted(runs_root.rglob('manifest.json')):
        pkg = manifest_path.parent
        rel = pkg.relative_to(runs_root).as_posix()
        files = [manifest_path, pkg / 'official_grade.json']
        before = {p.name: _sha(p) for p in files}
        row: dict[str, Any] = dict(path=rel, cohort_id=rel.split('/')[0],
                                  hashes_before=before, status='error')
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if not isinstance(manifest, dict):
                raise ValueError('manifest must be an object')
            row['episode_id'] = manifest.get('episode_id') or pkg.name
            if manifest.get('recording_mode') != 'live':
                row.update(status='skipped', reason='not_live_recording')
            elif not (pkg / 'official_grade.json').is_file():
                row.update(status='skipped', reason='incomplete_missing_official_grade')
            else:
                # Malformed grade errors are recorded, never silently counted as valid.
                grade = json.loads((pkg / 'official_grade.json').read_text(encoding='utf-8'))
                if not isinstance(grade, dict):
                    raise ValueError('official_grade must be an object')
                both = study.study_outcome_by_version(pkg)
                row.update(status='processed', outcomes={v: both[v] for v in study.SUPPORTED_RULE_VERSIONS},
                           changed=both['changed'])
        except Exception as exc:
            row.update(status='error', reason=f'{type(exc).__name__}: {exc}')
        row['hashes_after'] = {p.name: _sha(p) for p in files}
        row['inputs_unchanged_during_read'] = row['hashes_before'] == row['hashes_after']
        rows.append(row)
    cohorts = []
    for cid in sorted({r['cohort_id'] for r in rows}):
        cr = [r for r in rows if r['cohort_id'] == cid]
        cohorts.append(dict(cohort_id=cid, total_packages=len(cr),
                            processed_count=sum(r['status'] == 'processed' for r in cr),
                            skipped_count=sum(r['status'] == 'skipped' for r in cr),
                            error_count=sum(r['status'] == 'error' for r in cr),
                            v1_0=_summary(cr, 'v1.0'), v1_1=_summary(cr, 'v1.1'),
                            affected_count=sum(r.get('changed', False) for r in cr)))
    source_paths = [Path(__file__), *sorted(Path(study.__file__).parent.glob('*.py'))]
    sources = {str(p.relative_to(REPO_ROOT)): _sha(p) for p in source_paths}
    coverage = dict(discovered_manifests=len(rows), processed=sum(r['status'] == 'processed' for r in rows),
                    skipped=sum(r['status'] == 'skipped' for r in rows), errors=sum(r['status'] == 'error' for r in rows))
    return dict(collection_started_at=started, generated_at=datetime.now(timezone.utc).isoformat(),
                scan_root=str(runs_root), discovery='recursive **/manifest.json; nested runs and attempts included',
                default_rule_version=study.RULE_VERSION, supported_rule_versions=list(study.SUPPORTED_RULE_VERSIONS),
                endpoint_version=study.ENDPOINT_VERSION, source_sha256=sources,
                added_in_v1_1=[x for x in study.DELIVERY_ERRORS_V1_1 if x not in study.DELIVERY_ERRORS_V1_0],
                raw_official_grade_files_modified=False,
                inputs_unchanged_during_read=all(r['inputs_unchanged_during_read'] for r in rows),
                coverage=coverage, coverage_manifest=rows, cohorts=cohorts,
                total_affected_packages_across_cohorts=sum(r.get('changed', False) for r in rows),
                affected_packages=[r for r in rows if r.get('changed')])


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ['# R6 研究终点规则版本敏感性只读审计', '',
             f"- 采集区间：`{payload['collection_started_at']}` 至 `{payload['generated_at']}`。",
             f"- 扫描范围：`{payload['scan_root']}`，递归发现所有 manifest，含嵌套 runs/attempt。",
             f"- 覆盖：`{payload['coverage']}`；逐包路径、跳过原因、错误与原件前后哈希见 JSON `coverage_manifest`。",
             f"- 规则：`{payload['supported_rule_versions']}`；默认 `{payload['default_rule_version']}`。",
             f"- 读取期间原件哈希一致：`{payload['inputs_unchanged_during_read']}`；只证明本次读取前后，不推断历史全过程。",
             '- 源码版本以 JSON `source_sha256` 内容哈希绑定；原始官方分保持只读。实时目录仍可能增长，本报告是上述采集区间的清单。', '',
             '| 批次 | manifest | 已处理/跳过/错误 | v1.0 有效/无效/均分 | v1.1 有效/无效/均分 | 变化包数 |',
             '|---|---:|---|---|---|---:|']
    for c in payload['cohorts']:
        a, b = c['v1_0'], c['v1_1']
        lines.append(f"| `{c['cohort_id']}` | {c['total_packages']} | {c['processed_count']}/{c['skipped_count']}/{c['error_count']} | {a['valid_count']}/{a['invalid_count']}/{a['mean_score']} | {b['valid_count']}/{b['invalid_count']}/{b['mean_score']} | {c['affected_count']} |")
    lines += ['', '## 版本变化包', '', '| 包路径 | v1.0 | v1.1 |', '|---|---|---|']
    for row in payload['affected_packages']:
        lines.append(f"| `{row['path']}` | `{json.dumps(row['outcomes']['v1.0'], ensure_ascii=False)}` | `{json.dumps(row['outcomes']['v1.1'], ensure_ascii=False)}` |")
    lines += ['', '## 跳过与错误', '', '| 包路径 | 状态 | 原因 |', '|---|---|---|']
    for row in payload['coverage_manifest']:
        if row['status'] != 'processed':
            lines.append(f"| `{row['path']}` | {row['status']} | {row['reason'].replace('|', '/') } |")
    return '\n'.join(lines) + '\n'


def main() -> int:
    payload = build_audit(L1_ROOT / 'runs')
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    OUT_MD.write_text(render_markdown(payload), encoding='utf-8')
    print(json.dumps(payload['coverage']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
