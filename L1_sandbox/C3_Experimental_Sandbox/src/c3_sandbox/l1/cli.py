from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

for _p in [Path('.env'), Path(__file__).resolve().parents[3] / '.env']:
    if _p.is_file():
        for _line in _p.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                _k, _v = _k.strip(), _v.strip().strip('"\'')
                if _k not in os.environ:
                    os.environ[_k] = _v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='c3-l1')
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('import-gemini', help='Import one retained Gemini episode; no model or grader call')
    one.add_argument('--source', type=Path, required=True)
    one.add_argument('--output', type=Path, required=True)
    one.add_argument('--bundle', type=Path)
    one.add_argument('--episode-id')
    batch = commands.add_parser('import-gemini-run', help='Import all retained target/result.json episodes')
    batch.add_argument('--run', type=Path, required=True)
    batch.add_argument('--output', type=Path, required=True)
    batch.add_argument('--bundle', type=Path)
    for name in ('verify','inspect','events'):
        sub = commands.add_parser(name)
        sub.add_argument('package', type=Path)
        if name == 'verify': sub.add_argument('--expected-index-sha256')
        if name == 'events':
            sub.add_argument('--kind')
            sub.add_argument('--start-seq', type=int, default=0)
            sub.add_argument('--limit', type=int, default=20)
    live = commands.add_parser('run', help='Execute a native episode; Docker only, no host fallback')
    live.add_argument('--config', type=Path, required=True)
    live.add_argument('--output', type=Path, required=True)
    resume = commands.add_parser('resume', help='Cold resume a qualified checkpoint into a NEW evidence package')
    resume.add_argument('--package', type=Path, required=True)
    resume.add_argument('--checkpoint', required=True)
    resume.add_argument('--output', type=Path, required=True)
    doctor = commands.add_parser('doctor', help='Check local prerequisites without a paid model request')
    doctor.add_argument('--config', type=Path, required=True)
    commands.add_parser('registry', help='List implemented built-in modules, not roadmap placeholders')
    benchmark = commands.add_parser('benchmark')
    subs = benchmark.add_subparsers(dest='operation', required=True)
    for operation in ('list','show','fetch','lock-local'):
        b = subs.add_parser(operation)
        b.add_argument('--config', type=Path, required=True)
        if operation in ('fetch','lock-local'): b.add_argument('--task', action='append', dest='tasks')
    journal = commands.add_parser('journal', help='Read a valid journal prefix even when an episode was interrupted before sealing')
    journal.add_argument('package', type=Path)
    journal.add_argument('--limit', type=int, default=20)
    args = parser.parse_args(argv)
    def emit(value): print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    try:
        if args.command == 'journal':
            if args.limit < 0: raise ValueError('limit must be nonnegative')
            from .evidence.live import scan
            result=scan(args.package/'trajectory.jsonl')
            result['events']=result['events'][-args.limit:] if args.limit else []
            result['scope']='valid recorded prefix only; not sealed episode validity or restore qualification'
            emit(result)
            return 0 if result['complete'] else 1
        if args.command == 'registry':
            from .models.registry import registry as models
            from .benchmarks.registry import registry as benchmarks
            from .runtime.registry import registry as runtimes
            from .evaluation.registry import registry as evaluators
            emit({name:factory().names() for name,factory in [('models',models),('benchmarks',benchmarks),('runtimes',runtimes),('evaluators',evaluators)]})
            return 0
        if args.command in ('run','doctor','benchmark','resume'):
            from .core.config import resolve
            from .storage import read_json
            config = read_json(args.package / 'raw/resolved_config.json') if args.command=='resume' else resolve(args.config)
            if args.command in ('run','resume'):
                from .core.orchestrator import run
                result=run(config,args.output,parent=args.package if args.command=='resume' else None,
                           checkpoint_id=args.checkpoint if args.command=='resume' else None)
                emit(result)
                if not result['verification']['verified']: return 3
                if result['status']['execution_validity']=='invalid' or result['status']['evaluation_validity']=='invalid': return 2
                return 0
            if args.command=='doctor':
                from .core.doctor import doctor
                result=doctor(config); emit(result); return 0 if result['ready'] else 2
            if args.operation in ('fetch','lock-local'):
                from .benchmarks.officeval.source import fetch, lock_local
                emit((fetch if args.operation=='fetch' else lock_local)(config['benchmark'],args.tasks or [config['benchmark']['task_id']]))
                return 0
            from .benchmarks.registry import registry
            r=registry()
            if config['plugins']['load_installed']: r.load_installed()
            adapter=r.create(config['benchmark']['adapter'],config['benchmark'])
            if args.operation=='list': emit(adapter.list_tasks())
            else:
                task=adapter.load(config['benchmark']['task_id'])
                emit({'benchmark':task.benchmark,'task_id':task.task_id,'prompt':task.prompt,'provenance':task.provenance,
                      'assets':[{'name':a.name,'sha256':a.sha256,'size_bytes':a.size_bytes} for a in task.assets],
                      'requirements':task.requirements})
            return 0
        if args.command == 'import-gemini':
            from .import_gemini import import_episode
            result = import_episode(args.source, args.output, bundle=args.bundle, episode_id=args.episode_id)
            emit(result)
            return 0 if result['verification']['verified'] else 1
        if args.command == 'import-gemini-run':
            from .import_gemini import import_episode
            from .storage import write_json, utc_now
            sources = sorted((args.run / 'target').glob('*/result.json'))
            if not sources: raise ValueError('No target/*/result.json in the supplied run')
            args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
            results = []
            for source in sources:
                try:
                    result = import_episode(source.parent, args.output / source.parent.name, bundle=args.bundle)
                except Exception as exc:
                    result = {'source':str(source), 'error_type':type(exc).__name__, 'error':str(exc),
                              'verification':{'verified':False}}
                results.append(result)
            report = {'schema_version':'0.1.0','mode':'offline_legacy_import','created_at':utc_now(),
                      'episodes':results,'all_verified':all(r['verification']['verified'] for r in results)}
            write_json(args.output, 'import_report.json', report)
            emit(report)
            return 0 if report['all_verified'] else 1
        from .package import verify_package
        report = verify_package(args.package, expected_index_sha256=getattr(args, 'expected_index_sha256', None))
        if args.command == 'verify':
            emit(report)
            return 0 if report['verified'] else 1
        if not report['verified']:
            emit(report)
            return 1
        from .storage import read_json
        if args.command == 'inspect':
            manifest = read_json(args.package / 'manifest.json')
            emit({k: manifest[k] for k in ('episode_id','recording_mode','task','agent','status','counts','capabilities','coverage')})
        elif args.command == 'events':
            from .recorder import scan_journal
            manifest = read_json(args.package / 'manifest.json')
            if manifest['schema_version']=='0.2.0':
                from .evidence.live import scan as scan_journal
            if args.start_seq < 0 or args.limit < 1: raise ValueError('Invalid event slice')
            events = scan_journal(args.package / 'trajectory.jsonl')['events']
            selected = [e for e in events if e['seq'] >= args.start_seq and (not args.kind or e['kind'] == args.kind)]
            for event in selected[:args.limit]:
                print(json.dumps(event, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, __import__("subprocess").SubprocessError) as exc:
        emit({'completed':False,'error_type':type(exc).__name__,'error':str(exc)})
        return 2
