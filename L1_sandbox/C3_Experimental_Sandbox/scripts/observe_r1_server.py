"""Read only the known local model process and its bound loopback listener."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1')
    parser.add_argument('--pid-file', type=Path, default=ROOT / 'runs/r1-setup/model.pid')
    args = parser.parse_args()
    pid = args.pid_file.read_text().strip()
    if not pid.isdecimal():
        raise ValueError('explicit owned model PID required')
    command = subprocess.check_output(['/bin/ps', '-p', pid, '-o', 'args='], text=True).strip()
    started = subprocess.check_output(['/bin/ps', '-p', pid, '-o', 'lstart='], text=True).strip()
    sockets = subprocess.run(['/usr/sbin/lsof', '-nP', '-a', '-p', pid, '-iTCP:18080', '-sTCP:LISTEN'],
                             text=True, capture_output=True, timeout=30)
    required = ('mlx_lm.server', 'models/Qwen3-4B-Instruct-2507-4bit', '--host 127.0.0.1',
                '--port 18080', '--prompt-cache-size 0', '--decode-concurrency 1', '--prompt-concurrency 1')
    verified = all(item in command for item in required) and sockets.returncode == 0 and '127.0.0.1:18080' in sockets.stdout
    report = {'pid': int(pid), 'process_start': started, 'observed_argv': command,
              'listener_observation': sockets.stdout, 'configuration_verified': verified,
              'scope': 'owned live process and listener; disk/tokenizer separately checked per branch'}
    with (args.run_root / 'server_observation.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    if not verified:
        raise RuntimeError('local model serving configuration mismatch')
    print('Local model serving configuration verified.')


if __name__ == '__main__':
    main()
