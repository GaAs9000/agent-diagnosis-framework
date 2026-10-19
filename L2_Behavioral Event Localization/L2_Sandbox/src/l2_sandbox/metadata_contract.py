"""Pure, bounded post-hoc metadata comparison; never rewrites evidence."""
from __future__ import annotations

import copy
import hashlib
import re
import shlex

VERSION = 'bounded-mechanism-contract-v2'
POINTER_COMMAND = ('pip install unoconv 2>&1 | tail -5 || true\n'
                   'which libreoffice 2>/dev/null || which soffice 2>/dev/null || echo "No libreoffice found"')
POINTER = re.compile(r'(<[A-Za-z_][A-Za-z0-9_.]* object at )0x[0-9a-fA-F]+(>)')
LS_ROW = re.compile(r'^(?P<mode>[bcdlps-][rwxStTs-]{9})\s+(?P<links>\d+)\s+'
                    r'(?P<uid>\S+)\s+(?P<gid>\S+)\s+(?P<size>\d+)\s+'
                    r'(?P<mtime>[A-Z][a-z]{2}\s+\d{1,2}\s+(?:\d{2}:\d{2}|\d{4}))\s+'
                    r'(?P<name>[^\r\n]+)(?P<newline>\n?)$')


def listing_command(command):
    """Accept only cd/cp workspace operands followed by a single literal ls -la."""
    if not isinstance(command, str) or any(x in command for x in ('`', '$', '\\', ';', '|', '<', '>')):
        return False
    segments = command.strip().replace('&&', '\n').splitlines()
    if not segments or any(not s.strip() for s in segments):
        return False
    try:
        rows = [shlex.split(s) for s in segments]
    except ValueError:
        return False
    def workspace(p):
        return p.startswith('/workspace/') and '..' not in p.split('/') and not any(c in p for c in '*?[]&')
    if rows[-1][:2] != ['ls', '-la'] or len(rows[-1]) not in (2, 3):
        return False
    if len(rows[-1]) == 3 and not workspace(rows[-1][2]):
        return False
    for row in rows[:-1]:
        if not ((len(row) == 2 and row[0] == 'cd' and workspace(row[1])) or
                (len(row) == 3 and row[0] == 'cp' and all(workspace(p) for p in row[1:]))):
            return False
    return len(rows[-1]) == 3 or any(r[0] == 'cd' for r in rows[:-1])


def normalize_listing(data):
    text = data.decode('utf-8', errors='strict')
    lines = text.splitlines(keepends=True)
    if not lines:
        raise ValueError('empty ls output')
    total = bool(re.fullmatch(r'total \d+\n?', lines[0]))
    if total:
        lines = lines[1:]
    entries = []
    for line in lines:
        match = LS_ROW.fullmatch(line)
        if not match:
            raise ValueError('non-listing output')
        row = match.groupdict()
        # Only the fields named by v2 are removed. Order, names, permissions,
        # link counts and logical byte sizes remain observable constraints.
        entries.append((row['mode'], row['links'], row['size'], row['name'], row['newline']))
    if not entries or len({r[3] for r in entries}) != len(entries):
        raise ValueError('empty or duplicate listing entries')
    return total, tuple(entries)


def classify_metadata(call, original_obs, actual_obs, original_stdout, actual_stdout,
                      original_stderr, actual_stderr, *, state_exact, image_exact):
    command = call.get('arguments', {}).get('command', '')
    answer = {'classification': 'unexplained', 'excluded_fields': [],
              'raw_bytes_equal': original_stdout == actual_stdout and original_stderr == actual_stderr,
              'command_sha256': hashlib.sha256(command.encode()).hexdigest()}
    if not state_exact or not image_exact or original_stderr != actual_stderr:
        return answer
    if original_obs == actual_obs and answer['raw_bytes_equal']:
        answer['classification'] = 'exact'
        return answer
    if call.get('name') != 'bash' or original_obs.get('exit_code') != 0 or actual_obs.get('exit_code') != 0:
        return answer
    kind = None
    if listing_command(command):
        try:
            if normalize_listing(original_stdout) == normalize_listing(actual_stdout):
                kind = 'ls_runtime_metadata_v2'
        except (ValueError, UnicodeDecodeError):
            pass
        normalize = lambda t: normalize_listing(t.encode())
        fields = ['uid', 'gid', 'mtime', 'directory_total_allocated_blocks']
    elif command == POINTER_COMMAND:
        def normalize(t):
            return POINTER.sub(r'\1<ADDRESS>\2', t)
        left = original_stdout.decode('utf-8', errors='strict')
        right = actual_stdout.decode('utf-8', errors='strict')
        if POINTER.search(left) and POINTER.search(right) and normalize(left) == normalize(right):
            kind = 'python_object_repr_address_v1'
        fields = ['default_python_object_repr_address']
    else:
        return answer
    if kind:
        left, right = copy.deepcopy(original_obs), copy.deepcopy(actual_obs)
        try:
            for field in ('stdout', 'output'):
                if not isinstance(left.get(field), str) or not isinstance(right.get(field), str):
                    return answer
                left[field], right[field] = normalize(left[field]), normalize(right[field])
            if left == right:
                answer.update(classification=kind, excluded_fields=fields)
        except (ValueError, UnicodeDecodeError):
            pass
    return answer


def validate_projected_report(report):
    """Structural gate check; callers must additionally bind/recompute report refs."""
    rows = report.get('rows', [])
    classes = {'exact', 'declared_runtime_variance_v1', 'ls_runtime_metadata_v2', 'python_object_repr_address_v1'}
    return (report.get('schema') == VERSION + '-reaudit' and report.get('posthoc_extension') is True
            and report.get('post_hoc_forensic_extension') is True
            and report.get('new_tool_calls') == report.get('actor_requests') == report.get('analyst_requests') == 0
            and report.get('statistical_equivalence_established') is False
            and bool(rows) and len(rows) == report.get('comparisons_reviewed')
            and len({(r['batch_id'], r['replay'], r['call_id']) for r in rows}) == len(rows)
            and all(r['v2_classification'] in classes for r in rows)
            and report.get('all_comparisons_admissible') is True)
