"""Native tool surfaces for minimal (2-tool) and Pi (@mariozechner/pi-coding-agent@0.73.1 4-tool) harnesses."""
from __future__ import annotations
import difflib
import json
from pathlib import Path
import re
import unicodedata
from ..core.contracts import L1Error, ProtocolError, Result
from ..storage import confined, regular_file, utc_now

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB

TOOLS = [
 {'name': 'run', 'description': 'Execute a shell command in an isolated Linux container. Working directory is /workspace. Each call starts a fresh container. Only /workspace persists. No network. Use input/ for original files and submission/ for final deliverables.',
  'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'POSIX shell command, e.g. python with a heredoc'}}, 'required': ['command']}},
 {'name': 'read_image', 'description': 'View an existing PNG/JPEG/WebP image in /workspace. Returns actual image bytes to the model.',
  'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Workspace-relative image path'}}, 'required': ['path']}}
]

PI_TOOLS = [
 {
  'name': 'read',
  'description': (
      f'Read the contents of a file. Supports text files and images (jpg, png, gif, webp). '
      f'Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines '
      f'or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. '
      f'When you need the full file, continue with offset until complete.'
  ),
  'parameters': {
      'type': 'object',
      'properties': {
          'path': {'type': 'string', 'description': 'Path to the file to read (relative or absolute)'},
          'offset': {'type': 'number', 'description': 'Line number to start reading from (1-indexed)'},
          'limit': {'type': 'number', 'description': 'Maximum number of lines to read'},
      },
      'required': ['path'],
  },
 },
 {
  'name': 'bash',
  'description': (
      f'Execute a bash command in the current working directory. Returns stdout and stderr. '
      f'Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB '
      f'(whichever is hit first). If truncated, full output is saved to a temp file. '
      f'Optionally provide a timeout in seconds.'
  ),
  'parameters': {
      'type': 'object',
      'properties': {
          'command': {'type': 'string', 'description': 'Bash command to execute'},
          'timeout': {'type': 'number', 'description': 'Timeout in seconds (optional, no default timeout)'},
      },
      'required': ['command'],
  },
 },
 {
  'name': 'edit',
  'description': (
      'Edit a single file using exact text replacement. Every edits[].oldText must match a unique, '
      'non-overlapping region of the original file. If two changes affect the same block or nearby lines, '
      'merge them into one edit instead of emitting overlapping edits. Do not include large unchanged '
      'regions just to connect distant changes.'
  ),
  'parameters': {
      'type': 'object',
      'properties': {
          'path': {'type': 'string', 'description': 'Path to the file to edit (relative or absolute)'},
          'edits': {
              'type': 'array',
              'description': (
                  'One or more targeted replacements. Each edit is matched against the original file, '
                  'not incrementally. Do not include overlapping or nested edits. If two changes touch '
                  'the same block or nearby lines, merge them into one edit instead.'
              ),
              'items': {
                  'type': 'object',
                  'properties': {
                      'oldText': {
                          'type': 'string',
                          'description': (
                              'Exact text for one targeted replacement. It must be unique in the original '
                              'file and must not overlap with any other edits[].oldText in the same call.'
                          ),
                      },
                      'newText': {'type': 'string', 'description': 'Replacement text for this targeted edit.'},
                  },
                  'required': ['oldText', 'newText'],
                  'additionalProperties': False,
              },
          },
      },
      'required': ['path', 'edits'],
      'additionalProperties': False,
  },
 },
 {
  'name': 'write',
  'description': (
      "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
      'Automatically creates parent directories.'
  ),
  'parameters': {
      'type': 'object',
      'properties': {
          'path': {'type': 'string', 'description': 'Path to the file to write (relative or absolute)'},
          'content': {'type': 'string', 'description': 'Content to write to the file'},
      },
      'required': ['path', 'content'],
  },
 },
]


def active_harness(config: dict) -> str:
    agent = config.get('agent') or {}
    h = agent.get('harness')
    if h in ('minimal', 'pi'):
        return h
    name = str(agent.get('name') or '')
    if name in ('pi', 'pi-coding-agent', '@mariozechner/pi-coding-agent'):
        return 'pi'
    return 'minimal'


def tools_for_config(config: dict) -> list[dict]:
    supports_images = bool((config.get('model') or {}).get('supports_images', True))
    if active_harness(config) == 'pi':
        return PI_TOOLS
    return TOOLS if supports_images else TOOLS[:1]


def _prepare_edit_arguments(args: dict) -> dict:
    if not isinstance(args, dict):
        return args
    out = dict(args)
    if isinstance(out.get('edits'), str):
        try:
            parsed = json.loads(out['edits'])
            if isinstance(parsed, list):
                out['edits'] = parsed
        except (ValueError, TypeError):
            pass
    if isinstance(out.get('oldText'), str) and isinstance(out.get('newText'), str):
        edits = list(out['edits']) if isinstance(out.get('edits'), list) else []
        edits.append({'oldText': out.pop('oldText'), 'newText': out.pop('newText')})
        out['edits'] = edits
    return out


def validate_call(call, images: bool = True, harness: str | None = None):
    if not isinstance(call.arguments, dict):
        raise ProtocolError('Invalid native tool arguments')
    if harness in (None, 'minimal') and call.name in ('run', 'read_image'):
        if call.name == 'read_image' and not images:
            raise ProtocolError('Unknown or unavailable tool: ' + call.name)
        field = 'command' if call.name == 'run' else 'path'
        if (
            set(call.arguments) != {field}
            or not isinstance(call.arguments[field], str)
            or not call.arguments[field]
            or '\x00' in call.arguments[field]
        ):
            raise ProtocolError('Invalid native tool arguments')
        if len(call.arguments[field]) > 262144:
            raise ProtocolError('Tool argument size exceeded')
        return
    if harness in (None, 'pi') and call.name in ('read', 'bash', 'edit', 'write'):
        args = call.arguments
        if call.name == 'read':
            if not set(args) <= {'path', 'offset', 'limit'} or 'path' not in args:
                raise ProtocolError('Invalid read tool arguments')
            path = args['path']
            if not isinstance(path, str) or not path or '\x00' in path or len(path) > 4096:
                raise ProtocolError('Invalid read path argument')
            for k in ('offset', 'limit'):
                if k in args and args[k] is not None:
                    if isinstance(args[k], bool) or not isinstance(args[k], (int, float)):
                        raise ProtocolError(f'Invalid read {k} argument')
            return
        if call.name == 'bash':
            if not set(args) <= {'command', 'timeout'} or 'command' not in args:
                raise ProtocolError('Invalid bash tool arguments')
            cmd = args['command']
            if not isinstance(cmd, str) or not cmd or '\x00' in cmd:
                raise ProtocolError('Invalid bash command argument')
            if len(cmd) > 262144:
                raise ProtocolError('Tool argument size exceeded')
            if 'timeout' in args and args['timeout'] is not None:
                if isinstance(args['timeout'], bool) or not isinstance(args['timeout'], (int, float)):
                    raise ProtocolError('Invalid bash timeout argument')
            return
        if call.name == 'write':
            if set(args) != {'path', 'content'}:
                raise ProtocolError('Invalid write tool arguments')
            if not isinstance(args['path'], str) or not args['path'] or '\x00' in args['path'] or len(args['path']) > 4096:
                raise ProtocolError('Invalid write path argument')
            if not isinstance(args['content'], str) or '\x00' in args['content']:
                raise ProtocolError('Invalid write content argument')
            if len(args['content']) > 4194304:
                raise ProtocolError('Tool argument size exceeded')
            return
        if call.name == 'edit':
            norm = _prepare_edit_arguments(args)
            if not set(norm) <= {'path', 'edits'} or 'path' not in norm or 'edits' not in norm:
                raise ProtocolError('Invalid edit tool arguments')
            if not isinstance(norm['path'], str) or not norm['path'] or '\x00' in norm['path'] or len(norm['path']) > 4096:
                raise ProtocolError('Invalid edit path argument')
            edits = norm['edits']
            if not isinstance(edits, list) or not edits:
                raise ProtocolError('Edit tool edits must be a non-empty list')
            total_len = 0
            for item in edits:
                if not isinstance(item, dict) or set(item) != {'oldText', 'newText'}:
                    raise ProtocolError('Invalid edit replacement entry')
                if not isinstance(item['oldText'], str) or not isinstance(item['newText'], str):
                    raise ProtocolError('Edit oldText/newText must be strings')
                if '\x00' in item['oldText'] or '\x00' in item['newText']:
                    raise ProtocolError('Null byte in edit arguments')
                total_len += len(item['oldText']) + len(item['newText'])
            if total_len > 4194304:
                raise ProtocolError('Tool argument size exceeded')
            return
    raise ProtocolError('Unknown or unavailable tool: ' + str(call.name))


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f'{num_bytes}B'
    if num_bytes < 1024 * 1024:
        return f'{num_bytes / 1024:.1f}KB'
    return f'{num_bytes / (1024 * 1024):.1f}MB'


def truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    total_bytes = len(content.encode('utf-8'))
    lines = content.split('\n')
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            'content': content,
            'truncated': False,
            'truncatedBy': None,
            'totalLines': total_lines,
            'totalBytes': total_bytes,
            'outputLines': total_lines,
            'outputBytes': total_bytes,
            'lastLinePartial': False,
            'firstLineExceedsLimit': False,
            'maxLines': max_lines,
            'maxBytes': max_bytes,
        }
    first_line_bytes = len(lines[0].encode('utf-8'))
    if first_line_bytes > max_bytes:
        return {
            'content': '',
            'truncated': True,
            'truncatedBy': 'bytes',
            'totalLines': total_lines,
            'totalBytes': total_bytes,
            'outputLines': 0,
            'outputBytes': 0,
            'lastLinePartial': False,
            'firstLineExceedsLimit': True,
            'maxLines': max_lines,
            'maxBytes': max_bytes,
        }
    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = 'lines'
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        lb = len(line.encode('utf-8')) + (1 if i > 0 else 0)
        if out_bytes + lb > max_bytes:
            truncated_by = 'bytes'
            break
        out_lines.append(line)
        out_bytes += lb
    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = 'lines'
    out_content = '\n'.join(out_lines)
    return {
        'content': out_content,
        'truncated': True,
        'truncatedBy': truncated_by,
        'totalLines': total_lines,
        'totalBytes': total_bytes,
        'outputLines': len(out_lines),
        'outputBytes': len(out_content.encode('utf-8')),
        'lastLinePartial': False,
        'firstLineExceedsLimit': False,
        'maxLines': max_lines,
        'maxBytes': max_bytes,
    }


def _truncate_str_bytes_from_end(s: str, max_bytes: int) -> str:
    buf = s.encode('utf-8')
    if len(buf) <= max_bytes:
        return s
    start = len(buf) - max_bytes
    while start < len(buf) and (buf[start] & 0xC0) == 0x80:
        start += 1
    return buf[start:].decode('utf-8', 'replace')


def truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    total_bytes = len(content.encode('utf-8'))
    lines = content.split('\n')
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            'content': content,
            'truncated': False,
            'truncatedBy': None,
            'totalLines': total_lines,
            'totalBytes': total_bytes,
            'outputLines': total_lines,
            'outputBytes': total_bytes,
            'lastLinePartial': False,
            'firstLineExceedsLimit': False,
            'maxLines': max_lines,
            'maxBytes': max_bytes,
        }
    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = 'lines'
    last_line_partial = False
    for i in range(len(lines) - 1, -1, -1):
        if len(out_lines) >= max_lines:
            break
        line = lines[i]
        lb = len(line.encode('utf-8')) + (1 if out_lines else 0)
        if out_bytes + lb > max_bytes:
            truncated_by = 'bytes'
            if not out_lines:
                t_line = _truncate_str_bytes_from_end(line, max_bytes)
                out_lines.insert(0, t_line)
                out_bytes = len(t_line.encode('utf-8'))
                last_line_partial = True
            break
        out_lines.insert(0, line)
        out_bytes += lb
    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = 'lines'
    out_content = '\n'.join(out_lines)
    return {
        'content': out_content,
        'truncated': True,
        'truncatedBy': truncated_by,
        'totalLines': total_lines,
        'totalBytes': total_bytes,
        'outputLines': len(out_lines),
        'outputBytes': len(out_content.encode('utf-8')),
        'lastLinePartial': last_line_partial,
        'firstLineExceedsLimit': False,
        'maxLines': max_lines,
        'maxBytes': max_bytes,
    }


def _resolve_workspace_target(workspace: Path, raw_path: str) -> tuple[Path, str]:
    clean = raw_path.strip()
    if clean == '/workspace':
        rel = '.'
    elif clean.startswith('/workspace/'):
        rel = clean[len('/workspace/'):]
    else:
        rel = clean
    try:
        p = confined(workspace, rel)
    except ValueError as exc:
        raise L1Error(str(exc)) from exc
    rel_posix = p.relative_to(workspace.resolve()).as_posix()
    return p, rel_posix


def _detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if len(data) >= 12 and data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    return None


def _tool_error_result(err_msg: str, boundary: str, start: str) -> Result:
    err_bytes = err_msg.encode('utf-8', 'replace')
    obs = {
        'output': err_msg,
        'stdout': '',
        'stderr': err_msg,
        'exit_code': 1,
        'stop_reason': None,
        'output_complete': True,
    }
    prov = {'boundary': boundary, 'started_at': start, 'finished_at': utc_now(), 'quiescent': True}
    return Result(obs, b'', err_bytes, provenance=prov)


def execute_pi_read(call, workspace: Path, max_image_bytes: int, supports_images: bool = True) -> Result:
    start = utc_now()
    raw_path = call.arguments['path']
    try:
        p, rel_posix = _resolve_workspace_target(workspace, raw_path)
    except L1Error as exc:
        return _tool_error_result(f'Could not read file {raw_path}: {exc}', 'pi_workspace_read', start)
    if not p.exists():
        return _tool_error_result(f'ENOENT: no such file or directory, open \'{raw_path}\'', 'pi_workspace_read', start)
    if p.is_dir():
        return _tool_error_result(f'EISDIR: illegal operation on a directory, read \'{raw_path}\'', 'pi_workspace_read', start)
    regular_file(p)
    data = p.read_bytes()
    mime = _detect_image_mime(data)
    prov = {'boundary': 'pi_workspace_read', 'started_at': start, 'finished_at': utc_now(), 'quiescent': True}
    if mime is not None:
        if len(data) > max_image_bytes:
            return _tool_error_result(
                f'Image file {raw_path} ({format_size(len(data))}) exceeds max_image_bytes ({format_size(max_image_bytes)})',
                'pi_workspace_read',
                start,
            )
        note = f'Read image file [{mime}]'
        if not supports_images:
            note += '\n[Current model does not support images. The image will be omitted from this request.]'
        obs = {
            'output': note,
            'stdout': note,
            'stderr': '',
            'path': rel_posix,
            'mime_type': mime,
            'size_bytes': len(data),
            'exit_code': 0,
            'stop_reason': None,
            'output_complete': True,
        }
        return Result(
            obs,
            note.encode('utf-8'),
            b'',
            image=data if supports_images else None,
            image_mime=mime if supports_images else None,
            provenance=prov,
        )

    text_content = data.decode('utf-8', 'replace')
    all_lines = text_content.split('\n')
    total_file_lines = len(all_lines)
    offset = call.arguments.get('offset')
    limit = call.arguments.get('limit')
    start_line = max(0, int(offset) - 1) if offset is not None else 0
    start_line_display = start_line + 1
    if start_line >= len(all_lines):
        return _tool_error_result(
            f'Offset {offset} is beyond end of file ({len(all_lines)} lines total)',
            'pi_workspace_read',
            start,
        )
    user_limited_lines = None
    if limit is not None:
        end_line = min(start_line + max(0, int(limit)), len(all_lines))
        selected_content = '\n'.join(all_lines[start_line:end_line])
        user_limited_lines = end_line - start_line
    else:
        selected_content = '\n'.join(all_lines[start_line:])
    truncation = truncate_head(selected_content)
    if truncation['firstLineExceedsLimit']:
        first_line_size = format_size(len(all_lines[start_line].encode('utf-8')))
        output_text = (
            f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
            f"Use bash: sed -n '{start_line_display}p' {raw_path} | head -c {DEFAULT_MAX_BYTES}]"
        )
    elif truncation['truncated']:
        end_line_display = start_line_display + truncation['outputLines'] - 1
        next_offset = end_line_display + 1
        output_text = truncation['content']
        if truncation['truncatedBy'] == 'lines':
            output_text += f'\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. Use offset={next_offset} to continue.]'
        else:
            output_text += (
                f'\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} '
                f'({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]'
            )
    elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
        remaining = len(all_lines) - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        output_text = f"{truncation['content']}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
    else:
        output_text = truncation['content']

    out_bytes = output_text.encode('utf-8')
    obs = {
        'output': output_text,
        'stdout': output_text,
        'stderr': '',
        'path': rel_posix,
        'exit_code': 0,
        'stop_reason': None,
        'output_complete': True,
    }
    return Result(obs, out_bytes, b'', provenance=prov)


def execute_pi_write(call, workspace: Path) -> Result:
    start = utc_now()
    raw_path = call.arguments['path']
    content = call.arguments['content']
    try:
        p, rel_posix = _resolve_workspace_target(workspace, raw_path)
    except L1Error as exc:
        return _tool_error_result(f'Could not write file {raw_path}: {exc}', 'pi_workspace_write', start)
    if p.exists() and p.is_dir():
        return _tool_error_result(f'EISDIR: illegal operation on a directory, open \'{raw_path}\'', 'pi_workspace_write', start)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode('utf-8')
    p.write_bytes(data)
    msg = f'Successfully wrote {len(content)} bytes to {raw_path}'
    obs = {
        'output': msg,
        'stdout': msg,
        'stderr': '',
        'path': rel_posix,
        'bytes_written': len(data),
        'exit_code': 0,
        'stop_reason': None,
        'output_complete': True,
    }
    prov = {'boundary': 'pi_workspace_write', 'started_at': start, 'finished_at': utc_now(), 'quiescent': True}
    return Result(obs, msg.encode('utf-8'), b'', provenance=prov)


def _detect_line_ending(content: str) -> str:
    crlf_idx = content.find('\r\n')
    lf_idx = content.find('\n')
    if lf_idx == -1 or crlf_idx == -1:
        return '\n'
    return '\r\n' if crlf_idx < lf_idx else '\n'


def _normalize_to_lf(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _restore_line_endings(text: str, ending: str) -> str:
    return text.replace('\n', '\r\n') if ending == '\r\n' else text


def _normalize_for_fuzzy_match(text: str) -> str:
    s = unicodedata.normalize('NFKC', text)
    s = '\n'.join(line.rstrip() for line in s.split('\n'))
    s = re.sub(r'[\u2018\u2019\u201A\u201B]', "'", s)
    s = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', s)
    s = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]', '-', s)
    s = re.sub(r'[\u00A0\u2002-\u200A\u202F\u205F\u3000]', ' ', s)
    return s


def _fuzzy_find_text(content: str, old_text: str) -> dict:
    exact_idx = content.find(old_text)
    if exact_idx != -1:
        return {
            'found': True,
            'index': exact_idx,
            'matchLength': len(old_text),
            'usedFuzzyMatch': False,
            'contentForReplacement': content,
        }
    fuzzy_content = _normalize_for_fuzzy_match(content)
    fuzzy_old = _normalize_for_fuzzy_match(old_text)
    fuzzy_idx = fuzzy_content.find(fuzzy_old)
    if fuzzy_idx == -1:
        return {
            'found': False,
            'index': -1,
            'matchLength': 0,
            'usedFuzzyMatch': False,
            'contentForReplacement': content,
        }
    return {
        'found': True,
        'index': fuzzy_idx,
        'matchLength': len(fuzzy_old),
        'usedFuzzyMatch': True,
        'contentForReplacement': fuzzy_content,
    }


def _apply_edits_to_normalized_content(normalized_content: str, edits: list[dict], path: str) -> tuple[str, str]:
    norm_edits = [
        {'oldText': _normalize_to_lf(e['oldText']), 'newText': _normalize_to_lf(e['newText'])}
        for e in edits
    ]
    total = len(norm_edits)
    for i, e in enumerate(norm_edits):
        if not e['oldText']:
            if total == 1:
                raise ValueError(f'oldText must not be empty in {path}.')
            raise ValueError(f'edits[{i}].oldText must not be empty in {path}.')
    initial_matches = [_fuzzy_find_text(normalized_content, e['oldText']) for e in norm_edits]
    base_content = (
        _normalize_for_fuzzy_match(normalized_content)
        if any(m['usedFuzzyMatch'] for m in initial_matches)
        else normalized_content
    )
    fuzzy_base = _normalize_for_fuzzy_match(base_content)
    matched_edits = []
    for i, e in enumerate(norm_edits):
        m = _fuzzy_find_text(base_content, e['oldText'])
        if not m['found']:
            if total == 1:
                raise ValueError(
                    f'Could not find the exact text in {path}. '
                    f'The old text must match exactly including all whitespace and newlines.'
                )
            raise ValueError(
                f'Could not find edits[{i}] in {path}. '
                f'The oldText must match exactly including all whitespace and newlines.'
            )
        fuzzy_old = _normalize_for_fuzzy_match(e['oldText'])
        occurrences = len(fuzzy_base.split(fuzzy_old)) - 1
        if occurrences > 1:
            if total == 1:
                raise ValueError(
                    f'Found {occurrences} occurrences of the text in {path}. '
                    f'The text must be unique. Please provide more context to make it unique.'
                )
            raise ValueError(
                f'Found {occurrences} occurrences of edits[{i}] in {path}. '
                f'Each oldText must be unique. Please provide more context to make it unique.'
            )
        matched_edits.append({
            'editIndex': i,
            'matchIndex': m['index'],
            'matchLength': m['matchLength'],
            'newText': e['newText'],
        })
    matched_edits.sort(key=lambda x: x['matchIndex'])
    for i in range(1, len(matched_edits)):
        prev = matched_edits[i - 1]
        curr = matched_edits[i]
        if prev['matchIndex'] + prev['matchLength'] > curr['matchIndex']:
            raise ValueError(
                f"edits[{prev['editIndex']}] and edits[{curr['editIndex']}] overlap in {path}. "
                f'Merge them into one edit or target disjoint regions.'
            )
    new_content = base_content
    for item in reversed(matched_edits):
        idx = item['matchIndex']
        length = item['matchLength']
        new_content = new_content[:idx] + item['newText'] + new_content[idx + length:]
    if base_content == new_content:
        if total == 1:
            raise ValueError(
                f'No changes made to {path}. The replacement produced identical content. '
                f'This might indicate an issue with special characters or the text not existing as expected.'
            )
        raise ValueError(f'No changes made to {path}. The replacements produced identical content.')
    return base_content, new_content


def execute_pi_edit(call, workspace: Path) -> Result:
    start = utc_now()
    norm_args = _prepare_edit_arguments(call.arguments)
    raw_path = norm_args['path']
    edits = norm_args['edits']
    try:
        p, rel_posix = _resolve_workspace_target(workspace, raw_path)
    except L1Error as exc:
        return _tool_error_result(f'Could not edit file: {raw_path}. {exc}.', 'pi_workspace_edit', start)
    if not p.exists() or not p.is_file():
        return _tool_error_result(f'Could not edit file: {raw_path}. Error code: ENOENT.', 'pi_workspace_edit', start)
    regular_file(p)
    raw_content = p.read_bytes().decode('utf-8', 'replace')
    bom = '\ufeff' if raw_content.startswith('\ufeff') else ''
    content = raw_content[1:] if bom else raw_content
    orig_ending = _detect_line_ending(content)
    norm_content = _normalize_to_lf(content)
    try:
        base_content, new_content = _apply_edits_to_normalized_content(norm_content, edits, raw_path)
    except ValueError as exc:
        return _tool_error_result(str(exc), 'pi_workspace_edit', start)
    final_content = bom + _restore_line_endings(new_content, orig_ending)
    p.write_bytes(final_content.encode('utf-8'))
    diff_lines = list(
        difflib.unified_diff(
            base_content.splitlines(),
            new_content.splitlines(),
            fromfile=raw_path,
            tofile=raw_path,
            lineterm='',
            n=4,
        )
    )
    msg = f'Successfully replaced {len(edits)} block(s) in {raw_path}.'
    obs = {
        'output': msg,
        'stdout': msg,
        'stderr': '',
        'path': rel_posix,
        'diff': '\n'.join(diff_lines),
        'exit_code': 0,
        'stop_reason': None,
        'output_complete': True,
    }
    prov = {'boundary': 'pi_workspace_edit', 'started_at': start, 'finished_at': utc_now(), 'quiescent': True}
    return Result(obs, msg.encode('utf-8'), b'', provenance=prov)


def format_pi_bash_observation(
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    exit_code: int | None,
    stop_reason: str | None,
    output_complete: bool,
    oom_killed: bool | None,
) -> dict:
    stdout_str = stdout_bytes.decode('utf-8', 'replace')
    stderr_str = stderr_bytes.decode('utf-8', 'replace')
    combined = stdout_str if not stderr_str else (f'{stdout_str}\n{stderr_str}' if stdout_str else stderr_str)
    truncation = truncate_tail(combined) if combined else {
        'content': '(no output)',
        'truncated': False,
        'totalLines': 0,
        'outputLines': 0,
        'outputBytes': 0,
        'lastLinePartial': False,
        'truncatedBy': None,
    }
    text = truncation['content'] or '(no output)'
    if truncation.get('truncated'):
        start_line = truncation['totalLines'] - truncation['outputLines'] + 1
        end_line = truncation['totalLines']
        if truncation.get('lastLinePartial'):
            text += f"\n\n[Showing last {format_size(truncation['outputBytes'])} of line {end_line}]"
        elif truncation.get('truncatedBy') == 'lines':
            text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation['totalLines']}]"
        else:
            text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation['totalLines']} ({format_size(DEFAULT_MAX_BYTES)} limit)]"
    if stop_reason == 'timeout':
        text = f'{text}\n\nCommand timed out' if text else 'Command timed out'
    elif exit_code not in (0, None):
        text = f'{text}\n\nCommand exited with code {exit_code}' if text else f'Command exited with code {exit_code}'
    return {
        'output': text,
        'stdout': stdout_str,
        'stderr': stderr_str,
        'exit_code': exit_code,
        'stop_reason': stop_reason,
        'output_complete': output_complete,
        'oom_killed': oom_killed,
    }

