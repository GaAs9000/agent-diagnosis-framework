#!/usr/bin/env python3
"""Build a clean, high-compression, cross-platform portable ZIP of deliverables_by_model.

Enforces UTF-8 flag (General Purpose Bit 11, 0x800) on all zip entries so that standard
unzip utilities and standard Python zipfile on Linux/macOS/Windows decode non-ASCII
filenames (e.g. Chinese names like 部门协调会记录.docx) correctly without garbling.
Excludes system noise like .DS_Store and __MACOSX.
Performs end-to-end verification by default-extracting to a temp directory and running
c3-l1 verify on all 9 evidence packages.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def create_portable_zip(source_dir: Path, output_zip: Path) -> dict:
    source_dir = Path(source_dir).resolve()
    output_zip = Path(output_zip).resolve()

    files_to_pack = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d not in ('__MACOSX', '.pytest_cache', '__pycache__')]
        for f in files:
            if f in ('.DS_Store', 'Thumbs.db') or f.startswith('._'):
                continue
            full_path = Path(root) / f
            arcname = 'deliverables_by_model/' + str(full_path.relative_to(source_dir).as_posix())
            files_to_pack.append((full_path, arcname))

    files_to_pack.sort(key=lambda x: x[1])

    temp_zip = output_zip.with_suffix('.tmp.zip')
    if temp_zip.exists():
        temp_zip.unlink()

    print(f"Packaging {len(files_to_pack)} files into {output_zip.name}...")
    start_time = time.time()

    with zipfile.ZipFile(temp_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full_path, arcname in files_to_pack:
            zinfo = zipfile.ZipInfo.from_file(full_path, arcname)
            # Crucial: Enforce General Purpose Bit 11 (0x800) for universal UTF-8 support
            zinfo.flag_bits |= 0x800
            zinfo.compress_type = zipfile.ZIP_DEFLATED

            with open(full_path, 'rb') as src:
                zf.writestr(zinfo, src.read(), compresslevel=9)

    temp_zip.replace(output_zip)
    duration = time.time() - start_time
    zip_size = output_zip.stat().st_size
    zip_sha256 = hash_file(output_zip)

    print(f"Packaged {len(files_to_pack)} entries in {duration:.2f}s.")
    print(f"Size: {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"SHA-256: {zip_sha256}")

    return {
        'zip_path': str(output_zip),
        'size_bytes': zip_size,
        'sha256': zip_sha256,
        'entry_count': len(files_to_pack),
    }


def verify_portable_zip(zip_path: Path, python_bin: Path) -> bool:
    print("\nStarting roundtrip verification of portable ZIP...")
    with tempfile.TemporaryDirectory(prefix="zip_verify_") as tmpdir:
        tmp_path = Path(tmpdir)
        # Test extraction using standard Python zipfile default settings (no custom encoding override)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_path)

        extracted_root = tmp_path / 'deliverables_by_model'
        assert extracted_root.exists(), "Extracted root deliverables_by_model does not exist!"

        # Check Chinese filenames
        chinese_files = list(extracted_root.glob("**/*部门协调会记录*.docx"))
        print(f"Found {len(chinese_files)} Chinese document paths after default extraction (expected 11).")
        assert len(chinese_files) == 11, f"Expected 11 Chinese files, got {len(chinese_files)}"

        # Run c3-l1 verify on all 9 packages
        packages = [
            '01_gemini-3.5-flash-lite/episode',
            '01_gemini-3.5-flash-lite/branch',
            '01_gemini-3.5-flash-lite/branch_stateful',
            '02_gemini-3.8-flash/episode',
            '03_gemini-3.1-pro-preview/episode',
            '03_gemini-3.1-pro-preview/branch',
            '04_Qwen2.5-32B-Instruct/episode',
            '04_Qwen2.5-32B-Instruct/branch',
            '05_Meta-Llama-3.1-8B-Instruct/episode',
        ]

        for pkg_rel in packages:
            pkg_path = extracted_root / pkg_rel
            assert pkg_path.exists(), f"Package missing: {pkg_rel}"
            cmd = [str(python_bin), '-m', 'c3_sandbox.l1.cli', 'verify', str(pkg_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"FAILED verification on {pkg_rel}: {proc.stderr}")
                return False
            print(f"  [PASS] {pkg_rel}")

    print("\nROUNDTRIP VERIFICATION PASSED: 9/9 packages verified clean from default extracted zip!")
    return True


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[3]  # L1_sandbox
    source = base_dir / 'deliverables_by_model'
    target = base_dir / 'deliverables_by_model.zip'
    venv_py = base_dir / 'C3_Experimental_Sandbox/.venv/bin/python'

    info = create_portable_zip(source, target)
    ok = verify_portable_zip(target, venv_py)
    if not ok:
        sys.exit(1)
