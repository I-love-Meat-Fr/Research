"""
ENG Finger Vein JPG decoder
============================

Decodes the ``.jpg`` files captured by the ENG finger-vein camera
(shipped as ``FingerVein_v3_dev.dll`` / ``thin_FingerVein.dll`` /
``zhy_fv401.dll`` in ``ENG-SDK-C++\\DEMO\\Release``).

What the camera does on capture (see ``FingerVeinV3.cpp::raw2Image``):

    1. Acquire a 320x240 grayscale image from the sensor.
    2. ``cv::imencode(".jpg", pic, jpgBuf)`` with quality=85 produces
       a standard JPEG buffer starting with ``FF D8 FF E0 ... FF D9``.
    3. The whole JPEG buffer is fed through ``ZHY_Encrypt``
       (``zhy_fv401.dll``), a 16-byte block cipher.
    4. The encrypted bytes are written to ``<person>/jpg/mN.jpg``.

Decryption is the inverse: ``ZHY_Decrypt`` recovers the original JPEG
bytes. The first 32 bytes of every encrypted file appear identical
across captures because the JPEG header that gets encrypted is itself
almost constant (same SOI / APP0 / DQT marker bytes). It is **not** a
separate header that should be stripped.

This script wraps a small x86 helper
(``DecryptNir.exe``) that loads ``zhy_fv401.dll`` (32-bit, cannot be
loaded from 64-bit Python directly) and runs ``ZHY_Decrypt`` on each
file. The helper then writes the recovered JPEG next to the source.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder that contains zhy_fv401.dll (32-bit VC++ runtime).
# The DLL must be loadable from this process, so the helper runs in-place.
SDK_RELEASE_DIR = Path(
    r"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release"
)

# Self-contained x86 .NET 8 tool built from DecryptNir\DecryptNir.csproj.
HELPER_EXE = Path(
    r"D:\WorkSpace\Research\Research\MFFV\tools\DecryptNir"
    r"\publish\DecryptNir.exe"
)

DEFAULT_INPUT = Path(r"D:\WorkSpace\Research\Research\model")
DEFAULT_OUTPUT = Path(r"D:\WorkSpace\Research\Research\decoded")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decode_file(src: Path, dst: Path) -> bool:
    """Run the helper on a single file. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(HELPER_EXE), str(src), str(dst)],
        cwd=str(SDK_RELEASE_DIR),  # so the DLL can find its dependencies
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"[FAIL] {src}: {proc.stderr.strip()}\n")
        return False
    return True


def decode_folder(src_root: Path, dst_root: Path) -> tuple[int, int]:
    """Recursively decrypt every *.jpg under src_root into dst_root."""
    ok = 0
    fail = 0
    jpg_files = sorted(src_root.rglob("*.jpg"))
    for jpg in jpg_files:
        rel = jpg.relative_to(src_root)
        out = dst_root / rel
        print(f"[{ok + fail + 1:5d}/{len(jpg_files)}] {jpg.name}")
        if decode_file(jpg, out):
            ok += 1
        else:
            fail += 1
    return ok, fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if not HELPER_EXE.exists():
        sys.stderr.write(
            f"helper exe not found at {HELPER_EXE}; "
            "build it with: dotnet publish -c Release -r win-x86 "
            "--self-contained true -p:PublishSingleFile=true\n"
        )
        return 2

    if len(argv) >= 3:
        src_root = Path(argv[1])
        dst_root = Path(argv[2])
    elif len(argv) == 2:
        src_root = Path(argv[1])
        dst_root = src_root.parent / (src_root.name + "_decoded")
    else:
        src_root = DEFAULT_INPUT
        dst_root = DEFAULT_OUTPUT

    if not src_root.exists():
        sys.stderr.write(f"input not found: {src_root}\n")
        return 1

    print(f"input  : {src_root}")
    print(f"output : {dst_root}")
    ok, fail = decode_folder(src_root, dst_root)
    print(f"\nDone. {ok} files decoded, {fail} failed.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
