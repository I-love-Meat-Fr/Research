"""
SDK Decryptor — thin wrapper around DecryptNir.exe
===================================================
Option A path: handles raw .jpg files produced by the ENG finger-vein SDK
(which prepends a 32-byte fixed header before encryption via zhy_fv401.dll).

The actual decryption is performed by the C# host `DecryptNir.exe`. This
module is just a Python facade that:

  * locates DecryptNir.exe,
  * runs it on a single file or a whole folder,
  * returns the path(s) to the decrypted JPEG(s).

It is intentionally side-effect free: it never imports the proprietary DLL
directly (only the .exe), so it works on any Windows host that has
DecryptNir.exe published.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ============================================================================
# Default path (Option A repo layout)
# ============================================================================

DEFAULT_DECRYPTOR_EXE = (
    r"D:\WorkSpace\Research\Research\MFFV\tools\DecryptNir\publish\DecryptNir.exe"
)


class SDKDecryptor:
    """
    Thin subprocess wrapper for DecryptNir.exe.

    Usage::

        d = SDKDecryptor()                       # uses default .exe path
        out = d.decrypt_file(Path("10/jpg/m1.jpg"))      # → Path("..._dec.jpg")
        d.decrypt_folder(Path("model/10"), Path("tmp/10"))  # batch
    """

    def __init__(
        self,
        exe_path: str | Path = DEFAULT_DECRYPTOR_EXE,
        timeout_s: int = 30,
    ):
        self.exe_path = Path(exe_path)
        if not self.exe_path.exists():
            raise FileNotFoundError(
                f"DecryptNir.exe not found at: {self.exe_path}\n"
                "Build/publish it from tools/DecryptNir first."
            )
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def decrypt_file(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Decrypt one encrypted .jpg.

        Args:
            input_path:  path to the encrypted .jpg
            output_path: optional explicit output path; if None, a sibling
                         "<stem>_dec.jpg" is used.

        Returns:
            Path to the decrypted JPEG.
        """
        in_path = Path(input_path).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        if output_path is None:
            out_path = in_path.with_name(f"{in_path.stem}_dec.jpg")
        else:
            out_path = Path(output_path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [str(self.exe_path), str(in_path), str(out_path)]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )

        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"DecryptNir failed for {in_path.name} "
                f"(rc={proc.returncode})\n"
                f"stdout: {proc.stdout}\n"
                f"stderr: {proc.stderr}"
            )

        return out_path

    # ------------------------------------------------------------------
    # Folder (batch)
    # ------------------------------------------------------------------

    def decrypt_folder(
        self,
        input_root: str | Path,
        output_root: Optional[str | Path] = None,
    ) -> list[Path]:
        """
        Recursively decrypt every m*.jpg under input_root.

        Args:
            input_root:  root folder (e.g. model/10)
            output_root: root for decrypted files; if None, a sibling
                         "<stem>_decoded" is used.

        Returns:
            list of paths to successfully decrypted JPEGs.
        """
        in_root = Path(input_root).resolve()
        if not in_root.is_dir():
            raise NotADirectoryError(f"Not a folder: {in_root}")

        if output_root is None:
            out_root = in_root.with_name(f"{in_root.name}_decoded")
        else:
            out_root = Path(output_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        cmd = [str(self.exe_path), str(in_root), str(out_root)]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_s * 10,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"DecryptNir batch failed (rc={proc.returncode})\n"
                f"stderr: {proc.stderr}"
            )

        return sorted(out_root.rglob("*.jpg"))

    # ------------------------------------------------------------------
    # Pillow loader (handy for app.py)
    # ------------------------------------------------------------------

    @staticmethod
    def open_decrypted_as_pil(decrypted_path: str | Path):
        """Open a decrypted JPEG with Pillow (lazy import to avoid cost)."""
        from PIL import Image

        img = Image.open(decrypted_path)
        if img.mode != "L":
            img = img.convert("L")
        return img


def is_encrypted_sdk_jpg(path: str | Path) -> bool:
    """
    Quick sniff for the 32-byte fixed header used by the SDK.

    Useful when a user uploads a file whose origin we don't know.
    """
    p = Path(path)
    try:
        with p.open("rb") as f:
            head = f.read(32)
    except OSError:
        return False

    fixed = bytes(
        [
            0x2D, 0x6A, 0x00, 0xDD, 0xD7, 0xC5, 0xAE, 0xD7,
            0x47, 0x0E, 0xFF, 0x72, 0x62, 0x52, 0x91, 0xB8,
            0xEE, 0x07, 0x94, 0x78, 0xFB, 0x1E, 0x75, 0x5E,
            0x0E, 0x4A, 0x8D, 0x2D, 0x87, 0xB8, 0xB5, 0x82,
        ]
    )
    return head == fixed


if __name__ == "__main__":
    # CLI smoke-test: `python services/sdk_decryptor.py <in.jpg>`
    if len(sys.argv) < 2:
        print("Usage: python sdk_decryptor.py <input.jpg> [output.jpg]")
        sys.exit(1)

    d = SDKDecryptor()
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    result = d.decrypt_file(inp, out)
    print(f"OK → {result}")
