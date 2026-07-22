# /// script
# requires-python = ">=3.11"
# dependencies = ["qrcode[pil]>=8.0,<9"]
# ///
"""Generate the print-safe repository QR code used on the book's final leaf."""

from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "assets" / "verbx_github_qr.png"
REPOSITORY_URL = "https://github.com/TheColby/verbx"


def main() -> None:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=24,
        border=4,
    )
    qr.add_data(REPOSITORY_URL)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white").convert("1")
    image.save(OUTPUT, optimize=True)
    print(f"Wrote {OUTPUT} for {REPOSITORY_URL}")


if __name__ == "__main__":
    main()
