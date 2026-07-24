"""Generate the deterministic sample document used by DecodeVision."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "samples" / "decodevision_invoice.png"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def build_sample() -> Image.Image:
    image = Image.new("RGB", (1400, 900), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    navy = "#10233f"
    blue = "#2563eb"
    muted = "#596579"
    line = "#d7deea"

    draw.rounded_rectangle((70, 60, 1330, 840), radius=28, fill="white", outline=line, width=3)
    draw.rounded_rectangle((70, 60, 1330, 200), radius=28, fill=navy)
    draw.rectangle((70, 170, 1330, 200), fill=navy)
    draw.text((120, 98), "DECODEVISION LABS", font=_font(42, bold=True), fill="white")
    draw.text((1010, 105), "INVOICE", font=_font(36, bold=True), fill="#9cc1ff")

    draw.text((120, 245), "Invoice ID", font=_font(22, bold=True), fill=muted)
    draw.text((120, 278), "DV-2026-004", font=_font(30, bold=True), fill=navy)
    draw.text((620, 245), "Issued", font=_font(22, bold=True), fill=muted)
    draw.text((620, 278), "23 July 2026", font=_font(30), fill=navy)
    draw.text((1000, 245), "Status", font=_font(22, bold=True), fill=muted)
    draw.rounded_rectangle((1000, 278, 1195, 326), radius=22, fill="#dcfce7")
    draw.text((1033, 289), "PAID", font=_font(24, bold=True), fill="#15803d")

    draw.line((120, 370, 1280, 370), fill=line, width=3)
    draw.text((120, 400), "DESCRIPTION", font=_font(20, bold=True), fill=muted)
    draw.text((940, 400), "QTY", font=_font(20, bold=True), fill=muted)
    draw.text((1135, 400), "AMOUNT", font=_font(20, bold=True), fill=muted)
    rows = [
        ("OCR pipeline implementation", "1", "$ 480.00"),
        ("Confidence validation suite", "1", "$ 160.00"),
        ("Visual inspection workspace", "1", "$ 240.00"),
    ]
    for index, (description, quantity, amount) in enumerate(rows):
        y = 465 + index * 82
        draw.text((120, y), description, font=_font(25), fill=navy)
        draw.text((958, y), quantity, font=_font(25), fill=navy)
        draw.text((1135, y), amount, font=_font(25), fill=navy)
        draw.line((120, y + 52, 1280, y + 52), fill=line, width=2)

    draw.text((930, 720), "TOTAL", font=_font(24, bold=True), fill=muted)
    draw.text((1100, 710), "$ 880.00", font=_font(34, bold=True), fill=blue)
    draw.text(
        (120, 770),
        "Machine-readable sample · Confidence gate: 80%",
        font=_font(20),
        fill=muted,
    )
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_sample().save(OUTPUT, format="PNG", optimize=True)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
