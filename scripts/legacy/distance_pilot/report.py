"""Render measured baseline results beside the existing point annotations."""

from pathlib import Path
import csv, html
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2] / "output/distance_pilot_20"
rows = list(csv.DictReader((root / "baseline.csv").open()))
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
items = []
for start in range(0, len(rows), 10):
    canvas = Image.new("RGB", (1280, 5 * 420), "white")
    draw = ImageDraw.Draw(canvas)
    for j, r in enumerate(rows[start : start + 10]):
        frame = f"{int(r['frame']):04d}"
        im = Image.open(root / "annotated" / f"{frame}.jpg")
        im.thumbnail((640, 360))
        x = j % 2 * 640
        y = j // 2 * 420
        canvas.paste(im, (x, y))
        pred = (
            f"{float(r['predicted_cm']):.2f} cm"
            if r["predicted_cm"]
            else "invalid (model mask)"
        )
        status = "SCORED" if r["label_qc_pass"] == "True" else "EXCLUDED: label QC"
        draw.text(
            (x + 8, y + 362),
            f'{frame} | reference: {float(r["reference_cm"]):.2f} cm | {status}',
            font=font,
            fill="black",
        )
        draw.text((x + 8, y + 388), f"Prediction: {pred}", font=font, fill="darkred")
    canvas.save(root / f"baseline_comparison_{start//10+1}.jpg")
print("Saved baseline_comparison_1.jpg and baseline_comparison_2.jpg")
