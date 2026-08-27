from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-scientific Qwen smoke fixture")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "smoke.png"
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 170, 170), fill="red")
    draw.ellipse((342, 30, 482, 170), fill="green")
    draw.polygon(((100, 360), (30, 482), (170, 482)), fill="yellow")
    draw.rectangle((226, 226, 286, 286), fill="blue")
    image.save(image_path)
    manifest_path = args.output_dir / "manifest.jsonl"
    record = {
        "state_id": "synthetic-smoke-000",
        "image_id": "synthetic-smoke-image",
        "source_id": "synthetic-smoke-source",
        "image_path": image_path.name,
        "question": (
            "What color is the small square at the center? "
            "(A) red (B) blue (C) green (D) yellow. "
            "Answer with the option letter only."
        ),
        "target": "B",
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
