"""Extract embedded base64 PNG figures from the docling markdown into assets/."""
import base64
import re
from pathlib import Path

SRC = Path("EBSCO-FullText-13_05_2026.md")
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

text = SRC.read_text(encoding="utf-8")

# Match ![caption](data:image/png;base64,XXXX)
pattern = re.compile(r"!\[([^\]]*)\]\(data:image/png;base64,([A-Za-z0-9+/=]+)\)")

matches = pattern.findall(text)
print(f"Found {len(matches)} images")

# Order in the document: [logo, Fig1, Fig2, Fig3, Fig4]
names = ["logo", "fig1_peak_demand", "fig2_features", "fig3_sma", "fig4_comparison"]

for i, (cap, b64) in enumerate(matches):
    name = names[i] if i < len(names) else f"image_{i}"
    path = OUT / f"{name}.png"
    path.write_bytes(base64.b64decode(b64))
    print(f"  -> {path} ({path.stat().st_size} bytes) cap={cap!r}")

print("Done.")
