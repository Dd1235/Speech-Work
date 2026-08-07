"""Extract text, tables, and images from the two notebooks into report_apr22/."""
import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent
ASSETS_DIR = OUT_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOKS = [
    ("nb1_raw", ROOT / "dbscan_4phone_classes_dim12.ipynb",
     "Notebook 1: DBSCAN on raw 4-class MFCC frames"),
    ("nb2_mcmc", ROOT / "dbscan_4phone_mcmc_gmm_dim12.ipynb",
     "Notebook 2: DBSCAN on MCMC-sampled GMM cluster cores"),
]

# Map (notebook tag, zero-based code-cell ordinal) -> figure filename stem
FIGURE_NAMES = {
    ("nb1_raw", 3): "nb1_truth_tsne",
    ("nb1_raw", 4): "nb1_kdistance",
    ("nb1_raw", 8): "nb1_truth_vs_dbscan_tsne",
    ("nb2_mcmc", 4): "nb2_real_vs_core_tsne",
    ("nb2_mcmc", 5): "nb2_core_kdistance",
    ("nb2_mcmc", 8): "nb2_core_truth_vs_dbscan_tsne",
}


def html_table_to_markdown(html_text: str) -> str:
    """Very small HTML table -> Markdown converter good enough for pandas outputs."""
    text = html_text
    rows = re.findall(r"<tr>(.*?)</tr>", text, flags=re.S)
    md_rows = []
    header_written = False
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)
        if not cells:
            continue
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        clean = [html.unescape(c).replace("\n", " ") for c in clean]
        md_rows.append("| " + " | ".join(clean) + " |")
        if not header_written:
            md_rows.append("| " + " | ".join(["---"] * len(clean)) + " |")
            header_written = True
    return "\n".join(md_rows)


def extract_notebook(tag: str, path: Path, title: str) -> str:
    nb = json.loads(path.read_text())
    parts = [f"# {title}\n\nSource: `{path.name}`\n"]
    code_ordinal = -1
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            md = "".join(cell["source"]).rstrip()
            parts.append(f"\n{md}\n")
            continue
        if cell["cell_type"] != "code":
            continue
        code_ordinal += 1
        src = "".join(cell["source"]).rstrip()
        parts.append(f"\n### Cell {idx} (code #{code_ordinal})\n")
        parts.append("```python\n" + src + "\n```\n")

        for out_i, out in enumerate(cell.get("outputs", [])):
            t = out.get("output_type")
            if t == "stream":
                text = "".join(out.get("text", [])).rstrip()
                if text:
                    parts.append("\n**stdout:**\n\n```\n" + text + "\n```\n")
            elif t in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "image/png" in data:
                    stem = FIGURE_NAMES.get((tag, code_ordinal),
                                            f"{tag}_cell{idx:02d}_{out_i}")
                    fname = f"{stem}.png"
                    (ASSETS_DIR / fname).write_bytes(
                        base64.b64decode(data["image/png"]))
                    parts.append(f"\n![{stem}](assets/{fname})\n")
                elif "text/html" in data:
                    html_text = "".join(data["text/html"])
                    md_table = html_table_to_markdown(html_text)
                    if md_table:
                        parts.append("\n" + md_table + "\n")
                    else:
                        text = "".join(data.get("text/plain", [])).rstrip()
                        if text:
                            parts.append("\n```\n" + text + "\n```\n")
                elif "text/plain" in data:
                    text = "".join(data["text/plain"]).rstrip()
                    if text:
                        parts.append("\n```\n" + text + "\n```\n")
    return "\n".join(parts)


def main() -> None:
    sections = []
    for tag, path, title in NOTEBOOKS:
        sections.append(extract_notebook(tag, path, title))

    md_path = OUT_DIR / "report_dump.md"
    md_path.write_text("\n\n---\n\n".join(sections))
    print(f"Wrote {md_path}")

    pngs = sorted(ASSETS_DIR.glob("*.png"))
    print(f"Extracted {len(pngs)} images to {ASSETS_DIR}:")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
