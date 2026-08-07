"""Extract text, tables, and images from dbscan_40phone_mcmc_gmm_dim12.ipynb into report_apr29/.

Run after the notebook has been executed end-to-end so cell outputs are populated.
"""
import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent
ASSETS_DIR = OUT_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOK = ROOT / "dbscan_40phone_mcmc_gmm_dim12.ipynb"
TITLE = "Notebook: 40-phone GMM + MCMC + DBSCAN + Support Set"

# Map cell index (0-based, matching nbformat) -> figure filename stem.
# Stable across runs as long as cell order does not change.
FIGURE_NAMES = {
    11: "real_vs_core_tsne_tau15",       # initial τ=1.5 sanity check
    13: "core_kdistance",                 # k-distance for DBSCAN
    18: "dbscan_clusters_tau15_tsne",     # DBSCAN @ τ=1.5: ground truth vs cluster
    24: "support_set_tsne",               # support frames overlaid on real-frame t-SNE
    31: "dbscan_clusters_besttau_tsne",   # DBSCAN @ best τ: truth vs cluster (pre-labels)
    36: "confusion_matrix",               # cluster × phone heatmap (best τ)
    38: "labeled_cluster_tsne",           # phone-name labeled clusters (best τ)
    40: "tau_progression",                # NMI/ARI/clusters/noise vs τ
    42: "pairwise_mean_distance",         # phone × phone Euclidean μ-distance heatmap
}


def html_table_to_markdown(html_text: str) -> str:
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


def extract_notebook(path: Path, title: str) -> str:
    nb = json.loads(path.read_text())
    parts = [f"# {title}\n\nSource: `{path.name}`\n"]
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            md = "".join(cell["source"]).rstrip()
            parts.append(f"\n{md}\n")
            continue
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]).rstrip()
        parts.append(f"\n### Cell {idx}\n")
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
                    stem = FIGURE_NAMES.get(idx, f"cell{idx:02d}_{out_i}")
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
    md = extract_notebook(NOTEBOOK, TITLE)
    md_path = OUT_DIR / "report_dump.md"
    md_path.write_text(md)
    print(f"wrote {md_path}")

    pngs = sorted(ASSETS_DIR.glob("*.png"))
    print(f"extracted {len(pngs)} images to {ASSETS_DIR}:")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
