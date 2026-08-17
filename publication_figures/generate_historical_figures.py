#!/usr/bin/env python3
"""Generate publication figures from the completed SET-M33 post-WEKA audit.

The script does not refit the surrogate, recompute SHAP, or rerun UMAP. It uses
the archived reference-importance tables and historical Euclidean UMAP
coordinates produced by the v2.1 HPC workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from PIL import Image


DEFAULT_RESULTS_ROOT = (
    Path.home()
    / "Desktop"
    / "shap_umap"
    / "SET_M33_post_WEKA_results_review_v2.1"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "Desktop" / "shap_umap" / "SET_M33_publication_figures_v2.1"
)

SPECIES = {
    "ecoli": {
        "display": r"$\it{E.\ coli}$ TG1",
        "plain": "E. coli TG1",
        "panel": "A",
        "keys": ["ecoli_4plus", "ecoli_5plus", "ecoli_6plus", "ecoli_7plus"],
    },
    "paeruginosa": {
        "display": r"$\it{P.\ aeruginosa}$ PAO-1",
        "plain": "P. aeruginosa PAO-1",
        "panel": "B",
        "keys": [
            "paeruginosa_4plus",
            "paeruginosa_5plus",
            "paeruginosa_6plus",
            "paeruginosa_7plus",
        ],
    },
    "kpneumoniae": {
        "display": r"$\it{K.\ pneumoniae}$ ATCC 13833",
        "plain": "K. pneumoniae ATCC 13833",
        "panel": "C",
        "keys": [
            "kpneumoniae_4plus",
            "kpneumoniae_5plus",
            "kpneumoniae_6plus",
        ],
    },
}

KEY_TO_SPECIES = {
    key: species for species, meta in SPECIES.items() for key in meta["keys"]
}
KEY_TO_CHARGE = {
    key: int(key.rsplit("_", 1)[1].replace("plus", "")) for key in KEY_TO_SPECIES
}

SHAP_PINK = "#C83E6B"
GRID_COLOR = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Extracted v2.1 review directory containing results/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory for figures and source-data tables.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--heatmap-n", type=int, default=10)
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Nimbus Roman", "Times New Roman"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def resolve_results_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "results").is_dir():
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and (p / "results").is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not locate a results/ directory under {path}. "
        "Pass --results-root explicitly."
    )


def validate_inputs(root: Path) -> None:
    aggregation = root / "results" / "summary" / "aggregation.json"
    if not aggregation.exists():
        raise FileNotFoundError(f"Missing {aggregation}")
    status = json.loads(aggregation.read_text(encoding="utf-8"))
    if status.get("partial") is not False or status.get("status") != "completed":
        raise RuntimeError(f"Aggregation is not complete: {status}")
    missing: list[Path] = []
    for key in KEY_TO_SPECIES:
        p = root / "results" / "shap_reference" / key / "reference_importance.csv"
        if not p.exists():
            missing.append(p)
    for species in SPECIES:
        p = root / "results" / "historical_umap" / species / "historical_embedding.csv"
        if not p.exists():
            missing.append(p)
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(map(str, missing)))


def prepare_directories(output_root: Path) -> dict[str, Path]:
    dirs = {
        "top20_individual": output_root / "figures" / "shap_top20" / "individual",
        "top20_composite": output_root / "figures" / "shap_top20" / "composite",
        "heatmap_individual": output_root / "figures" / "shap_heatmaps" / "individual",
        "heatmap_composite": output_root / "figures" / "shap_heatmaps" / "composite",
        "umap_individual": output_root / "figures" / "historical_umap" / "individual",
        "umap_composite": output_root / "figures" / "historical_umap" / "composite",
        "source_top20": output_root / "source_data" / "shap_top20",
        "source_heatmap": output_root / "source_data" / "shap_heatmaps",
        "source_umap": output_root / "source_data" / "historical_umap",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: mpl.figure.Figure, base: Path, dpi: int) -> list[Path]:
    written: list[Path] = []
    # Very large 600-dpi multi-panel PNGs are unnecessary because PDF/SVG are
    # also exported, and some PNG readers struggle with >7,000-pixel canvases.
    # Keep individual panels at 600 dpi while capping the longest PNG edge.
    longest_edge_inches = max(float(v) for v in fig.get_size_inches())
    png_dpi = min(dpi, max(150, int(4800 / longest_edge_inches)))
    for suffix in (".png", ".pdf", ".svg"):
        path = base.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = png_dpi
        # Render in a local temporary directory, validate, then copy. This
        # avoids partial PNG writes on synchronized/network filesystems.
        with tempfile.TemporaryDirectory(prefix="setm33_figure_") as tmp:
            temporary = Path(tmp) / path.name
            fig.savefig(temporary, **kwargs)
            if suffix == ".png":
                with Image.open(temporary) as image:
                    image.verify()
            shutil.copy2(temporary, path)
            if file_sha256(temporary) != file_sha256(path):
                raise OSError(f"Checksum mismatch while writing {path}")
            if suffix == ".png":
                with Image.open(path) as image:
                    image.verify()
        written.append(path)
    return written


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_importance(root: Path, key: str) -> pd.DataFrame:
    path = root / "results" / "shap_reference" / key / "reference_importance.csv"
    data = pd.read_csv(path)
    required = {"descriptor", "mean_abs_shap_full", "mean_signed_shap_full"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    data = data.copy()
    total = float(data["mean_abs_shap_full"].sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid total mean absolute SHAP in {path}")
    data["relative_mean_abs_shap"] = data["mean_abs_shap_full"] / total
    data["key"] = key
    data["species"] = KEY_TO_SPECIES[key]
    data["charge"] = KEY_TO_CHARGE[key]
    return data.sort_values("relative_mean_abs_shap", ascending=False).reset_index(drop=True)


def style_axis(ax: mpl.axes.Axes, grid_axis: str | None = "x") -> None:
    sns.despine(ax=ax)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def plot_top20_axis(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    title: str,
    top_n: int,
    compact: bool = False,
) -> pd.DataFrame:
    top = data.head(top_n).sort_values("relative_mean_abs_shap", ascending=True).copy()
    y = np.arange(len(top))
    ax.barh(y, top["relative_mean_abs_shap"], color=SHAP_PINK, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(top["descriptor"])
    ax.set_xlabel(r"Relative mean $|\mathrm{SHAP}|$")
    ax.set_title(title, pad=8)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    if compact:
        ax.tick_params(axis="y", labelsize=6.4)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel(r"Relative mean $|\mathrm{SHAP}|$", fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
    style_axis(ax, "x")
    return top.sort_values("relative_mean_abs_shap", ascending=False)


def generate_top20(
    root: Path,
    dirs: dict[str, Path],
    importance: dict[str, pd.DataFrame],
    top_n: int,
    dpi: int,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for key, data in importance.items():
        species = KEY_TO_SPECIES[key]
        charge = KEY_TO_CHARGE[key]
        title = f"{SPECIES[species]['display']}, charge +{charge}"
        fig, ax = plt.subplots(figsize=(7.2, 6.6))
        top = plot_top20_axis(ax, data, title, top_n)
        fig.subplots_adjust(left=0.47, right=0.98, bottom=0.12, top=0.92)
        base = dirs["top20_individual"] / f"shap_top{top_n}_{key}"
        paths = save_figure(fig, base, dpi)
        plt.close(fig)
        source = dirs["source_top20"] / f"shap_top{top_n}_{key}.csv"
        top.to_csv(source, index=False)
        for path in paths:
            manifest.append(
                figure_record(path, "SHAP top-20", key, source, "relative mean absolute SHAP")
            )

    for species, meta in SPECIES.items():
        keys = meta["keys"]
        if len(keys) == 4:
            fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
            axes_flat = axes.flat
        else:
            fig, axes = plt.subplots(1, 3, figsize=(19, 7.2))
            axes_flat = np.ravel(axes)
        for ax, key in zip(axes_flat, keys):
            charge = KEY_TO_CHARGE[key]
            plot_top20_axis(ax, importance[key], f"Charge +{charge}", top_n, compact=True)
        fig.suptitle(meta["display"], fontsize=13, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=2.5, h_pad=2.5)
        base = dirs["top20_composite"] / f"shap_top{top_n}_{species}_composite"
        paths = save_figure(fig, base, dpi)
        plt.close(fig)
        for path in paths:
            manifest.append(
                figure_record(path, "SHAP top-20 composite", species, None, "relative mean absolute SHAP")
            )
    return manifest


def build_heatmap_matrix(
    importance: dict[str, pd.DataFrame], species: str, heatmap_n: int
) -> pd.DataFrame:
    meta = SPECIES[species]
    series: dict[int, pd.Series] = {}
    for key in meta["keys"]:
        charge = KEY_TO_CHARGE[key]
        series[charge] = importance[key].set_index("descriptor")["relative_mean_abs_shap"]
    matrix = pd.concat(series, axis=1).fillna(0.0)
    matrix.columns = [int(c) for c in matrix.columns]
    rank = matrix.mean(axis=1).sort_values(ascending=False)
    return matrix.loc[rank.head(heatmap_n).index].copy()


def annotate_heatmap(ax: mpl.axes.Axes, matrix: pd.DataFrame, vmax: float) -> None:
    cmap = mpl.colormaps["magma"]
    norm = Normalize(vmin=0, vmax=vmax)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix.iloc[row, col])
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.58 else "white"
            ax.text(col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)


def draw_heatmap(
    ax: mpl.axes.Axes,
    matrix: pd.DataFrame,
    vmax: float,
    title: str,
    show_xlabel: bool = True,
) -> mpl.collections.QuadMesh:
    hm = sns.heatmap(
        matrix,
        ax=ax,
        cmap="magma",
        vmin=0,
        vmax=vmax,
        cbar=False,
        linewidths=0.35,
        linecolor=(1, 1, 1, 0.15),
        square=False,
    )
    annotate_heatmap(ax, matrix, vmax)
    ax.set_xticklabels([f"+{int(c)}" for c in matrix.columns], rotation=0)
    ax.set_yticklabels(matrix.index, rotation=0)
    ax.set_xlabel("Charge" if show_xlabel else "")
    ax.set_ylabel("")
    ax.set_title(title, pad=8)
    return hm.collections[0]


def generate_heatmaps(
    dirs: dict[str, Path],
    importance: dict[str, pd.DataFrame],
    heatmap_n: int,
    dpi: int,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    matrices = {
        species: build_heatmap_matrix(importance, species, heatmap_n)
        for species in SPECIES
    }
    vmax = max(float(matrix.to_numpy().max()) for matrix in matrices.values())
    for species, matrix in matrices.items():
        source = dirs["source_heatmap"] / f"shap_heatmap_top{heatmap_n}_{species}.csv"
        matrix.rename(columns=lambda x: f"+{x}").to_csv(source, index_label="descriptor")
        fig, ax = plt.subplots(figsize=(7.1, 5.5))
        mesh = draw_heatmap(ax, matrix, vmax, SPECIES[species]["display"])
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label(r"Relative mean $|\mathrm{SHAP}|$")
        fig.subplots_adjust(left=0.46, right=0.91, bottom=0.13, top=0.91)
        base = dirs["heatmap_individual"] / f"shap_heatmap_top{heatmap_n}_{species}"
        paths = save_figure(fig, base, dpi)
        plt.close(fig)
        for path in paths:
            manifest.append(
                figure_record(path, "SHAP heatmap", species, source, "relative mean absolute SHAP")
            )

    fig = plt.figure(figsize=(7.7, 12.4))
    gs = fig.add_gridspec(3, 1, left=0.42, right=0.84, top=0.975, bottom=0.055, hspace=0.42)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    mesh = None
    for ax, (species, matrix) in zip(axes, matrices.items()):
        mesh = draw_heatmap(ax, matrix, vmax, SPECIES[species]["display"])
        ax.text(-0.16, 1.08, SPECIES[species]["panel"], transform=ax.transAxes, fontsize=15, fontweight="bold")
    cax = fig.add_axes((0.875, 0.20, 0.025, 0.60))
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"Relative mean $|\mathrm{SHAP}|$")
    base = dirs["heatmap_composite"] / "Figure_2_SHAP_heatmaps"
    paths = save_figure(fig, base, dpi)
    plt.close(fig)
    for path in paths:
        manifest.append(
            figure_record(path, "Figure 2 SHAP heatmaps", "all species", None, "relative mean absolute SHAP")
        )
    return manifest


def load_umap(root: Path, species: str) -> pd.DataFrame:
    path = root / "results" / "historical_umap" / species / "historical_embedding.csv"
    data = pd.read_csv(path)
    required = {"ID", "cationic_charge", "UMAP1", "UMAP2"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    data = data.copy()
    data["cationic_charge"] = data["cationic_charge"].astype(int)
    return data


def scatter_size(n_rows: int) -> float:
    return float(np.clip(18000 / max(n_rows, 1), 3.5, 12.0))


def draw_umap(ax: mpl.axes.Axes, data: pd.DataFrame, title: str) -> mpl.collections.PathCollection:
    norm = Normalize(vmin=4, vmax=7)
    scatter = ax.scatter(
        data["UMAP1"],
        data["UMAP2"],
        c=data["cationic_charge"],
        cmap="viridis",
        norm=norm,
        s=scatter_size(len(data)),
        alpha=0.80,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    return scatter


def generate_umaps(
    root: Path,
    dirs: dict[str, Path],
    dpi: int,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    embeddings = {species: load_umap(root, species) for species in SPECIES}
    for species, data in embeddings.items():
        source = dirs["source_umap"] / f"historical_umap_{species}.csv.gz"
        data.to_csv(source, index=False, compression="gzip")
        fig, ax = plt.subplots(figsize=(7.2, 6.1))
        scatter = draw_umap(ax, data, SPECIES[species]["display"])
        cbar = fig.colorbar(scatter, ax=ax, ticks=[4, 5, 6, 7], fraction=0.046, pad=0.035)
        cbar.set_label("Charge (+)")
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.10, top=0.91)
        base = dirs["umap_individual"] / f"historical_umap_{species}"
        paths = save_figure(fig, base, dpi)
        plt.close(fig)
        for path in paths:
            manifest.append(
                figure_record(path, "Historical grouped Euclidean UMAP", species, source, "cationic charge")
            )

    fig = plt.figure(figsize=(11.4, 10.5))
    gs = fig.add_gridspec(2, 4, left=0.065, right=0.89, top=0.965, bottom=0.065, wspace=0.38, hspace=0.30)
    axes = {
        "ecoli": fig.add_subplot(gs[0, 0:2]),
        "paeruginosa": fig.add_subplot(gs[0, 2:4]),
        "kpneumoniae": fig.add_subplot(gs[1, 1:3]),
    }
    scatter = None
    for species, ax in axes.items():
        scatter = draw_umap(ax, embeddings[species], SPECIES[species]["display"])
        ax.text(-0.075, 1.06, SPECIES[species]["panel"], transform=ax.transAxes, fontsize=15, fontweight="bold")
    cax = fig.add_axes((0.92, 0.20, 0.018, 0.60))
    cbar = fig.colorbar(scatter, cax=cax, ticks=[4, 5, 6, 7])
    cbar.set_label("Charge (+)")
    base = dirs["umap_composite"] / "Figure_3_historical_grouped_euclidean_UMAP"
    paths = save_figure(fig, base, dpi)
    plt.close(fig)
    for path in paths:
        manifest.append(
            figure_record(path, "Figure 3 historical grouped Euclidean UMAP", "all species", None, "cationic charge")
        )
    return manifest


def figure_record(
    figure: Path,
    figure_type: str,
    scope: str,
    source: Path | None,
    encoded_variable: str,
) -> dict[str, str]:
    return {
        "figure": str(figure),
        "figure_type": figure_type,
        "scope": scope,
        "source_data": "" if source is None else str(source),
        "encoded_variable": encoded_variable,
    }


def write_captions(output_root: Path) -> None:
    text = """# Draft figure captions

## Figure 2. SHAP feature importance across charge states

Heatmaps show the ten descriptors with the highest mean relative absolute SHAP
importance across charge states for each bacterial target: (A) *E. coli* TG1,
(B) *P. aeruginosa* PAO-1, and (C) *K. pneumoniae* ATCC 13833. Within each
charge-specific surrogate, mean absolute SHAP values were divided by their sum
over all descriptors; absent descriptor/charge combinations are displayed as
zero. SHAP importance quantifies the contribution of a descriptor to the
Python Random Forest surrogate of the archived WEKA `predicted` values and
does not establish biological causality or directionality.

## Figure 3. Historical grouped Euclidean UMAP

Historical UMAP projection of grouped SHAP values for each bacterial target:
(A) *E. coli* TG1, (B) *P. aeruginosa* PAO-1, and (C) *K. pneumoniae* ATCC
13833. Each point represents one candidate analog and is colored by cationic
charge. Descriptors were grouped from their charge-dependent mean signed SHAP
dynamics by KMeans (six groups for *E. coli*, six for *P. aeruginosa*, and
three for *K. pneumoniae*), and each analog was represented by the mean SHAP
value within each group. The grouped matrix was projected without additional
scaling by Euclidean UMAP (`n_neighbors=15`, `min_dist=0.1`, two components,
`random_state=42`) without subsampling. Axis units are arbitrary; cluster
geometry is shown descriptively and is not interpreted as evidence of
biological classes.

## Supplementary SHAP top-20 panels

The twenty descriptors with the highest relative mean absolute SHAP value are
shown for each target- and charge-specific surrogate. Values were normalized
to sum to one within each model to permit comparison of relative importance
across libraries. These plots describe the surrogate fitted to archived WEKA
predictions rather than the serialized WEKA trees.
"""
    (output_root / "CAPTIONS.md").write_text(text, encoding="utf-8")


def write_readme(output_root: Path, results_root: Path) -> None:
    text = f"""# SET-M33 publication figures v2.1

Generated from the completed post-WEKA robustness results. The default local
input directory is:

`{DEFAULT_RESULTS_ROOT}`

The figures use the archived reference SHAP importances and the precomputed
historical grouped Euclidean UMAP coordinates. No model or embedding is refit.

## Contents

- `figures/shap_top20/`: individual charge panels and species composites.
- `figures/shap_heatmaps/`: individual species panels and the Figure 2 composite.
- `figures/historical_umap/`: individual species panels and the Figure 3 composite.
- `source_data/`: exact plotting tables used for every figure.
- `figure_manifest.csv`: figure-to-source mapping.
- `CAPTIONS.md`: draft manuscript captions with the corrected UMAP method.

Every figure is exported as high-resolution PNG, PDF and SVG. Individual PNG
panels use 600 dpi; multi-panel PNGs are capped at 4,800 pixels on the longest
edge because the PDF/SVG versions already preserve print-quality line work.
UMAP point layers are rasterized inside PDF/SVG to keep file sizes manageable;
axes and text remain vector elements.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def write_reproduction_readme(project_dir: Path) -> None:
    text = """# Generate the SET-M33 historical figures locally

The script defaults to the directories requested for the local workstation:

```bash
cd /home/ismael/Desktop/shap_umap
python generate_historical_figures.py
```

Explicit invocation:

```bash
python generate_historical_figures.py \\
  --results-root /home/ismael/Desktop/shap_umap/SET_M33_post_WEKA_results_review_v2.1 \\
  --output-root /home/ismael/Desktop/shap_umap/SET_M33_publication_figures_v2.1
```

Required packages: `numpy`, `pandas`, `matplotlib`, and `seaborn`. UMAP and SHAP
are not required because the script uses the archived coordinates and
importance tables.
"""
    (project_dir / "README_generate_figures.md").write_text(text, encoding="utf-8")


def copy_generator(project_dir: Path, output_root: Path) -> None:
    source = Path(__file__).resolve()
    destination = (output_root / source.name).resolve()
    if source != destination:
        shutil.copy2(source, destination)

    readme = (project_dir / "README_generate_figures.md").resolve()
    destination = (output_root / readme.name).resolve()
    if readme.exists() and readme != destination:
        shutil.copy2(readme, destination)


def main() -> None:
    args = parse_args()
    if args.top_n < 1 or args.heatmap_n < 1:
        raise ValueError("--top-n and --heatmap-n must be positive")
    configure_style()
    root = resolve_results_root(args.results_root)
    validate_inputs(root)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = prepare_directories(output_root)

    importance = {key: load_importance(root, key) for key in KEY_TO_SPECIES}
    manifest: list[dict[str, str]] = []
    manifest.extend(generate_top20(root, dirs, importance, args.top_n, args.png_dpi))
    manifest.extend(generate_heatmaps(dirs, importance, args.heatmap_n, args.png_dpi))
    manifest.extend(generate_umaps(root, dirs, args.png_dpi))

    write_captions(output_root)
    write_readme(output_root, root)
    project_dir = Path(__file__).resolve().parent
    write_reproduction_readme(project_dir)
    copy_generator(project_dir, output_root)

    for record in manifest:
        for field in ("figure", "source_data"):
            if record[field]:
                record[field] = str(Path(record[field]).relative_to(output_root))

    manifest_path = output_root / "figure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Generated {len(manifest)} figure files in {output_root}")
    print("Historical Figure 2:", dirs["heatmap_composite"] / "Figure_2_SHAP_heatmaps.png")
    print("Historical Figure 3:", dirs["umap_composite"] / "Figure_3_historical_grouped_euclidean_UMAP.png")


if __name__ == "__main__":
    main()
