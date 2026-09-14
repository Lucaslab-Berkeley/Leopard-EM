"""Build a slide deck summarising where the microtubule 2DTM work stands.

Companion to STATUS.md, which keeps the detail. This keeps the argument: the strategy,
then the four readouts (polarity, protofilament number, lattice spacing, seam) and how
far each one has got.

Every panel is plotted from data on disk, not from numbers typed in, with two labelled
exceptions noted on the slides themselves.

Usage:  python make_status_deck.py [-o microtubule_2dtm_status.pdf]
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results_cropped"
MAPS = HERE / "maps"
PIXEL_SIZE = 0.9194

# Templates shown on the strategy slides. The ring is axis-centred and 14-fold
# pseudo-symmetric; the patch sits 86.8 A off the axis, which is what makes its
# roll angle measurable.
RING = "6dpu_atoms_6dpu_lattice_2rings_flatB_0.9194_bscale0.5.mrc"
PATCH = "GMPCPP_4patches_0.9194_bscale0.5.mrc"

FULL_MICROGRAPH = HERE / "Frames" / (
    "2025-05-23_14.39.26_25May23_GMPCPP1_26-94_0005_X-1Y+1-0_sum_DW.mrc"
)
CROP_MICROGRAPH = HERE / "Frames" / (
    "2025-05-23_14.39.26_25May23_GMPCPP1_26-94_0005_X-1Y+1-0_sum_DW_cropped_4.mrc"
)

CACHE = pathlib.Path(
    "/tmp/claude-2002/-home-jdickerson-git-LucasLab-Leopard-EM/"
    "4ef68f68-efb0-483e-978f-b9e47c263b75/scratchpad/det_full_z6.parquet"
)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
BLUE = "#1f4e79"
GREEN = "#2e7d32"
AMBER = "#ef6c00"
RED = "#c62828"
LIGHT = "#e8eef4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def slide(title: str, kicker: str = "") -> plt.Figure:
    """A blank 16:9 slide with a title block."""
    fig = plt.figure(figsize=(13.33, 7.5), dpi=200)
    fig.patch.set_facecolor("white")
    fig.text(0.045, 0.925, title, fontsize=25, fontweight="bold", color=BLUE)
    if kicker:
        fig.text(0.045, 0.877, kicker, fontsize=13, color=MUTED)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.855, 0.855], color=LIGHT, lw=2.5))
    return fig


def note(fig: plt.Figure, text: str, y: float = 0.055) -> None:
    fig.text(0.045, y, text, fontsize=10.5, color=MUTED, va="bottom", wrap=True)


def _scale_bar(ax, shape, length_nm: float, angstrom_per_display_px: float,
               label: str, colour: str = "white") -> None:
    """Draw a scale bar sized in the display pixels of an imshow panel."""
    length_px = length_nm * 10.0 / angstrom_per_display_px
    height, width = shape
    x0 = width - length_px - 0.04 * width
    y0 = height - 0.06 * height
    ax.add_patch(mpatches.Rectangle((x0, y0), length_px, 0.012 * height,
                                    facecolor=colour, edgecolor="black", lw=0.6))
    ax.text(x0 + length_px / 2, y0 - 0.02 * height, label, color=colour, fontsize=10,
            fontweight="bold", ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])


def template_projection(name: str, theta: float = 87.5, psi: float = 280.0,
                        phi: float = 0.0, crop: int = 330):
    """Bare projection of a simulated template, cropped for display.

    No CTF: the point is to show the model's own structure, not what the microscope
    does to it. Contrast is inverted so protein is dark, as in a micrograph.
    """
    import torch

    from leopard_em.utils.fourier_slice import get_real_space_projections_from_volume

    with mrcfile.open(MAPS / name, permissive=True) as handle:
        volume = torch.from_numpy(np.asarray(handle.data, dtype=np.float32).copy())
    projection = get_real_space_projections_from_volume(
        volume, torch.tensor(phi), torch.tensor(theta), torch.tensor(psi)
    ).numpy()
    centre = projection.shape[0] // 2
    half = crop // 2
    return projection[centre - half:centre + half, centre - half:centre + half]


def show_template(ax, image, title: str, scale_nm: float | None = 20.0) -> None:
    """Display a template projection with a consistent look."""
    v = 3.0 * image.std()
    ax.imshow(image, cmap="gray", vmin=image.mean() - v, vmax=image.mean() + v)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
    if scale_nm:
        _scale_bar(ax, image.shape, scale_nm, PIXEL_SIZE, f"{scale_nm:.0f} nm")


def bullet(ax, x: float, y: float, colour: str, label: str = "", size: int = 210) -> None:
    """A round marker in axes coordinates, immune to the axes aspect ratio."""
    ax.scatter([x], [y], s=size, color=colour, transform=ax.transAxes,
               clip_on=False, zorder=3)
    if label:
        ax.text(x, y, label, fontsize=10.5, fontweight="bold", color="white",
                ha="center", va="center", transform=ax.transAxes, zorder=4)


# --------------------------------------------------------------------------- data


def load_detections() -> pd.DataFrame:
    if not CACHE.exists():
        from leopard_em.pydantic_models.results.correlation_table import (
            detections_from_hdf5,
        )

        det = detections_from_hdf5(
            str(HERE / "results_full" / "output_correlation_table_2rings_full.h5"),
            min_z_score=6.0,
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        det.to_parquet(CACHE)
        return det
    return pd.read_parquet(CACHE)


def load_sites(detections: pd.DataFrame):
    from leopard_em.analysis.filament_lattice import (
        TemplateLatticeGeometry,
        extract_lattice_sites,
    )

    geometry = TemplateLatticeGeometry.from_pdb(
        str(HERE / "models" / "6dpu_2rings_aligned_zero.pdb")
    )
    return extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=8.0
    )


def load_competition() -> tuple[dict, np.ndarray]:
    """Per-site z for each cell of the 2x2, on a shared site list."""
    cells = [(a, b) for a in ("6dpu", "6dpv") for b in ("6dpu", "6dpv")]
    peaks, zmaps = {}, {}
    for atoms, lattice in cells:
        tag = f"{atoms}_atoms_{lattice}_lattice_2rings_flatB_crop"
        peaks[(atoms, lattice)] = pd.read_csv(RESULTS / f"results_{tag}.csv", index_col=0)
        with mrcfile.open(RESULTS / f"output_scaled_mip_{tag}.mrc", permissive=True) as h:
            zmaps[(atoms, lattice)] = np.asarray(h.data, dtype=np.float32)

    stacked = np.concatenate(
        [p[["pos_y", "pos_x", "scaled_mip"]].to_numpy() for p in peaks.values()]
    )
    stacked = stacked[np.argsort(-stacked[:, 2])]
    kept: list[np.ndarray] = []
    for row in stacked:
        if all(np.hypot(*(row[:2] - k[:2])) > 12.0 for k in kept):
            kept.append(row)
    sites = np.array([k[:2] for k in kept])

    z = {}
    for cell, zmap in zmaps.items():
        vals = np.empty(len(sites))
        for i, (y, x) in enumerate(sites.astype(int)):
            vals[i] = zmap[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].max()
        z[cell] = vals
    return z, sites


# --------------------------------------------------------------------------- data


def load_detections() -> pd.DataFrame:
    if not CACHE.exists():
        from leopard_em.pydantic_models.results.correlation_table import (
            detections_from_hdf5,
        )

        det = detections_from_hdf5(
            str(HERE / "results_full" / "output_correlation_table_2rings_full.h5"),
            min_z_score=6.0,
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        det.to_parquet(CACHE)
        return det
    return pd.read_parquet(CACHE)


def load_sites(detections: pd.DataFrame):
    from leopard_em.analysis.filament_lattice import (
        TemplateLatticeGeometry,
        extract_lattice_sites,
    )

    geometry = TemplateLatticeGeometry.from_pdb(
        str(HERE / "models" / "6dpu_2rings_aligned_zero.pdb")
    )
    return extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=8.0
    )


def load_competition() -> tuple[dict, np.ndarray]:
    """Per-site z for each cell of the 2x2, on a shared site list."""
    cells = [(a, b) for a in ("6dpu", "6dpv") for b in ("6dpu", "6dpv")]
    peaks, zmaps = {}, {}
    for atoms, lattice in cells:
        tag = f"{atoms}_atoms_{lattice}_lattice_2rings_flatB_crop"
        peaks[(atoms, lattice)] = pd.read_csv(RESULTS / f"results_{tag}.csv", index_col=0)
        with mrcfile.open(RESULTS / f"output_scaled_mip_{tag}.mrc", permissive=True) as h:
            zmaps[(atoms, lattice)] = np.asarray(h.data, dtype=np.float32)

    stacked = np.concatenate(
        [p[["pos_y", "pos_x", "scaled_mip"]].to_numpy() for p in peaks.values()]
    )
    stacked = stacked[np.argsort(-stacked[:, 2])]
    kept: list[np.ndarray] = []
    for row in stacked:
        if all(np.hypot(*(row[:2] - k[:2])) > 12.0 for k in kept):
            kept.append(row)
    sites = np.array([k[:2] for k in kept])

    z = {}
    for cell, zmap in zmaps.items():
        vals = np.empty(len(sites))
        for i, (y, x) in enumerate(sites.astype(int)):
            vals[i] = zmap[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].max()
        z[cell] = vals
    return z, sites


def template_contrast() -> list[tuple[str, float]]:
    from leopard_em.analysis.filament_lattice import (
        rise_from_template_autocorrelation,
        subunit_contrast,
    )

    named = [
        ("4-patch", "GMPCPP_4patches_0.9194_bscale0.5.mrc"),
        ("2-ring", "GMPCPP_2rings_0.9194_bscale0.5.mrc"),
        ("4-ring", "GMPCPP_4rings_0.9194_bscale0.5.mrc"),
        ("6DPU 2-ring\n(rebuilt)", "6dpu_atoms_6dpu_lattice_2rings_flatB_0.9194_bscale0.5.mrc"),
        ("6DPV 2-ring\n(rebuilt)", "6dpv_atoms_6dpv_lattice_2rings_flatB_0.9194_bscale0.5.mrc"),
    ]
    out = []
    for label, name in named:
        with mrcfile.open(MAPS / name, permissive=True) as handle:
            volume = np.asarray(handle.data, dtype=np.float32)
        rise = rise_from_template_autocorrelation(
            volume, approximate_rise_px=41.98 / PIXEL_SIZE, search_fraction=0.1
        )
        out.append((label, float(subunit_contrast(volume, 2 * rise))))
    return out


# -------------------------------------------------------------------------- slides


def slide_title(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(13.33, 7.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.70, "Microtubule structure from 2DTM", fontsize=38,
             fontweight="bold", color=BLUE)
    fig.text(0.08, 0.585, "Reading protofilament number, lattice spacing, seam and "
             "polarity\nout of a constrained template-matching search",
             fontsize=16, color=INK, linespacing=1.7)
    fig.add_artist(plt.Line2D([0.08, 0.52], [0.525, 0.525], color=LIGHT, lw=3))

    fig.text(0.08, 0.075, "One microtubule, one micrograph.  Leopard-EM, "
             "branch jd_constrained_segment.  Detail in MT_tests/STATUS.md",
             fontsize=10.5, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def slide_micrograph(pdf: PdfPages) -> None:
    """The data itself, low-passed into the CTF's first lobe and contrast-stretched."""
    from scipy.ndimage import gaussian_filter

    fig = slide("The data", "One microtubule, one 5760 x 4092 micrograph at 0.9194 A/px")

    with mrcfile.open(FULL_MICROGRAPH, permissive=True) as handle:
        full = np.asarray(handle.data, dtype=np.float32)

    # The CTF first zero is at 12.9 A here, so everything finer oscillates in sign and
    # only adds noise to a picture of a 25 nm tube. Stay inside the first lobe.
    lowpass_sigma = 20.0 / (2.355 * PIXEL_SIZE)
    smooth = gaussian_filter(full, lowpass_sigma)

    def stretch(img: np.ndarray, pct: float = 0.5) -> tuple[float, float]:
        """Contrast limits from percentiles, so a few outliers cannot flatten it."""
        return float(np.percentile(img, pct)), float(np.percentile(img, 100 - pct))

    ax = fig.add_axes([0.045, 0.175, 0.575, 0.625])
    view = smooth[::4, ::4]
    lo, hi = stretch(view)
    ax.imshow(view, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
    ax.set_title("full micrograph, low-passed to 20 A, contrast stretched",
                 fontsize=11.5, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)

    # Mark the crop used for most of the fast work.
    crop_y, crop_x, ch, cw = 1535, 2160, 1023, 1440
    ax.add_patch(mpatches.Rectangle(
        (crop_x / 4, crop_y / 4), cw / 4, ch / 4,
        fill=False, edgecolor=AMBER, lw=1.8))
    ax.text(crop_x / 4 + 4, crop_y / 4 + ch / 4 - 8,
            "crop used for the fast comparisons", fontsize=9.5, color=AMBER,
            fontweight="bold", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    with mrcfile.open(CROP_MICROGRAPH, permissive=True) as handle:
        crop = np.asarray(handle.data, dtype=np.float32)
    crop_s = gaussian_filter(crop, lowpass_sigma)
    ax2 = fig.add_axes([0.655, 0.175, 0.30, 0.625])
    lo, hi = stretch(crop_s)
    ax2.imshow(crop_s, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
    ax2.set_title("the crop, same filtering", fontsize=11.5, color=INK, loc="left")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor(MUTED)

    # Scale bars, in the display pixels of each panel.
    _scale_bar(ax, view.shape, 100.0, PIXEL_SIZE * 4, "100 nm")
    _scale_bar(ax2, crop_s.shape, 50.0, PIXEL_SIZE, "50 nm")
    pdf.savefig(fig)
    plt.close(fig)


def _body(fig, x, y, text, width=0.42, size=11.5):
    """A block of explanatory prose."""
    fig.text(x, y, text, fontsize=size, color=INK, va="top", linespacing=1.65)




def slide_constraint(pdf: PdfPages) -> None:
    """What the filament constraint is, drawn on the data it was drawn on."""
    import yaml
    from scipy.ndimage import gaussian_filter

    fig = slide("Constraining the search to the filament",
                "A line and a box drawn in napari, which decide where and at what "
                "angles a match may be reported")

    with open(HERE / "configs" / "filament_constraint_full.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    with mrcfile.open(FULL_MICROGRAPH, permissive=True) as handle:
        full = np.asarray(handle.data, dtype=np.float32)
    view = gaussian_filter(full, 20.0 / (2.355 * PIXEL_SIZE))[::4, ::4]

    ax = fig.add_axes([0.045, 0.10, 0.44, 0.70])
    lo, hi = np.percentile(view, 0.5), np.percentile(view, 99.5)
    ax.imshow(view, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)

    # The box corners are (y, x); order them around their centroid to draw a polygon.
    corners = np.array(cfg["spatial_box"]["corners"], dtype=float) / 4.0
    centre = corners.mean(axis=0)
    order = np.argsort(np.arctan2(corners[:, 0] - centre[0], corners[:, 1] - centre[1]))
    ax.set_xlim(0, view.shape[1])
    ax.set_ylim(view.shape[0], 0)
    _scale_bar(ax, view.shape, 100.0, PIXEL_SIZE * 4, "100 nm")

    _body(fig, 0.52, 0.79,
          "Both are drawn by hand in napari, on this micrograph, and\n"
          "saved as a sidecar the search reads.\n\n"
          "The box restricts WHERE. The line restricts AT WHAT ANGLE:\n"
          "ψ to a 10° cone about the drawn direction and its opposite\n"
          "pole, θ to within 10° of the image plane. φ is left completely\n"
          "free, because the roll is the measurement.\n\n"
          "It restricts only which tuples may WIN. The correlation\n"
          "statistics are still accumulated over the whole orientation\n"
          "grid, so the z-scores are normalised exactly as they would be\n"
          "without a constraint — the threshold keeps its meaning.\n\n"
          "And it is a prior on position and direction only: nothing about\n"
          "the repeat, the protofilament number or the polarity. Every\n"
          "structural readout that follows is free of it.\n\n"
          "One practical trap. The sidecar is stored in the frame the\n"
          "search's own maps use, indexed from the template's top-left\n"
          "corner, while the micrograph is indexed from a particle centre.\n"
          "The two are half a template width apart, so overlaying one on\n"
          "the other without adding 300 px puts it 276 Å off the filament.")

    note(fig, "Both poles are searched, so the constraint fixes the filament's axis "
              "but not which way it points — polarity is measured, not assumed.",
         y=0.018)
    pdf.savefig(fig)
    plt.close(fig)



def slide_what_is_2dtm(pdf: PdfPages) -> None:
    """What the search actually does, for a reader who does not run 2DTM."""
    fig = slide("What the search does",
                "Two-dimensional template matching, in one slide")

    _body(fig, 0.045, 0.80,
          "A known structure is simulated to a 3-D volume, projected at\n"
          "every orientation on a fine grid, and each projection is\n"
          "cross-correlated with the raw micrograph at every pixel and\n"
          "every defocus.\n\n"
          "Nothing is picked and nothing is averaged: every position in\n"
          "the image is scored against the model. And the score is\n"
          "calibrated — each correlation becomes a z-score against the\n"
          "spread of all orientations tried at that pixel, so a detection\n"
          "carries a false-positive rate rather than a rank.")

    fig.add_artist(plt.Line2D([0.045, 0.44], [0.455, 0.455], color=LIGHT, lw=2))
    fig.text(0.045, 0.435, "The search as run here", fontsize=12.5,
             fontweight="bold", color=BLUE, va="top")
    settings = [
        ("grid  (YAML)", "uniform over SO(3), ψ step 2.5°, θ step 3.5°"),
        ("range (YAML)", "none — the YAML restricts no angle at all"),
        ("ψ  (sidecar)", "270–287.5° and 90–107.5°, the two poles"),
        ("θ  (sidecar)", "80.5–98°, i.e. within 10° of in-plane"),
        ("φ", "free over 0–360°: the roll is never constrained"),
        ("defocus", "−600 to +600 Å in 200 Å steps, so 7 planes"),
        ("total", "485,856 orientations × 7 defocus × 24 M pixels"),
    ]
    for i, (label, value) in enumerate(settings):
        y = 0.385 - i * 0.038
        fig.text(0.055, y, label, fontsize=10.5, color=MUTED, va="top")
        fig.text(0.175, y, value, fontsize=10.5, color=INK, va="top")

    for i, phi in enumerate((0.0, 60.0, 120.0)):
        ax = fig.add_axes([0.535 + i * 0.145, 0.44, 0.135, 0.34])
        show_template(ax, template_projection(PATCH, phi=phi, crop=420),
                      f"φ = {phi:.0f}°", scale_nm=None)
    fig.add_artist(plt.Line2D([0.535, 0.955], [0.395, 0.395], color=LIGHT, lw=2))
    fig.text(0.535, 0.375, "Then refined off the grid", fontsize=12.5,
             fontweight="bold", color=GREEN, va="top")
    refine = [
        ("orientation", "off-grid: 2.0° coarse, then 0.4° fine in φ, θ, ψ"),
        ("defocus", "−200 to +200 Å in 50 Å steps; pixel size held"),
        ("stack", "the 82 lattice sites, 640 px boxes, 600 px template"),
        ("effect", "φ: 19 → 59 distinct values; mean z 13.3 → 15.5"),
    ]
    for i, (label, value) in enumerate(refine):
        y = 0.320 - i * 0.038
        fig.text(0.545, y, label, fontsize=10.5, color=MUTED, va="top")
        fig.text(0.655, y, value, fontsize=10.5, color=INK, va="top")

    note(fig, "The sidecar gates MIP ELIGIBILITY, not the search: all 485,856 "
              "orientations are still correlated at every pixel, and correlation mean "
              "and variance are accumulated over\nthe FULL grid, which is what keeps the "
              "z-scores on the same scale as an unconstrained search. The sidecar also "
              "carries psi_step 1.5 / theta_step 2.5, which are\nignored — the YAML's "
              "steps win, confirmed from the angles in the table: θ at 80.5 / 84 / 87.5 / "
              "91 / 94.5 / 98°.  ·  Refinement matters wherever φ is read as a "
              "continuous\nquantity — the supertwist and the monomer register both need "
              "it, since on the grid a whole protofilament holds only one or two distinct "
              "roll angles.", y=0.035)
    pdf.savefig(fig)
    plt.close(fig)


def slide_mip_problem(pdf: PdfPages) -> None:
    """Why the standard output discards exactly the information wanted here."""
    fig = slide("Why the usual output is not enough",
                "The map keeps one answer per pixel — but the answer is the orientation")

    ax = fig.add_axes([0.045, 0.30, 0.30, 0.50])
    show_template(ax, template_projection(RING, theta=0.0, crop=380),
                  "the template seen down its own axis — 14 protofilaments",
                  scale_nm=10.0)
    ax.annotate("", xy=(196, 58), xytext=(196, 322),
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.2))
    fig.text(0.145, 0.265,
             "The two ends of that arrow land on the same\n"
             "pixels when the tube is viewed from the side.",
             fontsize=10.5, color=AMBER, fontweight="bold", ha="center",
             va="top", linespacing=1.6)

    _body(fig, 0.40, 0.80,
          "A microtubule is hollow. Viewed from the side, the near and far\n"
          "walls fall on the same pixels but face opposite ways — they differ\n"
          "only in the roll angle φ.\n\n"
          "The standard output keeps the single best-scoring hypothesis at\n"
          "each pixel. So the two walls compete for the same pixels and one\n"
          "erases the other, and every sub-maximal orientation is discarded.\n\n"
          "That is fatal here, because the orientation IS the measurement:", 0.5)

    rows = [("φ", "roll about the tube axis", "which protofilament, near or far wall"),
            ("ψ", "in-plane direction", "the two poles are the two polarities"),
            ("θ", "tilt out of the image plane", "90° = axis in the image plane")]
    for i, (sym, meaning, use) in enumerate(rows):
        y = 0.365 - i * 0.062
        fig.text(0.405, y, sym, fontsize=15, fontweight="bold", color=BLUE)
        fig.text(0.435, y, meaning, fontsize=11.5, color=INK)
        fig.text(0.655, y, use, fontsize=11.5, color=MUTED)

    fig.text(0.40, 0.155,
             "So the analysis reads the sparse correlation table instead — every\n"
             "detection above a threshold, not just the winner at each pixel.",
             fontsize=12, color=BLUE, va="top", linespacing=1.6, fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def slide_templates(pdf: PdfPages) -> None:
    """The models used for matching, and why the shape of the model matters."""
    fig = slide("The models used for matching",
                "Built from PDB 6DPU, simulated at 0.9194 Å — the shape of the model "
                "decides what it can measure")

    panels = [
        (RING, 0.0, 380, "complete ring, down the axis", 10.0),
        (RING, 87.5, 380, "complete ring, from the side", 10.0),
        (PATCH, 87.5, 460, "patch of lattice, from the side", 10.0),
    ]
    for i, (name, theta, crop, title, bar) in enumerate(panels):
        ax = fig.add_axes([0.045 + i * 0.235, 0.40, 0.215, 0.40])
        show_template(ax, template_projection(name, theta=theta, crop=crop), title, bar)

    _body(fig, 0.76, 0.80,
          "Two shapes, and they are\n"
          "complementary rather than\n"
          "interchangeable.\n\n"
          "The ring is centred on the tube\n"
          "axis and 14-fold pseudo-\n"
          "symmetric, so rolling it barely\n"
          "changes the projection: φ is\n"
          "undetermined, but position is\n"
          "excellent.\n\n"
          "The patch sits 86.8 Å off the\n"
          "axis, so rolling it physically\n"
          "moves the mass: φ is well\n"
          "determined, position mediocre.", 0.22, 11.0)

    fig.text(0.045, 0.33,
             "Keeping the patch off-centre is deliberate. Centring it on the axis was "
             "tried and is worse on every measure —\nrolling φ then just sweeps the "
             "patch around the tube without moving it, so many orientations become "
             "degenerate.",
             fontsize=11, color=INK, va="top", linespacing=1.6)

    note(fig, "Ring: 56 chains, 14 protofilaments x 2 dimers, radius 118.3 Å, repeat "
              "83.96 Å.  Patch: 48 chains spanning ~6 protofilaments.\nThe ring is used "
              "for the axis, the repeat and polarity; the patch is the only one carrying "
              "per-protofilament information.")
    pdf.savefig(fig)
    plt.close(fig)


def slide_orientation_peaks(pdf: PdfPages) -> None:
    """Peak finding that keeps hypotheses which overlap in space but not in angle."""
    fig = slide("Finding peaks in position and angle together",
                "Ordinary peak finding would merge the two walls back into one")

    ax = fig.add_axes([0.045, 0.20, 0.42, 0.58])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.add_patch(mpatches.Circle((3.0, 7.0), 2.4, facecolor=LIGHT, edgecolor=BLUE, lw=1.6))
    ax.text(3.0, 9.8, "same position", ha="center", fontsize=11.5, fontweight="bold",
            color=BLUE)
    for dx, dy, ang, col in ((-0.8, 0.5, "φ = 20°", BLUE), (0.8, -0.5, "φ = 200°", AMBER)):
        ax.scatter([3.0 + dx], [7.0 + dy], s=190, color=col, zorder=3)
        ax.text(3.0 + dx + 0.35, 7.0 + dy, ang, fontsize=10.5, color=col, va="center")
    ax.text(3.0, 4.05, "different angle → BOTH kept", ha="center", fontsize=11.5,
            color=GREEN, fontweight="bold")

    ax.add_patch(mpatches.Circle((7.6, 7.0), 2.4, facecolor=LIGHT, edgecolor=BLUE, lw=1.6))
    for dx, dy, col in ((-0.5, 0.35, BLUE), (0.5, -0.35, MUTED)):
        ax.scatter([7.6 + dx], [7.0 + dy], s=190, color=col, zorder=3)
    ax.text(7.6 + 0.85, 6.65, "φ = 24°", fontsize=10.5, color=MUTED, va="center")
    ax.text(7.6 - 0.85, 7.35, "φ = 20°", fontsize=10.5, color=BLUE, va="center", ha="right")
    ax.text(7.6, 4.05, "same angle → merged", ha="center", fontsize=11.5, color=RED,
            fontweight="bold")

    _body(fig, 0.53, 0.79,
          "A detection is suppressed by a stronger one only when the two\n"
          "are close in position AND close in orientation — 12 px and 15°\n"
          "here.\n\n"
          "Both radii are bracketed. Below, by the width of one particle's\n"
          "correlation lobe, which is set by the template's resolution\n"
          "content and not by its box size — a 600-pixel microtubule\n"
          "template still gives a lobe a few pixels across. Above, by the\n"
          "closest spacing that must stay resolved: adjacent protofilaments\n"
          "are 25.7° apart, so the angular radius has to sit below half of\n"
          "that.\n\n"
          "Defocus is deliberately left out of the neighbourhood and maxed\n"
          "over instead: 200 Å steps are far too coarse to separate anything\n"
          "at the scale of a tube.", 0.45)

    note(fig, "This is what keeps the near and far wall of the same tube as two "
              "detections rather than one, and it is the step that makes everything "
              "after it possible.")
    pdf.savefig(fig)
    plt.close(fig)


def load_transverse() -> tuple:
    """Distance of patch detections from the fitted axis, before and after correcting.

    This is the whole slide in two numbers, so it is measured rather than asserted.
    """
    cache = CACHE.with_name("transverse.npy")
    if cache.exists():
        data = np.load(cache)
        return data[0], data[1]

    from leopard_em.analysis.filament_lattice import (
        TemplateLatticeGeometry,
        axis_points_from_peaks,
        filament_coordinates,
        filament_direction_from_angles,
        fit_filament_axis,
    )
    from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

    det = detections_from_hdf5(
        str(HERE / "results_full" / "output_correlation_table_patch4_full.h5"),
        min_z_score=9.0,
    )
    geometry = TemplateLatticeGeometry.from_pdb(
        str(HERE / "models" / "6dpu_4_patches_aligned_zero.pdb")
    )
    raw = det[["x", "y"]].to_numpy(float)
    corrected = axis_points_from_peaks(det, geometry, PIXEL_SIZE)
    axis = fit_filament_axis(
        corrected,
        det["z_score"].to_numpy(),
        filament_direction_from_angles(
            det["phi"].to_numpy(), det["theta"].to_numpy(), det["psi"].to_numpy()
        ),
    )
    t_raw = filament_coordinates(raw, axis)[1] * PIXEL_SIZE
    t_cor = filament_coordinates(corrected, axis)[1] * PIXEL_SIZE
    np.save(cache, np.vstack([t_raw, t_cor]))
    return t_raw, t_cor




def load_unwrap_demo() -> tuple:
    """Axial coordinate, unwrapped coordinate, azimuth and score for patch detections."""
    cache = CACHE.with_name("unwrap_demo.npy")
    if cache.exists():
        data = np.load(cache)
        return data[0], data[1], data[2], data[3]

    from leopard_em.analysis.filament_lattice import (
        TemplateLatticeGeometry,
        axis_points_from_peaks,
        filament_coordinates,
        filament_direction_from_angles,
        fit_filament_axis,
        unwrap_helical_axial_coordinate,
    )
    from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

    det = detections_from_hdf5(
        str(HERE / "results_full" / "output_correlation_table_patch4_full.h5"),
        min_z_score=9.5,
    )
    geometry = TemplateLatticeGeometry.from_pdb(
        str(HERE / "models" / "6dpu_4_patches_aligned_zero.pdb")
    )
    points = axis_points_from_peaks(det, geometry, PIXEL_SIZE)
    axis = fit_filament_axis(
        points,
        det["z_score"].to_numpy(),
        filament_direction_from_angles(
            det["phi"].to_numpy(), det["theta"].to_numpy(), det["psi"].to_numpy()
        ),
    )
    along = filament_coordinates(points, axis)[0] * PIXEL_SIZE
    phi = det["phi"].to_numpy()
    unwrapped = unwrap_helical_axial_coordinate(along, phi, geometry)
    out = np.vstack([along, unwrapped, phi, det["z_score"].to_numpy()])
    np.save(cache, out)
    return out[0], out[1], out[2], out[3]


def slide_unwrap(pdf: PdfPages) -> None:
    """Unrolling the tube, on real detections."""
    fig = slide("Unrolling the tube: what step 2 actually does",
                "A microtubule is a 3-start helix, so subunits on neighbouring "
                "protofilaments sit at different heights")

    along, unwrapped, phi, score = load_unwrap_demo()
    rise = 41.979
    # A window in the middle of the track, wide enough to show several repeats.
    centre = np.median(unwrapped)
    keep = np.abs(unwrapped - centre) < 130.0
    a, u, f = along[keep] - centre, unwrapped[keep] - centre, phi[keep]

    for i, (values, title) in enumerate(
        [(a, "as measured: no lattice"),
         (u, "after the offset — rows are site numbers")]
    ):
        ax = fig.add_axes([0.075 + i * 0.235, 0.24, 0.205, 0.50])
        ax.scatter(f, values, s=3, c=BLUE, alpha=0.25, linewidths=0)
        ax.set_xlim(0, 360)
        ax.set_ylim(-130, 130)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("azimuth φ (degrees)")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        if i == 0:
            ax.set_ylabel("distance along the axis (Å)")
            # The ramp the unwrap removes: -9 A per 25.71 deg of azimuth.
            grad = -9.0 / 25.714
            for k in (-2, -1, 0, 1, 2):
                ax.plot([0, 360], [k * rise, k * rise + 360 * grad],
                        color=AMBER, lw=1.4, ls="--")
            ax.text(6, -126, "each dashed line is\none ring of subunits", fontsize=9.5,
                    color=AMBER, fontweight="bold", va="bottom")
        else:
            ax.set_yticklabels([])
            for k in (-3, -2, -1, 0, 1, 2, 3):
                ax.axhline(k * rise, color=GREEN, lw=1.3, ls="--")
                ax.text(366, k * rise, f"{k:+d}" if k else " 0", fontsize=9.5,
                        color=GREEN, fontweight="bold", va="center")


    _body(fig, 0.58, 0.76,
          "Going once around the tube, each protofilament is displaced\n"
          "9 Å along the axis from the last. After all 14 that is 126 Å —\n"
          "exactly three monomers, which is what makes the lattice a\n"
          "3-start helix and forces a seam.\n\n"
          "So subunits at the same level on different protofilaments are\n"
          "not at the same height: they lie on the sloping dashed lines\n"
          "at the left, spread over three whole repeats. Binning by\n"
          "height alone would smear the lattice into nothing.\n\n"
          "Every detection reports its own φ, and the slope is known —\n"
          "9 Å per 25.71°, so 0.35 Å per degree. Subtracting it flattens\n"
          "the ramp, and all 14 protofilaments fall onto one lattice of\n"
          "period 41.98 Å. This is just unrolling the tube.\n\n"
          "Those flattened rows are numbered with integers — that is the\n"
          "site number. Fitting measured position against site number,\n"
          "over ~80 of them, is what gives the repeat: far more precise\n"
          "than measuring gaps between neighbours.", 0.38)

    note(fig, "Real detections, patch template, z > 9.5. Lattice sharpness at 41.98 Å "
              "goes from |R| = 0.097 — no lattice at all — to |R| = 0.577.")
    pdf.savefig(fig)
    plt.close(fig)



def slide_calibration(pdf: PdfPages) -> None:
    """Why an off-axis template needs a correction, and what it buys."""
    fig = slide("Patch templates: the position is not the axis",
                "An extra correction the ring does not need, because its box centre "
                "already sits on the axis")

    # --- left: the geometry, looking down the tube ---------------------------------
    ax = fig.add_axes([0.045, 0.20, 0.30, 0.60])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_title("looking down the tube axis", fontsize=11.5, color=INK, loc="left")

    for k in range(14):
        a = 2 * np.pi * k / 14
        ax.add_patch(mpatches.Circle((np.cos(a), np.sin(a)), 0.13,
                                     facecolor=LIGHT, edgecolor=MUTED, lw=1.0))
    ax.scatter([0], [0], marker="x", s=170, color=INK, lw=2.6, zorder=5)
    ax.text(0.07, -0.16, "axis", fontsize=11, color=INK, fontweight="bold")

    # Three detections at different roll angles. The box centre sits 86.8 A from the
    # axis against a 118.7 A subunit radius, hence 0.73 of the ring radius.
    for a_deg, colour in ((25.0, AMBER), (150.0, GREEN), (265.0, BLUE)):
        a = np.radians(a_deg)
        x, y = 0.73 * np.cos(a), 0.73 * np.sin(a)
        ax.annotate("", xy=(0, 0), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=2.2))
        ax.scatter([x], [y], s=110, color=colour, zorder=6,
                   edgecolor="white", lw=1.2)
    ax.text(0, -1.36,
            "three detections at different φ\n"
            "every arrow the same length, 86.8 Å\n"
            "every arrow a different direction",
            fontsize=10.5, color=INK, ha="center", va="top", linespacing=1.6)

    # --- right: what it does to real data ------------------------------------------
    t_raw, t_cor = load_transverse()
    ax2 = fig.add_axes([0.44, 0.30, 0.28, 0.47])
    bins = np.linspace(-140, 140, 90)
    ax2.hist(t_raw, bins=bins, color=AMBER, alpha=0.55,
             label=f"as reported   (sd {t_raw.std():.0f} Å)")
    ax2.hist(t_cor, bins=bins, color=BLUE, alpha=0.85,
             label=f"corrected      (sd {t_cor.std():.1f} Å)")
    ax2.set_xlabel("distance from the fitted axis (Å)")
    ax2.set_ylabel("detections")
    ax2.set_yticks([])
    ax2.legend(frameon=False, fontsize=10)
    ax2.set_title(f"{len(t_raw):,} real patch detections", fontsize=11.5,
                  color=INK, loc="left")

    # --- right column: the argument -------------------------------------------------
    _body(fig, 0.755, 0.79,
          "Everything so far assumed a\n"
          "detection already gives a point on\n"
          "the axis. For the ring that is true.\n\n"
          "For a patch it is not: the box centre\n"
          "sits 86.8 Å to one side, and that\n"
          "offset rotates with φ, pointing out\n"
          "towards whichever protofilament\n"
          "matched. So patch detections form a\n"
          "hollow tube, not a line — fitting\n"
          "them directly gives a radius of 70 Å\n"
          "instead of the true 118.8 Å.\n\n"
          "But every detection reports its own\n"
          "φ, so every detection knows which\n"
          "way to point:", 0.22, 10.5)

    fig.text(0.755, 0.283, "axis point = position + (Rᵀp)[:2]",
             fontsize=11.5, color=BLUE, fontweight="bold")
    _body(fig, 0.755, 0.243,
          "Transposed because the projection\n"
          "rotates the sampling grid, not the\n"
          "object — checked against rendered\n"
          "projections to 0.1 pixel.", 0.22, 10.5)

    note(fig, "The corrected scatter is the cheapest error detector in the pipeline: "
              "it tests the template calibration, the Euler convention, the transpose "
              "and the axis fit at once.\n"
              "One trap — a principal-axis fit gives the direction only up to sign, and "
              "the helical unwrap depends on that sign. Pin it from the detections' own "
              "orientations.")
    pdf.savefig(fig)
    plt.close(fig)


def slide_topdown(pdf: PdfPages) -> None:
    """The method that worked, in detail."""
    fig = slide("Top-down site extraction — the strategy that worked",
                "Use the model to decide where to look, instead of thresholding "
                "the whole image")

    _body(fig, 0.045, 0.79,
          "A global score threshold forces a bad trade: cut high and real subunits are\n"
          "lost, cut low and noise corrupts the axis fit, which biases everything\n"
          "measured along it. So the model is used to say where to look, and only then\n"
          "is the score read.", 0.5)

    steps = [
        ("1", "Fit the axis",
         "Take only the strongest peaks — z > 8, leaving 160 on the full\n"
         "frame — and fit a straight line through their positions. That\n"
         "line is the axis, and everything later is measured along it."),
        ("2", "Remove the helical offset",
         "Neighbouring protofilaments are displaced 9 Å along the axis:\n"
         "the lattice is a 3-start helix, not flat rings. Subtract that\n"
         "φ-dependent height and all 14 share one lattice of 41.9 Å."),
        ("3", "Assign each site its best detection",
         "Round every detection to its nearest lattice site and keep the\n"
         "highest-scoring one there — however weak it is."),
        ("4", "Re-fit the repeat, and iterate",
         "Fit the winners' positions against their site numbers to get a\n"
         "better repeat, then re-assign with it. Three rounds are run;\n"
         "nothing moves after the first, as the template's own repeat\n"
         "already starts within 0.1% of the answer."),
    ]
    # Step by the real height of each block rather than a fixed pitch, or the longer
    # bodies run into the next heading.
    line_height, head_gap, block_gap = 0.0292, 0.036, 0.020
    y = 0.615
    for num, head, body in steps:
        bullet_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=-1)
        bullet_ax.axis("off")
        fig.patches.append(mpatches.Circle(
            (0.062, y - 0.004), 0.0115, transform=fig.transFigure, facecolor=BLUE,
            zorder=3))
        fig.text(0.062, y - 0.004, num, fontsize=11, fontweight="bold", color="white",
                 ha="center", va="center", zorder=4)
        fig.text(0.088, y, head, fontsize=12, fontweight="bold", color=BLUE, va="top")
        fig.text(0.088, y - head_gap, body, fontsize=10.5, color=INK, va="top",
                 linespacing=1.5)
        y -= head_gap + line_height * (body.count("\n") + 1) + block_gap

    fig.add_artist(plt.Line2D([0.615, 0.615], [0.12, 0.80], color=LIGHT, lw=2))

    _body(fig, 0.645, 0.79,
          "Noise away from the lattice is never\n"
          "examined, so purity and completeness\n"
          "stop competing:\n", 0.32)
    table = [("", "z > 8 global", "top-down"),
             ("sites occupied", "71%", "99%"),
             ("weakest site kept", "z 8.0", "z 6.12"),
             ("repeat moves by", "0.083 Å", "0.008 Å")]
    for i, (label, a, b) in enumerate(table):
        y = 0.655 - i * 0.045
        weight = "bold" if i == 0 else "normal"
        fig.text(0.645, y, label, fontsize=11, color=MUTED if i else INK,
                 fontweight=weight)
        fig.text(0.845, y, a, fontsize=11, color=MUTED, fontweight=weight, ha="right")
        fig.text(0.945, y, b, fontsize=11, color=GREEN if i else INK,
                 fontweight="bold", ha="right")

    _body(fig, 0.645, 0.485,
          "\"Repeat moves by\" is the last row: re-run\n"
          "the whole thing from a different bootstrap\n"
          "cut and a different starting repeat, and\n"
          "see how far the answer shifts. Ten times\n"
          "steadier here.\n", 0.32)

    _body(fig, 0.645, 0.305,
          "Is it circular? The window is a full\n"
          "half-repeat, ±20.9 Å, so a site is free\n"
          "to land anywhere in it. Detections land\n"
          "2.9 Å rms from their predictions — so the\n"
          "data, not the model, sets the positions.\n\n"
          "The driver prints that number every run\n"
          "and warns when it approaches the window.", 0.32)
    pdf.savefig(fig)
    plt.close(fig)

def slide_scorecard(pdf: PdfPages) -> None:
    fig = slide("Where the four readouts stand", "and what limits each one")
    ax = fig.add_axes([0.045, 0.10, 0.91, 0.72])
    ax.axis("off")

    rows = [
        ("Polarity", GREEN, "SOLVED",
         "ψ = 279.2°, ~2500 : 1 at z > 7",
         "Decisive and cheap. The two ψ poles separate by orders of magnitude."),
        ("Lattice spacing", GREEN, "SOLVED",
         "41.86 Å monomer  ·  expanded, t = 10",
         "Two independent routes agree: direct site fit, and template competition\n"
         "against the deposited 6DPU / 6DPV models."),
        ("Protofilament number", AMBER, "PROBABLY 13",
         "13 at 7.6σ, by a statistic validated both ways",
         "Competing 12/13/14/15 models says 13; the patch harmonic and the absent\n"
         "supertwist agree. Overturns the earlier 14 — but GMPCPP favours 14."),
        ("Seam", RED, "BLOCKED",
         "α/β contrast ≈ 1.0 in every template",
         "Four routes closed. Not a data problem: the template cannot tell its own\n"
         "half-repeats apart, which is a noise-free upper bound."),
    ]
    for i, (name, colour, state, value, why) in enumerate(rows):
        y = 0.88 - i * 0.235
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.012, y - 0.155), 0.976, 0.195, boxstyle="round,pad=0.006",
            facecolor="white", edgecolor=colour, lw=1.4, transform=ax.transAxes))
        ax.add_patch(mpatches.Rectangle((0.012, y - 0.155), 0.006, 0.195,
                                        facecolor=colour, transform=ax.transAxes))
        ax.text(0.025, y - 0.005, name, fontsize=14, fontweight="bold",
                transform=ax.transAxes, va="center")
        ax.text(0.025, y - 0.075, state, fontsize=10.5, color=colour,
                fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.245, y - 0.005, value, fontsize=12.5, color=INK,
                transform=ax.transAxes, va="center")
        ax.text(0.245, y - 0.088, why, fontsize=10.5, color=MUTED,
                transform=ax.transAxes, va="center", linespacing=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    pdf.savefig(fig)
    plt.close(fig)


def slide_polarity(pdf: PdfPages, det: pd.DataFrame) -> None:
    fig = slide("Which way does the microtubule point?",
                "Ring template on the full micrograph — it is polar, so its two ψ poles "
                "are the two directions")

    cuts = np.arange(6.0, 9.6, 0.25)
    psi = det["psi"].to_numpy()
    zs = det["z_score"].to_numpy()
    forward, reverse = [], []
    for cut in cuts:
        sel = zs > cut
        forward.append(int(((psi > 240) & (psi < 320) & sel).sum()))
        reverse.append(int(((psi > 60) & (psi < 140) & sel).sum()))
    forward = np.array(forward)
    reverse = np.array(reverse)

    # Show the two poles. They are near-copies of one another, which is the point:
    # the polarity signal is small per particle and only decisive when summed.
    for i, (theta, psi_pole, label) in enumerate(
        [(87.5, 280.0, "ψ ≈ 280°"), (92.5, 100.0, "ψ ≈ 100°, flipped end-for-end")]
    ):
        axt = fig.add_axes([0.045 + i * 0.145, 0.44, 0.135, 0.33])
        show_template(axt, template_projection(RING, theta=theta, psi=psi_pole,
                                               crop=420), label, scale_nm=None)
    live = reverse > 0
    ax = fig.add_axes([0.40, 0.22, 0.25, 0.55])
    ax.semilogy(cuts, forward, color=BLUE, lw=2.4,
                label="ψ ≈ 279°  (the microtubule)")
    ax.semilogy(cuts[live], reverse[live], color=RED, lw=2.4,
                label="ψ ≈ 99°  (reversed)")
    ax.scatter(cuts[live][-1], reverse[live][-1], s=30, color=RED, zorder=3)
    ax.annotate("last reversed detection", xy=(cuts[live][-1], reverse[live][-1]),
                xytext=(cuts[live][-1] + 0.35, reverse[live][-1] * 6),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.set_xlabel("z-score threshold")
    ax.set_ylabel("detections")
    ax.set_title("The wrong pole dies out", fontsize=12.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    ax.grid(alpha=0.25, ls=":")

    ax2 = fig.add_axes([0.715, 0.22, 0.24, 0.55])
    ratio = forward[live] / reverse[live]
    ax2.semilogy(cuts[live], ratio, color=BLUE, lw=2.4, marker="o", ms=4)
    edge = cuts[live][-1]
    ax2.axvspan(edge, cuts[-1], color=LIGHT, alpha=0.8)
    ax2.text((edge + cuts[-1]) / 2, ratio.max() * 0.55,
             "beyond here the\nreversed pole is\nempty — the ratio\nis unbounded",
             fontsize=9.5, color=MUTED, ha="center", va="center")
    ax2.set_xlim(cuts[0], cuts[-1])
    ax2.set_xlabel("z-score threshold")
    ax2.set_ylabel("polarity ratio")
    ax2.set_title("Ratio, forward : reversed", fontsize=12.5, color=INK, loc="left")
    ax2.grid(alpha=0.25, ls=":")

    note(fig, "Full micrograph, 2-ring template, 5.1 M detections streamed from the "
              "correlation table.  At z > 7 the ratio is ~2500 : 1.\n"
              "This is the one readout that needs no lattice model at all — only that "
              "the template resolves the filament's direction.")
    pdf.savefig(fig)
    plt.close(fig)


def slide_spacing_direct(pdf: PdfPages, sites) -> None:
    """The measured repeat, and why it is not an artefact of the starting guess."""
    fig = slide("Lattice spacing (1) — direct measurement",
                "Ring template, full micrograph: fit a rough lattice from the strong "
                "peaks, then read every site it predicts")

    score = np.asarray(sites.score)
    axial = np.asarray(sites.axial_index, dtype=float)
    dev = np.asarray(sites.axial_deviation_angstrom)
    window = sites.rise_angstrom / 2
    observed = float(np.sqrt((dev ** 2).mean()))
    # With no lattice the detections would scatter uniformly inside each bin, which is
    # the only null worth comparing to: the +/- half-repeat bound is automatic, because
    # every detection is assigned to its nearest site.
    null_rms = window / np.sqrt(3.0)

    ax = fig.add_axes([0.045, 0.26, 0.27, 0.50])
    ax.scatter(axial, score, s=26, color=BLUE, alpha=0.85, edgecolor="none")
    ax.set_xlabel("site number")
    ax.set_ylabel("z-score")
    ax.set_title(f"{int(sites.occupancy * sites.n_predicted)} of {sites.n_predicted} "
                 f"predicted sites occupied ({sites.occupancy:.0%})",
                 fontsize=11.5, color=INK, loc="left")
    ax.grid(alpha=0.25, ls=":")

    ax2 = fig.add_axes([0.385, 0.26, 0.27, 0.50])
    ax2.hist(dev, bins=26, color=BLUE, alpha=0.9,
             label=f"observed, {observed:.1f} Å rms")
    top = ax2.get_ylim()[1]
    ax2.axhspan(0, top, xmin=0.0, xmax=1.0, color=AMBER, alpha=0.10,
                label=f"no lattice would give {null_rms:.1f} Å rms")
    for sign in (-1, 1):
        ax2.axvline(sign * window, color=RED, ls="--", lw=1.8)
    ax2.set_xlim(-window * 1.1, window * 1.1)
    ax2.set_ylim(0, top)
    ax2.set_xlabel("deviation from the predicted site (Å)")
    ax2.set_ylabel("sites")
    ax2.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax2.set_title("the detections pick their own positions", fontsize=11.5,
                  color=INK, loc="left")

    _body(fig, 0.70, 0.76,
          f"Measured monomer repeat {sites.rise_angstrom:.3f} Å —\n"
          "0.27% from the expanded model, 2.5%\n"
          "from the compacted one.\n\n"
          "Start the extraction from the COMPACTED\n"
          "repeat and it converges to the same\n"
          f"{sites.rise_angstrom:.3f} Å: same sites, same 100%\n"
          "occupancy, same 2.9 Å scatter. Starting\n"
          "values of 39, 40.8, 42 and 44 Å all land\n"
          "there. The data sets the repeat, not the\n"
          "starting guess.\n\n"
          "The ±20.9 Å bound is automatic — every\n"
          "detection is assigned to its nearest site,\n"
          "so it cannot be further than half a repeat\n"
          "from one, on any iteration. The test that\n"
          f"means something is against the {null_rms:.1f} Å\n"
          "a lattice-free scatter would give.", 0.26, 11.0)

    note(fig, "A wrong repeat would not show up as extra scatter but as drift, because "
              "the error accumulates one subunit at a time: 1.14 Å per site over 81 "
              "sites is 93 Å,\nfour times the half-repeat window. Nothing of the kind "
              "is present.")
    pdf.savefig(fig)
    plt.close(fig)


def slide_spacing_competition(pdf: PdfPages, z: dict, n_sites: int) -> None:
    fig = slide("Lattice spacing (2) — template competition",
                "Ring templates, cropped micrograph: expanded (6DPU / GMPCPP) against "
                "compacted (6DPV / GDP), 60 paired sites")

    exp, comp = "6dpu", "6dpv"
    lattice = 0.5 * ((z[("6dpu", exp)] - z[("6dpu", comp)])
                     + (z[("6dpv", exp)] - z[("6dpv", comp)]))
    quality = 0.5 * ((z[("6dpu", exp)] - z[("6dpv", exp)])
                     + (z[("6dpu", comp)] - z[("6dpv", comp)]))
    interaction = ((z[("6dpu", exp)] - z[("6dpu", comp)])
                   - (z[("6dpv", exp)] - z[("6dpv", comp)]))

    ax = fig.add_axes([0.065, 0.455, 0.275, 0.325])
    labels = ["6DPU atoms", "6DPV atoms"]
    width = 0.34
    xs = np.arange(2)
    means_exp = [z[("6dpu", "6dpu")].mean(), z[("6dpv", "6dpu")].mean()]
    means_comp = [z[("6dpu", "6dpv")].mean(), z[("6dpv", "6dpv")].mean()]
    ax.bar(xs - width / 2, means_exp, width, color=BLUE, label="expanded lattice")
    ax.bar(xs + width / 2, means_comp, width, color="#c8d6e5", edgecolor=BLUE,
           label="compacted lattice")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(6.5, 7.75)
    ax.set_ylabel("mean z at matched sites")
    ax.set_title("The 2×2", fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    ax2 = fig.add_axes([0.425, 0.455, 0.21, 0.325])
    contrasts = [("lattice", lattice, BLUE), ("model\nquality", quality, MUTED),
                 ("inter-\naction", interaction, "#9aa5b1")]
    for i, (name, values, colour) in enumerate(contrasts):
        sem = values.std(ddof=1) / np.sqrt(len(values))
        ax2.bar(i, values.mean(), 0.55, yerr=sem, capsize=5, color=colour)
        ax2.text(i, values.mean() + sem + 0.016,
                 f"t = {values.mean() / sem:.1f}", ha="center", fontsize=10.5,
                 fontweight="bold", color=colour if colour != MUTED else INK)
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_ylim(-0.04, 0.55)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([c[0] for c in contrasts], fontsize=10.5)
    ax2.set_ylabel("effect (z)")
    ax2.set_title("Paired contrasts", fontsize=11.5, color=INK, loc="left")

    ax3 = fig.add_axes([0.735, 0.455, 0.22, 0.325])
    ax3.scatter(z[("6dpu", "6dpv")], z[("6dpu", "6dpu")], s=22, color=BLUE,
                alpha=0.75, edgecolor="none")
    lims = [min(ax3.get_xlim()[0], ax3.get_ylim()[0]),
            max(ax3.get_xlim()[1], ax3.get_ylim()[1])]
    ax3.plot(lims, lims, color=MUTED, ls="--", lw=1.2)
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_xlabel("z, compacted")
    ax3.set_ylabel("z, expanded")
    ax3.set_title("Site by site", fontsize=11.5, color=INK, loc="left")

    def summarise(values):
        """Mean and standard error of a paired contrast."""
        sem = values.std(ddof=1) / np.sqrt(len(values))
        return values.mean(), sem

    columns = [
        ("lattice", BLUE, lattice,
         "½[(U·exp − U·cmp) + (V·exp − V·cmp)]",
         "Expanded minus compacted, averaged over both atom\n"
         "sources. THE STRUCTURAL QUESTION: does the specimen\n"
         "match the expanded repeat or the compacted one?\n"
         "Averaging over atom source is what stops the answer\n"
         "depending on which deposition happens to be better."),
        ("model quality", "#6b6b6b", quality,
         "½[(U·exp − V·exp) + (U·cmp − V·cmp)]",
         "6DPU minus 6DPV, averaged over both lattices. Two\n"
         "depositions are never equally good, so this is how\n"
         "much of a win is just the better MODEL rather than\n"
         "the right repeat. It is a coin flip — which is the\n"
         "point: the confound is measured absent, not assumed."),
        ("interaction", "#9aa5b1", interaction,
         "(U·exp − U·cmp) − (V·exp − V·cmp)",
         "Does the lattice effect itself depend on whose atoms\n"
         "were used? A large value would mean the two contrasts\n"
         "are not separable and the 2×2 has failed. At a fifth\n"
         "of the lattice term it is small but not zero: the\n"
         "expanded repeat helps 6DPU slightly more."),
    ]

    for i, (name, colour, values, formula, body) in enumerate(columns):
        x = 0.065 + i * 0.313
        mean, sem = summarise(values)
        fig.text(x, 0.385, f"{name}", fontsize=13, fontweight="bold", color=colour,
                 va="top")
        fig.text(x, 0.343, f"{mean:+.3f} ± {sem:.3f} z   (t = {mean / sem:.1f})",
                 fontsize=11, fontweight="bold", color=INK, va="top")
        fig.text(x, 0.303, formula, fontsize=9.5, color=MUTED, va="top",
                 family="monospace")
        fig.text(x, 0.262, body, fontsize=10, color=INK, va="top", linespacing=1.55)

    fig.text(0.065, 0.095, "U = 6DPU atoms, V = 6DPV atoms;  exp = expanded lattice, "
             "cmp = compacted.  Every contrast is paired site by site across runs, so "
             "defocus, ice and near/far wall cancel.",
             fontsize=9.5, color=MUTED, va="top")

    note(fig, "Verdict: EXPANDED, 54/60 sites, t = 10.  Caveat: the repeat ratio "
              "(1.0280) is degenerate with the pixel-size ratio (1.0259) to 0.2%.",
         y=0.038)
    pdf.savefig(fig)
    plt.close(fig)


def azimuth_harmonics(detections, sites, cuts=(7.0, 7.5, 8.0), orders=range(11, 17)):
    """Weighted azimuth harmonic amplitude per candidate protofilament number.

    Restricted to detections that sit on the fitted lattice and near the axis, so the
    sum is over real subunits rather than over noise.
    """
    from leopard_em.analysis.filament_lattice import (
        TemplateLatticeGeometry,
        axis_points_from_peaks,
        filament_coordinates,
        unwrap_helical_axial_coordinate,
    )

    geometry = TemplateLatticeGeometry.from_pdb(
        str(HERE / "models" / "6dpu_2rings_aligned_zero.pdb")
    )
    points = axis_points_from_peaks(detections, geometry, PIXEL_SIZE)
    along, across = filament_coordinates(points, sites.axis)
    unwrapped = unwrap_helical_axial_coordinate(
        along * PIXEL_SIZE, detections["phi"].to_numpy(), geometry
    )
    rise = sites.rise_angstrom
    phase = np.median(np.mod(unwrapped, rise))
    residual = np.abs(((unwrapped - phase + rise / 2) % rise) - rise / 2)
    on_lattice = (residual < 0.25 * rise) & (np.abs(across * PIXEL_SIZE) < 250.0)

    phi = detections["phi"].to_numpy()
    weight = detections["z_score"].to_numpy()
    orders = np.asarray(list(orders))
    out = []
    for cut in cuts:
        mask = on_lattice & (weight > cut)
        w, radians = weight[mask], np.radians(phi[mask])
        amps = np.array(
            [abs(np.sum(w * np.exp(1j * o * radians)) / w.sum()) for o in orders]
        )
        counts, _ = np.histogram(phi[mask], bins=24, range=(0, 360))
        coverage = int((counts > 0.02 * counts.max()).sum())
        floor = 1.0 / np.sqrt(w.sum() ** 2 / (w**2).sum())
        out.append((cut, int(mask.sum()), coverage, amps, floor))
    return orders, out


def slide_pf_number(pdf: PdfPages, detections, sites) -> None:
    """Reading protofilament number out of the azimuth distribution, and why it fails."""
    fig = slide("Protofilament number from the azimuths",
                "Ring template: if the tube has N protofilaments, the roll angles "
                "should repeat N times around the circle")

    orders, results = azimuth_harmonics(detections, sites)

    ax = fig.add_axes([0.045, 0.24, 0.44, 0.52])
    width, palette = 0.26, [BLUE, "#6699c4", "#bcd2e6"]
    for i, (cut, n, coverage, amps, floor) in enumerate(results):
        xs = orders + (i - 1) * width
        ax.bar(xs, amps, width, color=palette[i], edgecolor="white", lw=0.6,
               label=f"z > {cut}   ({n:,}, {coverage}/24 of the circumference)")
        winner = int(np.argmax(amps))
        ax.text(xs[winner], amps[winner] + 0.018, "▲", ha="center", fontsize=9,
                color=palette[i] if i < 2 else "#7fa5c6")
    ax.set_xticks(orders)
    ax.set_xlabel("candidate protofilament number N")
    ax.set_ylabel("harmonic amplitude  A(N)")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("▲ marks the winner — it moves, and lands on 15",
                 fontsize=11.5, color=INK, loc="left")
    ax.grid(alpha=0.25, ls=":", axis="y")

    _body(fig, 0.53, 0.77,
          "Every detection carries φ, the roll angle, which says which\n"
          "protofilament it matched. If the tube has N of them they sit at\n"
          "azimuths 360/N apart, so the φ distribution should repeat N times\n"
          "around the circle. Testing for an N-fold repeat is a Fourier\n"
          "component, and each candidate N is probed by the same sum with a\n"
          "different multiplier:\n", 0.44)

    fig.text(0.545, 0.545, "A(N)  =  | Σ w · exp( i · N · φ ) |  ⁄  Σ w",
             fontsize=12.5, color=BLUE, fontweight="bold")

    _body(fig, 0.53, 0.495,
          "Each detection becomes a unit vector at angle N × φ, weighted by\n"
          "its z-score, and they are summed. If the azimuths really are\n"
          "multiples of 360/N then every vector points the same way and they\n"
          "add: A → 1. If not they cancel, down to the noise floor — 0.006 to\n"
          "0.020 here. Sweep N and take the largest.\n\n"
          "It does not work at these cuts. The winner moves from 13 to 15 and\n"
          "never beats the runner-up by more than 1.4×. Protofilament number\n"
          "is a CENSUS, so it needs the whole circumference — and by z > 8\n"
          "only 15 of 24 azimuth bins survive, the near wall and the best\n"
          "protofilaments. The harmonic then measures that clustering.", 0.44)

    note(fig, "This estimator wants the opposite of every other one here — weak "
              "detections, not strong — and the ring template is itself 14-fold "
              "pseudo-symmetric, so any N = 14\nsignal may partly be reading the "
              "template rather than the specimen.", y=0.035)
    pdf.savefig(fig)
    plt.close(fig)


def slide_pf_competition(pdf: PdfPages) -> None:
    """Competing whole models of different N, and the two controls that validate it."""
    fig = slide("Protofilament number by competing whole models",
                "Ring templates, cropped micrograph: build at 12–15 protofilaments "
                "from the same 6DPU atoms, search with each, see which fits")

    panels = [
        ("CONTROL — truth is 13", {12: 8.236, 13: 10.328, 14: 8.239, 15: 7.420}, 13),
        ("CONTROL — truth is 14", {12: 7.675, 13: 7.872, 14: 10.093, 15: 7.891}, 14),
        ("REAL crop — unknown", {12: 9.277, 13: 12.928, 14: 9.038, 15: 8.251}, None),
    ]
    orders = np.array([12, 13, 14, 15])
    for i, (title, values, truth) in enumerate(panels):
        ax = fig.add_axes([0.045 + i * 0.185, 0.44, 0.155, 0.33])
        heights = [values[o] for o in orders]
        best = int(orders[int(np.argmax(heights))])
        ax.bar(orders, heights, 0.68,
               color=[GREEN if o == best else "#c8d6e5" for o in orders],
               edgecolor=BLUE, lw=0.7)
        ax.set_xticks(orders)
        ax.set_ylim(6.5, 14.2)
        ax.set_xlabel("N", fontsize=10)
        if i == 0:
            ax.set_ylabel("max z-score")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        if truth:
            ax.text(0.5, 1.10, "✓ recovers it", transform=ax.transAxes, ha="center",
                    fontsize=10, color=GREEN, fontweight="bold")
        ax.grid(alpha=0.25, ls=":", axis="y")

    _body(fig, 0.045, 0.38,
          "Radius is the lever: it scales with N at 8.45 Å per protofilament, because\n"
          "the lateral contact spacing belongs to the tubulin interface, not to N — a\n"
          "13-mer tube is narrower, not more loosely packed. All four templates share\n"
          "their atoms and B-factors, so unlike the expanded/compacted comparison there\n"
          "is no model-quality confound and no 2×2 is needed.", 0.55, 11.0)

    fig.add_artist(plt.Line2D([0.62, 0.62], [0.30, 0.78], color=LIGHT, lw=2))

    fig.text(0.645, 0.775, "What the controls are", fontsize=13,
             fontweight="bold", color=GREEN, va="top")
    _body(fig, 0.645, 0.720,
          "A control is a synthetic micrograph containing a\n"
          "microtubule of KNOWN protofilament number, at the\n"
          "same pixel size, defocus, SNR and position as the\n"
          "real one, so the identical search and constraint\n"
          "apply. The question is whether the competition\n"
          "returns the number that was put in.\n\n"
          "There are two because the 14-PF truth was built\n"
          "from the 14-PF template — a method that merely\n"
          "favoured whichever template made the truth would\n"
          "pass it. The 13 control breaks that symmetry.\n\n"
          "Both return their own truth, so the real panel\n"
          "is worth reading. It says 13.", 0.33, 11.0)

    note(fig, "Ring templates are used here even though each is N-fold "
              "pseudo-symmetric, which rules the ring out for the moiré route. It does "
              "not matter here: this test reads how well a\ntemplate FITS, not which φ "
              "it reports.  ·  Max z, not a per-site average: on data that is "
              "DEFINITIVELY 13-PF the paired mean for 13 against 14 gives t = 0.80 on "
              "55 of 127\nsites — a coin flip, because averaging 130 mostly-marginal "
              "sites dilutes it. Max z normally tracks template mass, but all four "
              "share their atoms, and the controls confirm it recovers the truth.")
    pdf.savefig(fig)
    plt.close(fig)


def load_refined_twist():
    """Per-protofilament azimuth tracks from the refined clean lattice sites.

    The clean sites are one detection per axial index, so they follow individual
    protofilaments up the tube. Refinement takes phi off the 3.495 deg search grid --
    without it each protofilament holds only one or two distinct angles and there is
    nothing to fit.
    """
    from leopard_em.analysis.filament_lattice import (
        TemplateLatticeGeometry,
        axis_points_from_peaks,
        filament_coordinates,
        fit_filament_axis,
    )

    geometry = TemplateLatticeGeometry.from_pdb(
        str(HERE / "models" / "6dpu_4_patches_aligned_zero.pdb")
    )
    table = pd.read_csv(HERE / "results_full" / "refined_clean_sites_patch4.csv")
    frame = pd.DataFrame({
        "x": table["refined_pos_x"],
        "y": table["refined_pos_y"],
        "phi": table["refined_phi"],
        "theta": table["refined_theta"],
        "psi": table["refined_psi"],
    })
    points = axis_points_from_peaks(frame, geometry, PIXEL_SIZE)
    axial = filament_coordinates(points, fit_filament_axis(points))[0] * PIXEL_SIZE
    axial = axial - axial.mean()
    phi = frame["phi"].to_numpy()
    weight = table["refined_scaled_mip"].to_numpy()

    order = np.argsort(phi)
    splits = np.where(np.diff(phi[order]) > 13.0)[0] + 1
    tracks = []
    for group in np.split(order, splits):
        if len(group) < 6 or axial[group].max() - axial[group].min() < 500.0:
            continue
        fit, cov = np.polyfit(axial[group], phi[group], 1, w=weight[group], cov=True)
        tracks.append({
            "s": axial[group],
            "phi": phi[group],
            "mean_phi": float(phi[group].mean()),
            "slope": float(fit[0] * 100.0),
            "sem": float(np.sqrt(cov[0, 0]) * 100.0),
        })
    return sorted(tracks, key=lambda t: t["mean_phi"])


def moire_strip(ax, period_nm: float | None, length_nm: float = 600.0,
                width_nm: float = 25.0, spacing_nm: float = 3.6) -> None:
    """Near and far wall protofilament gratings, superposed.

    In projection a tube shows both walls at once. Their protofilaments cross at twice
    the skew angle, so the two gratings beat against each other and the beat IS the
    moire fringe. At zero skew the gratings stay in register the whole way and there is
    no beat -- which is why 13 looks different in kind, not merely in period.
    """
    across = np.linspace(0, width_nm, 110)
    along = np.linspace(0, length_nm, 1600)
    grid_u, grid_v = np.meshgrid(across, along, indexing="ij")
    tilt = 0.0 if period_nm is None else spacing_nm / (2.0 * period_nm)
    image = (np.cos(2 * np.pi * (grid_u - grid_v * tilt) / spacing_nm)
             + np.cos(2 * np.pi * (grid_u + grid_v * tilt) / spacing_nm))
    ax.imshow(image, cmap="gray", aspect="auto", origin="lower",
              extent=(0, length_nm, 0, width_nm))
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)


def slide_moire_what(pdf: PdfPages) -> None:
    """What a moire fringe is, and why 13 is the one N without any."""
    fig = slide("The moiré route — what the fringes are",
                "An experienced eye reads protofilament number off a micrograph "
                "directly. This is what it is reading")

    cases = [(None, "N = 13", "protofilaments parallel to the axis — NO fringes"),
             (204.0, "N = 14", "204 nm fringe period"),
             (109.0, "N = 15", "109 nm fringe period")]
    for i, (period, label, caption) in enumerate(cases):
        ax = fig.add_axes([0.055, 0.585 - i * 0.175, 0.53, 0.115])
        moire_strip(ax, period)
        ax.set_ylabel(label, fontsize=11, color=INK, rotation=0, ha="right",
                      va="center", labelpad=12)
        ax.text(0.985, 0.90, caption, transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
        if i < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel("distance along the tube (nm)", fontsize=10)
        ax.axvline(326.0, color=AMBER, lw=2.6)

    fig.text(0.055, 0.155, "Orange line: 326 nm, the continuous tube in this "
             "micrograph. Under two fringe periods at N = 14,\nwhich is why the "
             "fringes themselves are not usable here — but the underlying skew "
             "still is.", fontsize=10.5, color=AMBER, va="top")

    _body(fig, 0.615, 0.775,
          "A projected tube shows BOTH walls at once. Each wall is a\n"
          "grating of protofilaments, and unless those run exactly\n"
          "parallel to the axis the two gratings are tilted in opposite\n"
          "senses — by twice the skew. Two gratings at a small relative\n"
          "angle beat against each other, and that beat is the moiré\n"
          "fringe. Its period is set by the skew, and the skew by N.\n\n"
          "13 is special. A 13₃ lattice closes with its protofilaments\n"
          "PARALLEL to the axis, so the two walls stay in register the\n"
          "whole length and there is no beat at all. Every other N must\n"
          "skew, so every other N must show fringes.\n\n"
          "So the fringes are not a nuisance pattern — they are a direct\n"
          "readout of N, and their ABSENCE is a readout too. This route\n"
          "is attractive because it is independent of everything else\n"
          "here: it needs no template competition, so it cannot be\n"
          "biased by template size or model quality.\n\n"
          "Measured next with the PATCH template, not the ring: a ring is\n"
          "14-fold pseudo-symmetric, so its φ is degenerate. The patch sits\n"
          "86.8 Å off the axis, which makes its roll angle real.", 0.36)

    note(fig, "Skew and fringe period from the measured monomer rise: −0.871°/175 nm "
              "at N = 12, zero/none at 13, +0.747°/204 nm at 14, +1.393°/109 nm at 15."
              "\nThe strips are schematic: the fringe periods are real, the grating "
              "spacing is chosen for visibility.", 0.045)
    pdf.savefig(fig)
    plt.close(fig)


def slide_moire_measured(pdf: PdfPages) -> None:
    """What 2DTM reads out instead of fringes, and the bound it gives."""
    fig = slide("The moiré route — what we actually read out",
                "Patch template, full micrograph: every detection already carries φ, "
                "so the skew can be measured without any fringes")

    tracks = load_refined_twist()

    ax = fig.add_axes([0.055, 0.255, 0.42, 0.50])
    palette = [BLUE, GREEN, "#7b1fa2", "#00838f"]
    for i, track in enumerate(tracks):
        ax.scatter(track["s"] / 10.0, track["phi"] - track["mean_phi"], s=32,
                   color=palette[i % len(palette)], alpha=0.9, edgecolor="none")
    span = np.array([-165.0, 165.0])
    for slope, label, colour in ((0.631, "what N = 14 requires", RED),
                                 (-0.859, "what N = 12 requires", AMBER)):
        ax.plot(span, slope * span / 10.0, ls="--", lw=2.2, color=colour, label=label)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("position along the tube (nm)")
    ax.set_ylabel("φ − ⟨φ⟩ for that protofilament  (deg)")
    ax.set_ylim(-16, 16)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.text(0.02, 0.03, "one colour per protofilament", transform=ax.transAxes,
            fontsize=9.5, color=MUTED)
    ax.set_title("Four protofilaments, followed up the tube", fontsize=11.5,
                 color=INK, loc="left")
    ax.grid(alpha=0.25, ls=":")

    combined = np.array([t["slope"] for t in tracks])
    weights = 1.0 / np.array([t["sem"] for t in tracks]) ** 2
    mean = float((combined * weights).sum() / weights.sum())
    err = max(float(np.sqrt(1.0 / weights.sum())),
              float(combined.std(ddof=1) / np.sqrt(len(combined))))

    _body(fig, 0.515, 0.775,
          "THE DATA: clean lattice sites from the PATCH search, one\n"
          "detection per axial index, grouped by azimuth — which sorts\n"
          "them into individual protofilaments. Each point is one\n"
          "subunit: position along the tube against its own roll angle φ.\n"
          "A skewed protofilament climbs in φ; a parallel one does not.\n\n"
          "This needs φ off the search grid. The grid steps 3.495°, and\n"
          "each protofilament held only 1–3 distinct values — a staircase,\n"
          "not a slope. refine_template at a 0.4° step gives 9–12 values\n"
          "each, with 0.45–0.69° of scatter, and mean z rises 13.3 → 15.5.\n\n"
          f"MEASURED: {mean:+.3f} ± {err:.3f} °/100 Å across "
          f"{len(tracks)} protofilaments.\n"
          "The dashed lines show what 12 and 14 would have to look like.\n\n"
          "The small residual is NOT a supertwist, because it is not\n"
          "constant: the two halves of a track disagree and 2 of 4 have a\n"
          "significant quadratic term. That is a gently bent tube.", 0.42)

    for i, line in enumerate((
        "So this is a BOUND, not a detection.",
        "Total excursion 3–5°; 14 needs +20.5°, 12 needs −28°, 15 needs +36°.",
        "It excludes 12, 14 and 15, and is consistent with 13.",
    )):
        fig.text(0.515, 0.215 - i * 0.036, line, fontsize=11, color=GREEN,
                 fontweight="bold", va="top")

    note(fig, "The RING template cannot do this — it is 14-fold pseudo-symmetric, so "
              "its φ is degenerate. No ground truth yet either: the synthetic controls "
              "tile ONE projection\nat a fixed repeat, so they carry zero supertwist "
              "whichever N built them. Sign follows the axis convention.", 0.045)
    pdf.savefig(fig)
    plt.close(fig)


def paired_lattice_effect(stems, radius=3):
    """Expanded minus compacted, per matched site, for one pair of templates."""
    import sys

    sys.path.insert(0, str(HERE / "run_scripts"))
    import analyse_lattice_competition as A  # noqa: PLC0415

    runs = A.load_stems(list(stems))
    sites = A.union_sites(runs)
    z = {k: A.sample(r["zmap"], sites, radius) for k, r in runs.items()}
    diff = z[stems[0]] - z[stems[1]]
    sem = diff.std(ddof=1) / np.sqrt(len(diff))
    return len(sites), float(diff.mean()), float(sem), int((diff > 0).sum())


def slide_spacing_revisited(pdf: PdfPages) -> None:
    """Is the spacing answer an artefact of using 14-protofilament templates?"""
    fig = slide("Lattice spacing revisited, at 13 protofilaments",
                "Ring templates, cropped micrograph: the spacing comparison was built "
                "entirely at N = 14 — does it survive if the tube is really 13?")

    pairs = [
        ("N = 14", ("6dpu_atoms_6dpu_lattice_2rings_flatB",
                    "6dpu_atoms_6dpv_lattice_2rings_flatB"), BLUE),
        ("N = 13", ("6dpu_13pf_2rings_flatB",
                    "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB"), GREEN),
    ]
    results = [(label, colour) + paired_lattice_effect(stems)
               for label, stems, colour in pairs]

    ax = fig.add_axes([0.045, 0.26, 0.29, 0.50])
    for i, (label, colour, n, mean, sem, pos) in enumerate(results):
        ax.bar(i, mean, 0.55, yerr=sem, capsize=6, color=colour)
        ax.text(i, mean + sem + 0.035, f"t = {mean / sem:.1f}", ha="center",
                fontsize=11, fontweight="bold", color=colour)
        ax.text(i, 0.03, f"{pos}/{n} sites", ha="center", fontsize=10, color="white",
                fontweight="bold")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r[0] for r in results], fontsize=12)
    ax.set_ylabel("expanded − compacted  (z)")
    ax.set_ylim(0, 0.78)
    ax.set_title("same answer, same size", fontsize=11.5, color=INK, loc="left")
    ax.grid(alpha=0.25, ls=":", axis="y")

    delta = results[1][3] - results[0][3]
    delta_sem = np.hypot(results[0][4], results[1][4])

    _body(fig, 0.40, 0.78,
          "Every template in the original spacing comparison was built at 14\n"
          "protofilaments. If the tube is really 13, all four were mismatched —\n"
          "so the obvious worry is that the expanded answer came out of that\n"
          "mismatch rather than out of the specimen.\n\n"
          "Rebuilt the pair at 13 protofilaments: the same 6DPU atoms, the same\n"
          "B-factors, the same radius, with only the repeat changed. Two\n"
          "searches rather than four, because with identical atoms on both\n"
          "sides no model-quality confound can arise and no 2×2 is needed.\n\n"
          f"Expanded wins at both: {results[0][3]:+.3f} ± {results[0][4]:.3f} z at "
          f"N = 14 and\n"
          f"{results[1][3]:+.3f} ± {results[1][4]:.3f} z at N = 13. The difference "
          f"between them is\n"
          f"{delta:+.3f} ± {delta_sem:.3f} — consistent with zero, and both agree on "
          "89%\nof sites.\n\n"
          "So the spacing measurement does not depend on getting the\n"
          "protofilament number right. That removes the objection rather than\n"
          "replacing one answer with another — which is the more useful\n"
          "outcome, because it means the spacing result stands whichever way\n"
          "the protofilament question eventually goes.", 0.55)

    note(fig, "This is a better-conditioned test OF SPACING, not a test of "
              "protofilament number: both cells are 13-PF, so nothing here bears on N. "
              "Built with build_mt_templates.py\n--protofilaments 13 --atoms 6dpu "
              "--lattice 6dpv, which takes the atoms from one deposition and the repeat "
              "from the other.")
    pdf.savefig(fig)
    plt.close(fig)



def slide_seam_problem(pdf: PdfPages) -> None:
    """What locating a seam actually requires, and the four routes to it."""
    fig = slide("The seam — what would have to be measured",
                "A 13₃ dimer lattice cannot close without one, but finding it is a "
                "different problem from knowing it is there")

    ax = fig.add_axes([0.045, 0.235, 0.375, 0.545])
    ax.set_xlim(-0.9, 16.8)
    ax.set_ylim(-0.9, 7.3)
    ax.axis("off")
    ax.set_title("The lattice, unrolled — and it does not close",
                 fontsize=11.5, color=INK, loc="left")

    # Each protofilament rises 3/N of a monomer relative to its left-hand neighbour,
    # so N of them climb exactly 3 monomers whatever N is. Three is odd, so the DIMER
    # register arrives back half a dimer out. The last column is column 1 redrawn
    # where the lattice says it must sit, which is what makes the mismatch visible.
    n_pf = 13
    rise, mono = 3.0 / n_pf, 0.5
    for pf in range(n_pf + 1):
        shift = pf * rise
        # Going once round climbs 3 monomers. Three is ODD, so the protofilament you
        # arrive back at presents the other subunit: the last column is PF 1 again, and
        # its colours are flipped. That colour change IS the seam -- everywhere else a
        # lateral neighbour is the same type of tubulin.
        flip = 1 if pf == n_pf else 0
        for m in range(-2, 18):
            y = (m + shift) * mono
            if not -0.05 < y < 6.5:
                continue
            ax.add_patch(mpatches.Rectangle(
                (pf, y), 0.84, mono * 0.92,
                facecolor=BLUE if (m + flip) % 2 == 0 else "#a8c6e0",
                edgecolor="white", lw=0.7))
        for j in range(-1, 9):  # dimer outlines: the register that fails to close
            y = (2 * j + shift) * mono
            if not -0.05 < y < 6.5 - 2 * mono:
                continue
            ax.add_patch(mpatches.Rectangle(
                (pf, y), 0.84, 2 * mono * 0.96, facecolor="none",
                edgecolor=INK, lw=1.1))

    ax.add_artist(plt.Line2D([n_pf - 0.07, n_pf - 0.07], [-0.05, 6.6],
                             color=RED, lw=3.0))
    ax.text(n_pf - 0.07, 6.72, "seam", fontsize=12, color=RED, fontweight="bold",
            ha="center", va="bottom")

    # A dimer boundary of protofilament 1, carried across to where the same
    # protofilament comes back round: it arrives half a dimer out.
    ax.add_artist(plt.Line2D([-0.3, 15.3], [3.0, 3.0], color=RED, lw=1.2, ls="--"))
    ax.add_artist(plt.Line2D([n_pf - 0.1, 15.3], [3.5, 3.5], color=RED, lw=1.2,
                             ls="--"))
    ax.annotate("", xy=(15.15, 3.5), xytext=(15.15, 3.0),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.7))
    ax.text(15.45, 3.25, "½ dimer\nout", fontsize=10.5, color=RED, va="center",
            fontweight="bold")

    ax.text(0.42, -0.2, "PF 1", fontsize=10, color=MUTED, ha="center", va="top")
    ax.text(n_pf + 0.42, -0.2, "PF 1 again", fontsize=10, color=MUTED,
            ha="center", va="top")
    ax.text(0.0, -0.66, "α dark   ·   β light   ·   boxes outline dimers",
            fontsize=10, color=MUTED, va="top")

    _body(fig, 0.50, 0.775,
          "Every protofilament sits 9.7 Å above its left-hand neighbour, so going\n"
          "once round the tube climbs 13 × 9.7 = 126 Å. That is exactly 3 monomers —\n"
          "and 3 is ODD, so the protofilament you arrive back at presents the OTHER\n"
          "subunit. Everywhere else α meets α and β meets β; at that one contact α\n"
          "meets β. That is the seam, and where it sits is what we want.\n\n"
          "The climb is 126 Å for ANY N, because the lateral offset is 3u/N: the\n"
          "mismatch is a property of the lattice, not of the protofilament count.\n\n"
          "The difficulty is not that the seam is subtle. It is that finding it means\n"
          "telling α from β.", 0.45)

    note(fig, "A seam certainly EXISTS — the geometry forces one. The question is "
              "only where it sits on this tube, and whether 2DTM can say. Route 1 was "
              "long recorded as closed; it is not.")
    pdf.savefig(fig)
    plt.close(fig)


REGISTER_SEARCHES = {
    "patch": ("output_correlation_table_patch4_full.h5",
              "6dpu_4_patches_aligned_zero.pdb"),
    "ring": ("output_correlation_table_2rings_full.h5",
             "6dpu_2rings_aligned_zero.pdb"),
}


def load_register_alternation(which: str = "patch", dimer: bool = False):
    """Site scores against axial index, for the register-alternation test.

    Sites are indexed at the MONOMER spacing, so consecutive indices differ by exactly
    one monomer: one parity has the dimer template in register, the other has it one
    monomer out. Passing dimer=True indexes at the dimer instead, which puts every site
    in the same register and must therefore show no alternation.
    """
    from leopard_em.analysis.filament_lattice import (
        TemplateLatticeGeometry,
        extract_lattice_sites,
    )
    from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

    table, model = REGISTER_SEARCHES[which]
    geometry = TemplateLatticeGeometry.from_pdb(str(HERE / "models" / model))
    detections = detections_from_hdf5(
        str(HERE / "results_full" / table), min_z_score=7.0
    ).reset_index(drop=True)
    sites = extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=9.0,
        initial_rise_angstrom=83.958 if dimer else None,
    )
    index = np.asarray(sites.axial_index).astype(int)
    score = np.asarray(sites.score)
    order = np.argsort(index)
    return index[order], score[order]


def harmonic_amplitude(index, score, rng, n_null=2000):
    """Period-2 amplitude of score against index, with a shuffled null."""
    weight = score - score.mean()
    phase = np.exp(1j * np.pi * index)
    amplitude = float(np.abs((weight * phase).sum()) / len(index))
    null = np.percentile(
        [float(np.abs((rng.permutation(weight) * phase).sum()) / len(index))
         for _ in range(n_null)], 95)
    return amplitude, float(null)


def slide_seam_register(pdf: PdfPages) -> None:
    """Does the real data show the register alternation the simulation predicts?"""
    fig = slide("Seam route 1 — does the real data see it?",
                "Patch template, full micrograph: the top-down sites are indexed at "
                "the MONOMER spacing, so this test comes for free")

    rng = np.random.default_rng(0)
    index, score = load_register_alternation()
    amplitude, null = harmonic_amplitude(index, score, rng)
    d_index, d_score = load_register_alternation(dimer=True)
    d_amplitude, d_null = harmonic_amplitude(d_index, d_score, rng)

    ax = fig.add_axes([0.045, 0.26, 0.36, 0.50])
    window = index < index.min() + 34
    even = window & (index % 2 == 0)
    odd = window & (index % 2 == 1)
    ax.plot(index[window], score[window], lw=1.0, color=MUTED, zorder=1)
    ax.scatter(index[even], score[even], s=58, color=BLUE, zorder=3,
               label="one register")
    ax.scatter(index[odd], score[odd], s=58, color=AMBER, zorder=3,
               label="one monomer out")
    ax.set_xlabel("site number (one monomer apart)")
    ax.set_ylabel("z-score")
    ax.set_ylim(top=score[window].max() + 2.2)
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)
    ax.set_title("The score alternates, site by site", fontsize=11.5, color=INK,
                 loc="left")
    ax.grid(alpha=0.25, ls=":")

    ax2 = fig.add_axes([0.455, 0.26, 0.155, 0.50])
    ax2.bar([0, 1], [amplitude / null, d_amplitude / d_null], 0.58,
            color=[BLUE, "#c8d6e5"], edgecolor=BLUE, lw=0.8)
    ax2.axhline(1.0, color=RED, lw=2.0)
    ax2.text(-0.42, 1.06, "chance", fontsize=9.5, color=RED, va="bottom", ha="left")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["monomer\nspacing", "dimer spacing\n(control)"], fontsize=9.5)
    ax2.set_ylabel("alternation ÷ chance")
    ax2.set_ylim(0, 2.7)
    ax2.set_title("and vanishes in the control", fontsize=11.5, color=INK, loc="left")
    ax2.grid(alpha=0.25, ls=":", axis="y")

    _body(fig, 0.645, 0.775,
          "THE TEST, free from the top-down extraction. Sites are\n"
          "indexed at the monomer spacing, so consecutive sites\n"
          "differ by exactly one monomer: alternate ones have the\n"
          "dimer template one monomer out of register, β on α.\n\n"
          "Sites one monomer apart differ by 1.35 z on a mean of\n"
          "13.3 — about 10% of the score, over 40 pairs. The\n"
          f"period-2 amplitude is {amplitude:.2f} against a chance level of\n"
          f"{null:.2f}, and it holds at every detection threshold and\n"
          "bootstrap cut tried.\n\n"
          "THE CONTROL. Index at the DIMER spacing instead and\n"
          "every site is in the SAME register: the alternation\n"
          f"vanishes ({d_amplitude:.3f} against {d_null:.2f}, t = 0.07). So it is the\n"
          "specimen, not the extraction.\n\n"
          "So YES — 2DTM can tell which monomer it is on. 10% sits\n"
          "between the simulated 9.8% at 4 Å and 16.7% at full.", 0.33, 10.5)

    note(fig, "CHANCE is a permutation null. The statistic is the period-2 amplitude "
              "|Σ (z − mean z)·(−1)^index| / N, the alternating sum of mean-centred "
              "scores. Shuffling the scores\namong the sites keeps every score and "
              "every index but destroys the pairing between them; the line is the 95th "
              "percentile of 2000 shuffles, and 0 of 3000 reached\nthe observed value. "
              "The two nulls differ as each scales with score variance and site count, "
              "so the ratio is plotted.  ·  Use the PATCH: the ring gives 3%, below "
              "its own chance level.", 0.045)
    pdf.savefig(fig)
    plt.close(fig)


def slide_next(pdf: PdfPages) -> None:
    """The two things worth doing next."""
    fig = slide("Where to go next", "Two questions, both about whether these "
                                    "measurements survive contact with real data")

    items = [
        ("1", "Test on in-situ-like data, with semi ground truth", GREEN,
         "Everything here rests on one filament in one clean micrograph. Lattice "
         "spacing, protofilament\n"
         "number and the monomer register all have to be shown to work where the "
         "specimen is crowded\n"
         "and the contrast is worse.\n\n"
         "Semi ground truth is what makes it a TEST rather than an application: a "
         "specimen whose answer\n"
         "is independently known, so a wrong answer shows as wrong. Agreement between "
         "two of our own\n"
         "methods is not evidence — they share the templates, the axis fit and the "
         "assumptions."),
        ("2", "Find the spatial resolution of the lattice-spacing measurement", BLUE,
         "The spacing is currently ONE number for a whole filament: 41.864 Å, 0.27% "
         "from expanded.\n"
         "The question that matters is whether it is the same number everywhere along "
         "the tube.\n\n"
         "Compaction is expected to vary — a GTP cap, a lattice defect or a binding "
         "partner would each\n"
         "show as a local change. So what to measure is not the spacing but the LENGTH "
         "SCALE over\n"
         "which it can be resolved: how short a stretch of tube still separates expanded "
         "from compacted.\n"
         "That is a precision-against-track-length curve, and it turns one number into a "
         "profile."),
    ]

    y = 0.775
    for number, title, colour, body in items:
        fig.add_artist(plt.Circle((0.068, y - 0.017), 0.019, color=colour,
                                  transform=fig.transFigure, zorder=3))
        fig.text(0.068, y - 0.017, number, fontsize=13, fontweight="bold",
                 color="white", ha="center", va="center", zorder=4)
        fig.text(0.105, y, title, fontsize=15, fontweight="bold", color=colour,
                 va="top")
        fig.text(0.105, y - 0.058, body, fontsize=11.5, color=INK, va="top",
                 linespacing=1.6)
        y -= 0.345

    note(fig, "Both are cheaper than more model building. The first needs a specimen, "
              "not a method; the second needs only the analysis already written, run on "
              "sub-sections of the track.")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="microtubule_2dtm_status.pdf")
    args = parser.parse_args()

    print("Loading data")
    det = load_detections()
    sites = load_sites(det)
    z, site_positions = load_competition()
    print(f"  {len(det):,} detections, {sites.n_predicted} lattice sites, "
          f"{len(site_positions)} competition sites")

    out = HERE / args.output
    with PdfPages(out) as pdf:
        slides = [
            slide_title,
            slide_micrograph,
            slide_constraint,
            slide_what_is_2dtm,
            slide_templates,
            slide_mip_problem,
            slide_orientation_peaks,
            slide_topdown,
            slide_unwrap,
            lambda p: slide_polarity(p, det),
            lambda p: slide_spacing_direct(p, sites),
            lambda p: slide_spacing_competition(p, z, len(site_positions)),
            slide_pf_competition,
            slide_moire_what,
            slide_moire_measured,
            slide_spacing_revisited,
            slide_seam_problem,
            slide_seam_register,
            slide_next,
        ]
        for make in slides:
            make(pdf)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(slides)} slides)")


if __name__ == "__main__":
    main()
