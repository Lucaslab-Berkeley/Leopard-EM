"""Continuous lattice spacing along each tube, from every lattice site at the right register.

Stage 6b reads the lattice in segments, and only on a tube whose sites fit ONE continuous
lattice end to end (<= 7 A rms). Long curved tubes do not: on GDP 10 of 13 sat 7-11.5 A
off it, so 6b gave them a competition call and no rise. The misfit is not the lattice;
it is the bookkeeping. The every-site step predicts each (axial index, protofilament)
from one whole-tube lattice, and where that lattice drifts more than half a monomer from
the real one, refine walks the site onto the neighbouring subunit and the label slips.
The refined POSITIONS are fine. So this script ignores the labels for measuring and
re-indexes the refined positions themselves, then reads the spacing locally.

Nothing is re-run: it reads what ``lattice_fill.py sites`` and ``stage7_every_site.py``
already wrote for a patch table.

Per filament:

1. **Register.** Stage 7 refined both patches (in-register, one monomer out) at every
   site. Per tube, the parity at which the in-register patch scores higher is read from
   the data; each site takes the refined pose of the patch that matches its parity
   (``--register in`` uses the in-register patch everywhere, for comparison).
2. **Sites.** Every lattice site whose mean z clears the decoys' 99th percentile and that
   refine did not push to its search-window edge (``--found-only``: Stage 4's sites).
3. **Axial coordinate.** Each site's axis point (the patch sits 86 A off the axis), a
   smooth curve through them (``fit_axis_curve``: knots >= 10 repeats, so it carries no
   power at the lattice period), arc length along it, and the helical unwrap by roll,
   which puts every protofilament on one 1-D lattice.
4. **Re-index.** On that coordinate the subunits are ~40.8 A apart and a subunit's sites
   agree to 2-3 A, so they are clustered directly (density peaks; a site joins the
   nearest peak within ``CAPTURE`` of a repeat) and the clusters are numbered by the
   gaps between them. A slip in the predicted labels cannot survive this.
5. **Ring average.** Each cluster's position is the clipped mean of its sites (all
   protofilaments at that axial position); its tilt is their mean sin(theta).
6. **Local spacing.** In a window of ``--window`` lattice indices centred on each index:
   the slope of cluster position against index, divided by the window's sin(theta).
   Error: the tube's own scatter about its slow trend, inflated for neighbour
   correlation, through 6b's formula (``RiseModel.expected_error``).

A sharp lattice boundary comes out as a ramp ``--window`` indices long, so a gradient in
the profile is only a gradient if it is longer than the window. ``--truth`` (the strong
synthetic, whose boundaries are sharp and known) scores the profile against the truth
seen through the same window, with pulls.

Outputs, in the table's folder:
    spacing_clusters_<label>.csv    one row per subunit cluster (ring-averaged)
    spacing_profile_<label>.csv     one row per window centre
    spacing_summary_<label>.csv     per filament (and per truth class with --truth)

Usage:
    python run_scripts/spacing_profile.py \\
        --table results_bundle/output_correlation_table_bundle_strong_patch.h5 \\
        --truth Frames/synthetic_bundle_lattice_domains_strong.truth.json
    python run_scripts/spacing_profile.py \\
        --table results_svd_pipeline/output_correlation_table_pipe_gdp_autotrace_fil7-8-9-10_14pf_patch.h5
"""

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    unwrap_helical_axial_coordinate,
)

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from continuous_unwrap_curved import fit_axis_curve, project  # noqa: E402
from frame_geometry import frame_geometry  # noqa: E402
from stage6b_local_lattice import (  # noqa: E402
    C_REPEAT,
    E_REPEAT,
    detrended,
    variance_inflation,
)

DEFAULT_WINDOW = 12   # lattice indices (~49 nm); tube 4 of the synthetic flips every ~15
COVERAGE = 0.75       # a window needs this fraction of its indices present
CAPTURE = 0.35        # a site joins a cluster within this fraction of a repeat
MIN_CLUSTER = 3       # sites; and at least MIN_CLUSTER_FRACTION of the tube's median
MIN_CLUSTER_FRACTION = 0.25
DENSITY_BIN_A = 0.25
DENSITY_SIGMA_A = 2.0


def parse_child_labels(tag: str) -> dict[int, int] | None:
    """A split run renumbers its filaments 1..k: ``..._fil7-8-9-10_14pf`` -> {1: 7, ...}."""
    for part in tag.split("_"):
        if part.startswith("fil") and part[3:].replace("-", "").isdigit():
            return {i + 1: int(v) for i, v in enumerate(part[3:].split("-"))}
    return None


def load(results: pathlib.Path, tag: str, label: str):
    """Every-site rows (non-decoy), both refines row for row, and the decoy z cut."""
    sites = pd.read_csv(results / f"lattice_fill_sites_{tag}.csv", index_col=0)
    for column in ("detected", "decoy"):
        sites[column] = sites[column].astype(bool)
    scores = pd.read_csv(results / f"stage7_every_register_{label}.csv", index_col=0)
    real = sites[~sites["decoy"]]
    refined = {}
    for side in ("in", "out"):
        frame = pd.read_csv(results / f"stage7_refined_{label}_every_{side}.csv",
                            index_col=0)
        if len(frame) != len(real) or not np.array_equal(
                frame["pos_x"].to_numpy(), real["pos_x"].to_numpy()):
            raise SystemExit(f"stage7_refined_{label}_every_{side}.csv is not the sites "
                             "row for row")
        frame.index = real.index
        refined[side] = frame
    mean_z = (scores["in"] + scores["out"]) / 2
    cut = float(np.percentile(mean_z[scores["decoy"].astype(bool)], 99))
    return real, scores.loc[real.index], refined, cut


def template_geometry(refined: pd.DataFrame) -> tuple[str, TemplateLatticeGeometry]:
    """The patch the sites were refined with, from refine's own template path."""
    name = pathlib.Path(refined["template_path"].iloc[0]).name
    stem = name.split("_bscale")[0].rsplit("_", 1)[0]
    return stem, TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))


def choose_register(part, scores, refined, mode):
    """Per site, (refined row, its z) from the patch matching its parity."""
    index = part["axial_index"].to_numpy()
    if mode == "in":
        use_in = np.ones(len(part), dtype=bool)
        in_even = True
    else:
        diff = (scores.loc[part.index, "in"] - scores.loc[part.index, "out"]).to_numpy()
        even = index % 2 == 0
        in_even = bool(np.nanmean(diff[even]) > np.nanmean(diff[~even]))
        use_in = even == in_even
    rows = pd.concat([refined["in"].loc[part.index[use_in]],
                      refined["out"].loc[part.index[~use_in]]]).loc[part.index]
    z = np.where(use_in, scores.loc[part.index, "in"], scores.loc[part.index, "out"])
    edge = np.where(use_in, scores.loc[part.index, "in_at_edge"],
                    scores.loc[part.index, "out_at_edge"]).astype(bool)
    return rows, z, edge, in_even


def axial_coordinate(rows, z, labels, geometry, pixel_size):
    """Per site: unwrapped axial position (A), axis point (img px), sin(theta)."""
    points = axis_points_from_peaks(
        rows, geometry, pixel_size,
        position_columns=("refined_pos_x_img", "refined_pos_y_img"))
    hint = labels.astype(float) * geometry.rise_angstrom / pixel_size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve, arc, _ = fit_axis_curve(points, hint, np.clip(z, 0.1, None),
                                       geometry.rise_angstrom, pixel_size)
    along = project(curve, arc, points) * pixel_size
    if np.corrcoef(along, labels)[0, 1] < 0:   # run the same way as the labels
        along = along.max() - along
    unwrapped = unwrap_helical_axial_coordinate(along, rows["refined_phi"].to_numpy(),
                                                geometry)
    sines = np.sin(np.radians(rows["refined_theta"].to_numpy()))
    return unwrapped, points, sines, curve


def clusters(unwrapped, repeat):
    """(cluster id per site or -1, peak positions): density peaks along the axis."""
    lo = unwrapped.min() - 3 * repeat
    edges = np.arange(lo, unwrapped.max() + 3 * repeat, DENSITY_BIN_A)
    density, _ = np.histogram(unwrapped, edges)
    density = gaussian_filter1d(density.astype(float), DENSITY_SIGMA_A / DENSITY_BIN_A)
    peaks, _ = find_peaks(density, distance=int(0.6 * repeat / DENSITY_BIN_A))
    centres = edges[peaks] + DENSITY_BIN_A / 2
    for _ in range(3):   # assign, drop thin peaks, re-centre
        nearest = np.abs(unwrapped[:, None] - centres[None, :])
        member = np.argmin(nearest, axis=1)
        member[nearest[np.arange(len(member)), member] > CAPTURE * repeat] = -1
        counts = np.bincount(member[member >= 0], minlength=len(centres))
        floor = max(MIN_CLUSTER, MIN_CLUSTER_FRACTION * np.median(counts[counts > 0]))
        keep = counts >= floor
        centres = np.array([np.median(unwrapped[member == i])
                            for i in np.flatnonzero(keep)])
    nearest = np.abs(unwrapped[:, None] - centres[None, :])
    member = np.argmin(nearest, axis=1)
    member[nearest[np.arange(len(member)), member] > CAPTURE * repeat] = -1
    return member, centres


def number_clusters(centres, repeat):
    """Lattice index of each (sorted) cluster from the gaps: a gap of ~2 repeats is a
    missing subunit, not a long one. The local repeat is the median nearby single gap."""
    order = np.argsort(centres)
    gaps = np.diff(centres[order])
    single = (gaps > 0.75 * repeat) & (gaps < 1.25 * repeat)
    index = np.zeros(len(centres), dtype=int)
    for i, gap in enumerate(gaps):
        near = np.flatnonzero(single)
        near = near[np.abs(near - i) <= 10]
        local = np.median(gaps[near]) if len(near) else repeat
        index[i + 1] = index[i] + max(1, int(round(gap / local)))
    out = np.empty(len(centres), dtype=int)
    out[order] = index
    return out


def ring_average(unwrapped, member, n_clusters):
    """Clipped mean position per cluster (sites beyond 3 robust sd of the median out)."""
    position = np.full(n_clusters, np.nan)
    for i in range(n_clusters):
        values = unwrapped[member == i]
        if len(values) == 0:
            continue
        centre = np.median(values)
        spread = max(1.4826 * np.median(np.abs(values - centre)), 1.0)
        position[i] = values[np.abs(values - centre) <= 3 * spread].mean()
    return position


def local_spacing(index, position, sines, window):
    """Per window centre: (centre, n, projected slope, sin, residual sd) -- full windows."""
    rows = []
    half = window / 2
    for centre in range(int(index.min() + half), int(index.max() - half + 1)):
        inside = (index >= centre - half) & (index < centre + half)
        if inside.sum() < COVERAGE * window:
            continue
        slope, intercept = np.polyfit(index[inside], position[inside], 1)
        residual = position[inside] - (slope * index[inside] + intercept)
        rows.append((centre, int(inside.sum()), slope, float(np.mean(sines[inside])),
                     float(residual.std(ddof=2))))
    return rows


def truth_lattice(points_img, truth):
    """Known lattice (C/E) at each point: the nearest simulated block's."""
    xy, lattice = [], []
    for tube in truth["tubes"]:
        for block in tube["blocks"]:
            xy.append((block["x"], block["y"]))
            lattice.append(block["lattice"])
    _, nearest = cKDTree(np.asarray(xy)).query(points_img)
    return np.asarray(lattice)[nearest]


# pylint: disable=too-many-locals,too-many-statements
def profile_filament(fil, part, scores, refined, cut, geometry, pixel_size, args,
                     truth):
    """Clusters and window profile of one filament, or None if too few sites."""
    rows, z, edge, in_even = choose_register(part, scores, refined, args.register)
    mean_z = ((scores.loc[part.index, "in"] + scores.loc[part.index, "out"]) / 2).to_numpy()
    keep = (mean_z >= cut) & ~edge
    if args.found_only:
        keep &= part["detected"].to_numpy()
    if keep.sum() < 50:
        print(f"  filament {fil}: {keep.sum()} usable sites, skipped")
        return None
    rows, z = rows[keep], z[keep]
    labels = part["axial_index"].to_numpy()[keep]
    unwrapped, points, sines, _ = axial_coordinate(rows, z, labels, geometry, pixel_size)
    repeat = geometry.rise_angstrom * float(np.median(sines))
    member, centres = clusters(unwrapped, repeat)
    index = number_clusters(centres, repeat)
    position = ring_average(unwrapped, member, len(centres))
    cl = pd.DataFrame({"filament": fil, "index": index, "position_A": position,
                       "n_sites": np.bincount(member[member >= 0], minlength=len(centres))})
    for name, values in (("sin_theta", sines), ("x_img", points[:, 0]),
                         ("y_img", points[:, 1]), ("z", z),
                         ("label_index", labels.astype(float))):
        cl[name] = [float(np.median(values[member == i])) if name in ("z", "label_index")
                    else float(np.mean(values[member == i])) for i in range(len(centres))]
    cl = cl.sort_values("index").reset_index(drop=True)
    within = unwrapped[member >= 0] - centres[member[member >= 0]]

    # noise model: deviations from the tube's straight lattice, slow trend removed
    a, b = np.polyfit(cl["index"], cl["position_A"], 1)
    deviation = cl["position_A"].to_numpy() - (a * cl["index"].to_numpy() + b)
    sigma = float(detrended(deviation).std(ddof=1))
    inflation = variance_inflation(deviation)
    prof = pd.DataFrame(local_spacing(cl["index"].to_numpy(), cl["position_A"].to_numpy(),
                                      cl["sin_theta"].to_numpy(), args.window),
                        columns=["centre_index", "n", "projected", "sin_theta",
                                 "residual_sd"])
    if prof.empty:
        print(f"  filament {fil}: no full window of {args.window}")
        return None
    prof.insert(0, "filament", fil)
    prof["spacing_A"] = prof["projected"] / prof["sin_theta"]
    n = prof["n"].to_numpy(dtype=float)
    prof["se_A"] = (sigma * np.sqrt(12.0 / (n * (n * n - 1.0))) * np.sqrt(inflation)
                    / prof["sin_theta"])
    lookup = cl.set_index("index")
    centre_rows = lookup.reindex(prof["centre_index"]).interpolate(limit_direction="both")
    for column in ("x_img", "y_img", "position_A"):
        prof[column] = centre_rows[column].to_numpy()
    prof["arc_nm"] = (prof["position_A"] - cl["position_A"].iloc[0]) / 10.0

    # split-half check, no noise model: the same clusters ring-averaged over the even
    # and the odd protofilaments separately. Their window spacings share the lattice but
    # not the site noise, so their agreement says which wiggles are real, and
    # sd(a - b) / 2 is the full profile's error if the halves' noise is independent.
    pf = part["protofilament"].to_numpy()[keep]
    order = np.argsort(index)
    rank = np.empty(len(index), dtype=int)
    rank[order] = np.arange(len(index))
    halves = {}
    for name_, mask in (("a", pf % 2 == 0), ("b", pf % 2 == 1)):
        sub = np.where(mask, member, -1)
        pos = ring_average(unwrapped, sub, len(centres))
        sin_half = np.array([np.mean(sines[sub == i]) if np.any(sub == i) else np.nan
                             for i in range(len(centres))])
        ok = ~np.isnan(pos)
        windows = local_spacing(index[ok], pos[ok], sin_half[ok], args.window)
        halves[name_] = pd.Series({c: s / sn for c, _, s, sn, _ in windows})
    prof["spacing_half_a"] = halves["a"].reindex(prof["centre_index"]).to_numpy()
    prof["spacing_half_b"] = halves["b"].reindex(prof["centre_index"]).to_numpy()
    both = prof[["spacing_half_a", "spacing_half_b"]].dropna()
    half_r = (float(np.corrcoef(both["spacing_half_a"], both["spacing_half_b"])[0, 1])
              if len(both) > 5 else float("nan"))
    se_split = (float((both["spacing_half_a"] - both["spacing_half_b"]).std() / 2)
                if len(both) > 5 else float("nan"))
    if truth is not None:
        cl["truth"] = truth_lattice(cl[["x_img", "y_img"]].to_numpy(), truth)
        repeat_of = np.where(cl["truth"] == "C", C_REPEAT, E_REPEAT)
        # truth positions: the cumulative true repeat, counted over the same indices
        steps = np.diff(cl["index"].to_numpy()) * repeat_of[1:]
        true_position = np.concatenate([[0.0], np.cumsum(steps)])
        expected, purity = [], []
        half = args.window / 2
        for centre in prof["centre_index"]:
            inside = ((cl["index"] >= centre - half) & (cl["index"] < centre + half)).to_numpy()
            expected.append(np.polyfit(cl["index"][inside], true_position[inside], 1)[0])
            share = float(np.mean(cl["truth"].to_numpy()[inside] == "C"))
            purity.append(share)
        prof["truth_A"] = expected
        prof["truth_C_share"] = purity
        prof["pull"] = (prof["spacing_A"] - prof["truth_A"]) / prof["se_A"]
    summary = {
        "filament": fil, "register_in_even": in_even, "sites_used": int(keep.sum()),
        "sites_total": len(part), "clusters": len(cl),
        "index_span": int(cl["index"].max() - cl["index"].min() + 1),
        "sites_unclustered": int((member < 0).sum()),
        "within_cluster_sd_A": float(1.4826 * np.median(np.abs(within))),
        "label_slips": int(np.sum(np.diff(cl["index"] - cl["label_index"].round()) != 0)),
        "whole_tube_spacing_A": float(a / np.mean(cl["sin_theta"])),
        "sigma_A": sigma, "inflation": inflation,
        "median_se_A": float(prof["se_A"].median()),
        "split_se_A": se_split, "split_r": half_r,
        "profile_sd_A": float(prof["spacing_A"].std()),
        "spacing_min_A": float(prof["spacing_A"].min()),
        "spacing_max_A": float(prof["spacing_A"].max()),
    }
    return cl, prof, summary


def main() -> None:
    """Re-index every refined site and read the spacing in a sliding window."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=pathlib.Path, required=True,
                        help="the PATCH table the every-site steps ran on")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"lattice indices per window (default {DEFAULT_WINDOW})")
    parser.add_argument("--register", choices=("right", "in"), default="right")
    parser.add_argument("--found-only", action="store_true",
                        help="Stage 4's sites only, for comparison")
    parser.add_argument("--truth", type=pathlib.Path, default=None,
                        help="synthetic truth JSON with lattice blocks")
    parser.add_argument("--label", default=None,
                        help="the every-site run's label if not the table's tag")
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = table.stem.removeprefix("output_correlation_table_")
    label = args.label or tag
    results = table.parent
    frame = frame_geometry(table)
    real, scores, refined, cut = load(results, tag, label)
    stem, geometry = template_geometry(refined["in"])
    truth = None
    if args.truth is not None:
        path = args.truth if args.truth.is_absolute() else MT_ROOT / args.truth
        truth = json.loads(path.read_text())
    variant = (f"w{args.window}" + ("_in" if args.register == "in" else "")
               + ("_found" if args.found_only else ""))
    name = f"{tag}_{variant}"
    print(f"{tag}: patch {stem} ({geometry.n_protofilaments} PF, repeat "
          f"{geometry.rise_angstrom:.3f} A), {frame.pixel_size} A/px; window {args.window} "
          f"indices; register {args.register}; "
          f"{'found sites only' if args.found_only else 'every site'}; decoy p99 z {cut:.2f}")

    all_clusters, all_profiles, summaries = [], [], []
    for fil in sorted(real["filament"].unique()):
        part = real[real["filament"] == fil]
        out = profile_filament(int(fil), part, scores, refined, cut, geometry,
                               frame.pixel_size, args, truth)
        if out is None:
            continue
        cl, prof, summary = out
        all_clusters.append(cl)
        all_profiles.append(prof)
        summaries.append(summary)
    clusters_df = pd.concat(all_clusters, ignore_index=True)
    profile = pd.concat(all_profiles, ignore_index=True)
    summary = pd.DataFrame(summaries)
    parents = parse_child_labels(tag)
    for frame_ in (clusters_df, profile, summary):
        frame_.insert(1, "tube", frame_["filament"].map(parents) if parents
                      else frame_["filament"])

    print(f"\n{'fil':>4}{'tube':>5}{'sites':>12}{'clusters':>9}{'span':>6}{'slips':>6}"
          f"{'site sd':>8}{'sigma':>7}{'infl':>6}{'whole A':>9}{'se A':>7}{'split se':>9}"
          f"{'halves r':>9}{'prof sd':>8}{'local min-max A':>18}")
    for _, s in summary.iterrows():
        print(f"{s['filament']:>4}{s['tube']:>5}{s['sites_used']:>6}/{s['sites_total']:<5}"
              f"{s['clusters']:>9}{s['index_span']:>6}{s['label_slips']:>6}"
              f"{s['within_cluster_sd_A']:>8.2f}{s['sigma_A']:>7.2f}{s['inflation']:>6.2f}"
              f"{s['whole_tube_spacing_A']:>9.3f}{s['median_se_A']:>7.3f}"
              f"{s['split_se_A']:>9.3f}{s['split_r']:>9.2f}{s['profile_sd_A']:>8.3f}"
              f"{s['spacing_min_A']:>10.2f}-{s['spacing_max_A']:.2f}")
    print("  site sd: a site's spread about its subunit (robust); sigma: per-index scatter "
          "about the slow trend;\n  slips: places where the predicted labels jump against "
          "the re-indexed clusters; se: median window error (noise model);\n  split se: "
          "sd(even-PF profile - odd-PF profile) / 2; halves r: their correlation along the "
          "tube;\n  prof sd: the profile's own spread along the tube (signal + noise)")

    if truth is not None:
        print(f"\nAGAINST THE TRUTH (through the same {args.window}-index window)")
        print(f"{'tube':>5}{'windows':>9}{'rms err A':>11}{'pull mean':>11}{'pull sd':>9}"
              f"{'pure C':>16}{'pure E':>16}")
        for fil, part in profile.groupby("filament"):
            err = part["spacing_A"] - part["truth_A"]
            pure_c = part[part["truth_C_share"] == 1]["spacing_A"]
            pure_e = part[part["truth_C_share"] == 0]["spacing_A"]
            print(f"{fil:>5}{len(part):>9}{np.sqrt(np.mean(err ** 2)):>11.3f}"
                  f"{part['pull'].mean():>+11.2f}{part['pull'].std():>9.2f}"
                  + (f"{pure_c.mean():>10.3f} ({len(pure_c):>3})" if len(pure_c) else
                     f"{'--':>16}")
                  + (f"{pure_e.mean():>10.3f} ({len(pure_e):>3})" if len(pure_e) else
                     f"{'--':>16}"))
        err = profile["spacing_A"] - profile["truth_A"]
        print(f"  all: rms error {np.sqrt(np.mean(err ** 2)):.3f} A, pull sd "
              f"{profile['pull'].std():.2f} (1 = the error bars are right); truth C "
              f"{C_REPEAT}, E {E_REPEAT}")

    clusters_df.to_csv(results / f"spacing_clusters_{name}.csv", index=False)
    profile.to_csv(results / f"spacing_profile_{name}.csv", index=False)
    summary.to_csv(results / f"spacing_summary_{name}.csv", index=False)
    print(f"\nwrote spacing_{{clusters,profile,summary}}_{name}.csv in "
          f"{results.relative_to(MT_ROOT)}/")


if __name__ == "__main__":
    main()
