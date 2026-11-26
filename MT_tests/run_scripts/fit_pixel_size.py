"""Fit the pixel size from the data: both lattices, scored over a grid of pixel sizes.

A nominal pixel size that is off rescales every spacing in the image, and an isotropic
scale change of the AXIAL repeat alone cannot be told from compaction -- 1% is 0.41 A,
nearly the 0.445 A that separates each lattice from the midpoint. But the same scale
change also stretches the tube's RADIUS and protofilament spacing, which compaction
barely changes. So the right (pixel size, lattice) pair should match best of all: the
radial features fix the scale, the axial ones the lattice.

The test: render the compacted and expanded rings at a grid of pixel sizes (same box in
pixels, so only the object's size changes), and score every site at its Stage 6b
NEUTRAL pose with each -- one pose per site, nothing searched, the same pose for every
map. The refine config keeps the search's optics (the nominal pixel size), so the CTF
is applied in the image's own pixel units: only the object is rescaled. (Refine's own
pixel-size refinement would also recompute the CTF at the trial pixel size, shifting its
rings against a defocus that was fitted at the nominal one, and pull the answer back to
nominal.)

Per tube, each lattice's mean z against pixel size has a peak; the lattice with the
higher peak is the tube's, and its peak position is the pixel size. The tubes share one
pixel size, so a joint fit takes, at each trial pixel size, each tube's better lattice.

Needs Stage 6b's neutral stack (``stack_sites_<tag>_neutral.csv``).

Usage:
    python run_scripts/fit_pixel_size.py \\
        --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5 \\
        --steps -3 -2 -1 0 1 2 3 4 5
"""

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compete_sites_curved import write_config  # noqa: E402
from frame_geometry import frame_geometry, map_for, template_half_width  # noqa: E402

RINGS = {"C": "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB",
         "E": "6dpu_13pf_2rings_flatB"}
# The 5-PF patch sees five protofilaments' spacing and local curvature, not the whole
# tube's diameter: if the two disagree on the scale, the ring's fit is reading the
# tube's cross-section, not the pixel size.
PATCHES = {"C": "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB",
           "E": "6dpu_13pf_patch5_2rings_flatB"}
BLOCK = 8   # lattice indices per bootstrap block (neighbouring sites are correlated)


def grid(nominal, steps):
    """Trial pixel sizes: nominal x (1 + step/100), the nominal itself exact."""
    return [nominal if step == 0 else round(nominal * (1 + step / 100.0), 4)
            for step in steps]


def simulate_missing(lattices, pixel_sizes, box, jobs):
    """Render every missing (lattice, pixel size) map, ``jobs`` at a time, on CPU."""
    missing = [(stem, p) for p in pixel_sizes for stem in lattices.values()
               if not map_for(stem, p).is_file()]
    if not missing:
        return
    print(f"simulating {len(missing)} map(s), {jobs} at a time")
    running = []
    for stem, p in missing:
        while len(running) >= jobs:
            running = [r for r in running if r.poll() is None]
            time.sleep(2)
        log = open(MT_ROOT / "logs" / f"simulate_{stem}_{p}.log", "w", encoding="utf-8")
        running.append(subprocess.Popen(
            [sys.executable, str(MT_ROOT / "run_scripts" / "simulate_lattice_competition.py"),
             "--pixel-size", str(p), "--box", str(box), "--stems", stem],
            cwd=MT_ROOT, stdout=log, stderr=log))
    for r in running:
        r.wait()
    still = [(s, p) for s, p in missing if not map_for(s, p).is_file()]
    if still:
        raise SystemExit(f"simulation failed for {still}; see logs/simulate_*.log")


def score(stack, stem, pixel_size, tag, results, gpus, batch_size, rescore):
    """Refined z of every site at its neutral pose with one map; cached."""
    out_dir = results / f"pixel_fit_{tag}"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{stem}_{pixel_size}.csv"
    if out.is_file() and not rescore:
        return pd.read_csv(out, index_col=0)["refined_scaled_mip"].to_numpy()
    template = str(map_for(stem, pixel_size).relative_to(MT_ROOT))
    base = stack.drop(columns=[c for c in stack.columns if c.startswith("refined_pos_")])
    base = base.reset_index(drop=True)
    base["particle_index"] = np.arange(len(base))
    base["template_path"] = template
    stack_path = out_dir / f"stack_{stem}_{pixel_size}.csv"
    base.to_csv(stack_path, index=True)
    config_path = out_dir / f"refine_{stem}_{pixel_size}.yaml"
    write_config(stack_path, config_path, template,
                 MT_ROOT / "configs" / f"match_tm_{tag}.yaml", gpus, neutral_pose=True)
    # pylint: disable=import-outside-toplevel
    from leopard_em.pydantic_models.managers import RefineTemplateManager
    RefineTemplateManager.from_yaml(str(config_path)).run_refine_template(
        str(out), batch_size)
    return pd.read_csv(out, index_col=0)["refined_scaled_mip"].to_numpy()


def peak(pixel_sizes, curve):
    """Sub-grid peak by a parabola through the best grid point and its neighbours."""
    i = int(np.argmax(curve))
    if i == 0 or i == len(curve) - 1:
        return float(pixel_sizes[i]), float(curve[i]), True   # at the edge: not bracketed
    x = np.asarray(pixel_sizes[i - 1:i + 2])
    a, b, c = np.polyfit(x, curve[i - 1:i + 2], 2)
    if a >= 0:
        return float(pixel_sizes[i]), float(curve[i]), False
    top = -b / (2 * a)
    return float(top), float(np.polyval((a, b, c), top)), False


def block_bootstrap(axial, n_boot, rng):
    """Resampled site indices, in blocks of BLOCK consecutive lattice indices."""
    blocks = axial // BLOCK
    unique = np.unique(blocks)
    members = [np.flatnonzero(blocks == u) for u in unique]
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), len(unique))
        yield np.concatenate([members[j] for j in pick])


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Score the grid, then fit the pixel size per tube and jointly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=pathlib.Path, required=True,
                        help="the table Stage 6b was run on (ring, or with --patch the "
                             "patch table)")
    parser.add_argument("--patch", action="store_true",
                        help="the compacted/expanded PATCHES at the patch sites' "
                             "neutral poses (Stage 6b --patch)")
    parser.add_argument("--steps", type=float, nargs="+",
                        default=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                        help="trial pixel sizes, percent from the nominal")
    parser.add_argument("--truth-pixel-size", type=float, default=None,
                        help="a synthetic's true pixel size, for the report")
    parser.add_argument("--jobs", type=int, default=6, help="simulations at a time")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = table.stem.removeprefix("output_correlation_table_")
    results = table.parent
    frame = frame_geometry(table)
    nominal = frame.pixel_size
    lattices = PATCHES if args.patch else RINGS
    box = 2 * template_half_width(map_for(lattices["C"], nominal))
    pixel_sizes = grid(nominal, args.steps)
    print(frame.describe())
    print(f"{'patches' if args.patch else 'rings'}: {lattices['C']} / {lattices['E']}")
    print(f"nominal {nominal} A/px, box {box}; trial pixel sizes "
          + ", ".join(f"{p:g}" for p in pixel_sizes))

    stack = pd.read_csv(results / f"stack_sites_{tag}_neutral.csv", index_col=0)
    stack = stack[stack["detected"].astype(bool)] if "detected" in stack else stack
    filament = stack["filament"].to_numpy()
    axial = stack["axial_index"].to_numpy()
    print(f"{len(stack)} sites at their Stage 6b neutral poses, "
          f"{len(np.unique(filament))} filament(s)")

    simulate_missing(lattices, pixel_sizes, box, args.jobs)
    z = {}
    for key, stem in lattices.items():
        for p in pixel_sizes:
            start = time.time()
            z[key, p] = score(stack, stem, p, tag, results, args.gpus, args.batch_size,
                              args.rescore)
            print(f"  {key} @ {p:<7g} mean z {z[key, p].mean():6.3f}  "
                  f"({time.time() - start:4.0f}s)")
    rows = pd.DataFrame({f"{k}_{p}": v for (k, p), v in z.items()})
    rows["filament"], rows["axial_index"] = filament, axial
    rows.to_csv(results / f"pixel_fit_{tag}.csv", index=False)

    rng = np.random.default_rng(0)
    p_arr = np.asarray(pixel_sizes)
    print(f"\n{'tube':<6} {'sites':>6} | {'C peak px':>10} {'C z':>7} | {'E peak px':>10} "
          f"{'E z':>7} | {'lattice':>8} {'pixel size':>18} {'C-E at peak':>12}")
    per_tube = {}
    for label in sorted(np.unique(filament)):
        mask = filament == label
        curves = {k: np.array([z[k, p][mask].mean() for p in pixel_sizes])
                  for k in lattices}
        fits = {k: peak(p_arr, curves[k]) for k in lattices}
        best = max(fits, key=lambda k: fits[k][1])
        boot_p, boot_gap = [], []
        for pick in block_bootstrap(axial[mask], args.bootstrap, rng):
            sub = {k: np.array([z[k, p][mask][pick].mean() for p in pixel_sizes])
                   for k in lattices}
            f = {k: peak(p_arr, sub[k]) for k in lattices}
            boot_p.append(f[best][0])
            boot_gap.append(f["C"][1] - f["E"][1])
        gap = fits["C"][1] - fits["E"][1]
        edge = " (EDGE)" if fits[best][2] else ""
        per_tube[label] = (curves, fits, best)
        print(f"{label:<6} {mask.sum():>6} | {fits['C'][0]:>10.4f} {fits['C'][1]:>7.3f} | "
              f"{fits['E'][0]:>10.4f} {fits['E'][1]:>7.3f} | {best:>8} "
              f"{fits[best][0]:>9.4f} ± {np.std(boot_p):.4f}{edge} "
              f"{gap:>+7.3f} ± {np.std(boot_gap):.3f}")

    # One pixel size for the micrograph: at each trial size, each tube's better lattice.
    joint = np.zeros(len(pixel_sizes))
    for label, (curves, _, _) in per_tube.items():
        weight = (filament == label).sum()
        joint += weight * np.maximum(curves["C"], curves["E"])
    joint /= len(filament)
    top, height, at_edge = peak(p_arr, joint)
    print(f"\nJOINT (each tube its better lattice at each size): pixel size {top:.4f} A/px"
          f"{' (at the grid EDGE: extend --steps)' if at_edge else ''}, "
          f"{100 * (top / nominal - 1):+.2f}% from nominal {nominal}")
    if args.truth_pixel_size:
        print(f"  truth {args.truth_pixel_size}: off by "
              f"{100 * (top / args.truth_pixel_size - 1):+.2f}%")
    for label, (curves, _, _) in per_tube.items():
        i = int(np.argmin(np.abs(p_arr - top)))
        print(f"  tube {label} at {p_arr[i]:g}: C {curves['C'][i]:.3f}, E {curves['E'][i]:.3f}"
              f" -> {'C' if curves['C'][i] > curves['E'][i] else 'E'}")
    plot(per_tube, p_arr, nominal, top, results / f"pixel_fit_{tag}.png",
         args.truth_pixel_size)
    print(f"\nwrote {(results / f'pixel_fit_{tag}.csv').relative_to(MT_ROOT)} and the .png")


def plot(per_tube, pixel_sizes, nominal, joint, out_path, truth):
    """Mean z against trial pixel size, both lattices, one panel per tube."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    labels = sorted(per_tube)
    fig, axes = plt.subplots(1, len(labels), figsize=(3.3 * len(labels), 3.2),
                             squeeze=False)
    for ax, label in zip(axes[0], labels):
        curves, fits, best = per_tube[label]
        for key, colour in (("C", "#1f4e79"), ("E", "#c62828")):
            ax.plot(pixel_sizes, curves[key], "o-", color=colour, ms=3.5,
                    label=f"{'compacted' if key == 'C' else 'expanded'}")
            ax.axvline(fits[key][0], color=colour, lw=0.8, ls=":")
        ax.axvline(nominal, color="black", lw=0.8, ls="--")
        ax.axvline(joint, color="#2e7d32", lw=1.2)
        if truth:
            ax.axvline(truth, color="#ef6c00", lw=1.0, ls="-.")
        ax.set_title(f"tube {label}: {best}", fontsize=10)
        ax.set_xlabel("template pixel size (Å)")
        ax.grid(alpha=0.25, ls=":")
    axes[0, 0].set_ylabel("mean z at the neutral poses")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("pixel size: dashed nominal, green joint fit"
                 + (", orange truth" if truth else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
