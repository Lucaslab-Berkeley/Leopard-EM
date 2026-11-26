"""The micrograph's pixel size from Leopard-EM's optimize_template, on a site table.

``OptimizeTemplateManager`` re-simulates the template from its PDB at a range of pixel
spacings and scores every site at its fixed pose (no position, angle or defocus search
when ``--slack 0``); the spacing whose template scores best is the micrograph's pixel
size, since a template sampled at spacing p shows the structure at the scale an image of
pixel p would. The CTF stays at the image's own calibration.

Two passes, same machinery:

1. Leopard's own search (``run_optimize_template``: coarse grid, stop after two
   decreases, fine grid round the best), as the program would be run;
2. a full regular grid with every site's z kept, so the curve is seen whole and the
   optimum gets a bootstrap error over sites (a parabola through the top of the curve,
   refitted on resampled sites).

Which sites matter. Peaks the search found were chosen because noise matched the
template at the search's pixel size, which pulls the optimum towards it; the Stage 6b
neutral poses (lattice prediction + neighbours, nothing searched) carry no such pull.

The template is simulated exactly as ``simulate_lattice_competition.py`` makes the maps
(the stored map is reproduced to cc 1.000000). ``--add-pdb PATH WEIGHT`` (repeatable)
adds further models, each simulated at the same pixel spacing and summed with its weight:
a template no single PDB can express, such as kinesin at half weight on every monomer
(the motors and the register-shifted motors at 0.5 each), which needs neither register
nor seam. Pass ``--neighborhood-b`` with it so the parts are cut off alike and add up.

Usage:
    python run_scripts/optimize_pixel_size_leopard.py --label kinesin_2_0031_neutral \\
        --stack results_kinesin_2_0031/stack_sites_kinesin_2_0031_14pf_neutral.csv \\
        --outdir results_kinesin_2_0031
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from simulate_lattice_competition import fix_neighborhood, sim_config  # noqa: E402


def build_manager(args):
    """An OptimizeTemplateManager for one site table and one (or a weighted sum of) PDB."""
    # pylint: disable=import-outside-toplevel
    import torch
    from leopard_em.pydantic_models.managers import OptimizeTemplateManager
    from leopard_em.utils.backend_setup import setup_particle_backend_kwargs
    from ttsim3d.models import Simulator

    if args.neighborhood_b is not None:
        fix_neighborhood(args.neighborhood_b)

    def make_simulator(pdb):
        return Simulator(
            pdb_filepath=str(pdb), pixel_spacing=args.pixel_size,
            volume_shape=(args.box,) * 3, center_atoms=False, remove_hydrogens=True,
            b_factor_scaling=0.5, additional_b_factor=0,
            simulator_config=sim_config(args.voltage, args.mtf, args.exposure, True))

    parts = [(make_simulator(pdb), float(weight)) for pdb, weight in (args.add_pdb or [])]

    class CompositeManager(OptimizeTemplateManager):
        """The optimizer with the template = base model + weighted further models."""

        def make_backend_core_function_kwargs(self, prefer_refined_angles: bool = True):
            device = self.computational_config.gpu_ids
            template = self.simulator.run(device=device)
            for simulator, weight in parts:
                simulator.pixel_spacing = self.simulator.pixel_spacing
                template = template + weight * simulator.run(device=device)
            # the rest as OptimizeTemplateManager: poses fixed, nothing searched
            return setup_particle_backend_kwargs(
                particle_stack=self.particle_stack, template=template,
                preprocessing_filters=self.preprocessing_filters,
                euler_angles=self.particle_stack.get_euler_angles(prefer_refined_angles),
                euler_angle_offsets=torch.zeros((1, 3)),
                defocus_offsets=torch.tensor([0.0]),
                pixel_size_offsets=torch.tensor([0.0]),
                apply_global_filtering=self.apply_global_filtering,
                device_list=self.computational_config.gpu_devices)

    simulator = make_simulator(args.pdb)
    box = args.box + 2 * args.slack
    manager_class = CompositeManager if parts else OptimizeTemplateManager
    return manager_class(
        particle_stack={"df_path": str(args.stack), "extracted_box_size": [box, box],
                        "original_template_size": [args.box, args.box]},
        pixel_size_coarse_search={"enabled": True, "pixel_size_min": -args.coarse_range,
                                  "pixel_size_max": args.coarse_range,
                                  "pixel_size_step": args.coarse_step},
        pixel_size_fine_search={"enabled": True, "pixel_size_min": -args.fine_range,
                                "pixel_size_max": args.fine_range,
                                "pixel_size_step": args.fine_step},
        preprocessing_filters={"whitening_filter": {"enabled": True,
                                                    "do_power_spectrum": True,
                                                    "max_freq": 1.0},
                               "bandpass_filter": {"enabled": False}},
        computational_config={"gpu_ids": [args.gpu], "num_cpus": 8},
        simulator=simulator)


def site_scores(manager, px):
    """Every site's z with the template simulated at ``px``."""
    manager.simulator.pixel_spacing = float(px)
    result = manager.get_correlation_result(manager.make_backend_core_function_kwargs(), 1)
    return np.asarray(result["refined_z_score"], dtype=float)


def parabola_peak(grid, curve, half_width):
    """Vertex of a parabola through the curve's top (+/- ``half_width`` points)."""
    top = int(np.nanargmax(curve))
    lo, hi = max(0, top - half_width), min(len(grid), top + half_width + 1)
    if hi - lo < 3:
        return float(grid[top])
    a, b, _ = np.polyfit(grid[lo:hi], curve[lo:hi], 2)
    if a >= 0:
        return float(grid[top])
    return float(np.clip(-b / (2 * a), grid[lo], grid[hi - 1]))


def main() -> None:
    """Leopard's optimizer, then the full curve with a bootstrap error."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", type=pathlib.Path, required=True,
                        help="site table in Leopard's results format")
    parser.add_argument("--label", required=True)
    parser.add_argument("--pdb", type=pathlib.Path,
                        default=MT_ROOT / "models" / "6dpu_14pf_2rings_flatB.pdb")
    parser.add_argument("--add-pdb", nargs=2, action="append", metavar=("PDB", "WEIGHT"),
                        help="add this model, times WEIGHT, to the template (repeatable)")
    parser.add_argument("--neighborhood-b", type=float, default=None,
                        help="fix ttsim3d's atom neighbourhood at this raw B (as the "
                             "_nb maps); needed for summed models to add up")
    parser.add_argument("--pixel-size", type=float, default=1.13,
                        help="the micrograph's nominal A/px (the search's template)")
    parser.add_argument("--box", type=int, default=488)
    parser.add_argument("--slack", type=int, default=0,
                        help="px of position search each side (0: score the pose as given)")
    parser.add_argument("--voltage", type=float, default=200.0)
    parser.add_argument("--mtf", default="k3_200kV_CDS")
    parser.add_argument("--exposure", type=float, default=50.0)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--coarse-range", type=float, default=0.05)
    parser.add_argument("--coarse-step", type=float, default=0.01)
    parser.add_argument("--fine-range", type=float, default=0.005)
    parser.add_argument("--fine-step", type=float, default=0.001)
    parser.add_argument("--grid-step", type=float, default=0.005,
                        help="step of the full curve (pass 2)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / f"optimize_px_{args.label}"
    manager = build_manager(args)
    n_sites = len(pd.read_csv(args.stack))
    extra = "".join(f" + {w} x {pathlib.Path(p).name}" for p, w in (args.add_pdb or []))
    print(f"{args.label}: {n_sites} sites from {args.stack.name}, template {args.pdb.name}"
          f"{extra}, nominal {args.pixel_size} A/px, slack {args.slack} px")

    # 1. Leopard's own search, as the program runs it (every site counted: the default
    #    keeps only z >= 8, which these sites barely reach)
    manager.run_optimize_template(output_text_path=f"{stem}_leopard.txt",
                                  min_snr=-np.inf)
    leopard_px = float(pathlib.Path(f"{stem}_leopard.txt").read_text().split(":")[1]
                       .split()[0])

    # 2. the whole curve, per site
    grid = np.round(np.arange(args.pixel_size - args.coarse_range,
                              args.pixel_size + args.coarse_range + args.grid_step / 2,
                              args.grid_step), 4)
    scores = np.stack([site_scores(manager, px) for px in grid], axis=1)  # (site, px)
    keep = np.isfinite(scores).all(axis=1)
    scores = scores[keep]
    curve = scores.mean(axis=0)
    best = parabola_peak(grid, curve, 3)
    rng = np.random.default_rng(0)
    draws = np.array([parabola_peak(grid, scores[rng.integers(0, len(scores),
                                                              len(scores))].mean(axis=0), 3)
                      for _ in range(args.bootstrap)])
    low, high = np.percentile(draws, [2.5, 97.5])
    at_nominal = scores[:, np.argmin(np.abs(grid - args.pixel_size))]
    at_best = scores[:, np.argmin(np.abs(grid - best))]
    gain = at_best - at_nominal
    t_gain = gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain)))

    pd.DataFrame(scores, columns=[f"{px:.4f}" for px in grid]).to_csv(f"{stem}_sites.csv",
                                                                     index=False)
    pd.DataFrame({"pixel_size": grid, "mean_z": curve,
                  "sem": scores.std(axis=0, ddof=1) / np.sqrt(len(scores))}).to_csv(
        f"{stem}_curve.csv", index=False)
    print(f"\nLeopard optimize_template: {leopard_px:.3f} A/px")
    print(f"full curve ({len(scores)} sites): peak {best:.4f} A/px, 95% bootstrap "
          f"{low:.4f}-{high:.4f} ({100 * (best / args.pixel_size - 1):+.2f}% vs nominal)")
    print(f"z at the peak vs at nominal: {gain.mean():+.3f} per site (paired t {t_gain:.1f})")
    plot(grid, scores, best, (low, high), leopard_px, args, f"{stem}.png")
    print(f"wrote {stem}_curve.csv, {stem}_sites.csv, {stem}.png, {stem}_leopard*.txt/csv")


def plot(grid, scores, best, interval, leopard_px, args, out_path):
    """Mean z against template pixel size, with the optimum and its interval."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    curve = scores.mean(axis=0)
    sem = scores.std(axis=0, ddof=1) / np.sqrt(len(scores))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.fill_between(grid, curve - sem, curve + sem, color="#1f4e79", alpha=0.2, lw=0)
    ax.plot(grid, curve, "o-", color="#1f4e79", ms=3, lw=1.2, label="mean z over sites")
    ax.axvspan(*interval, color="#c62828", alpha=0.12, lw=0, label="95% bootstrap")
    ax.axvline(best, color="#c62828", lw=1.2, label=f"peak {best:.4f}")
    ax.axvline(args.pixel_size, color="grey", ls="--", lw=1, label=f"nominal {args.pixel_size}")
    ax.axvline(leopard_px, color="black", ls=":", lw=1, label=f"Leopard {leopard_px:.3f}")
    ax.set_xlabel("template pixel spacing (A)")
    ax.set_ylabel("mean z")
    ax.set_title(f"{args.label}: {len(scores)} sites, {args.pdb.stem}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    main()
