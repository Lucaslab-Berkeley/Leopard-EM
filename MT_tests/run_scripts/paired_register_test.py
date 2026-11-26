"""The register as a PAIRED competition, calibrated on the synthetic.

Everything that has worked on this specimen was paired per site -- protofilament number
at t = 16.2, lattice at t = 8.69 -- because scoring two templates at the SAME site
cancels ice, defocus, occupancy and local order exactly. The register test was the one
readout not done that way: it compared even-index sites against odd-index ones, which
are different physical sites, so all of that variation was the noise. Measured, the
unpaired per-site noise is 1.95 z against the paired 0.92 -- twice what it needs to be,
on the readout least able to afford it.

So instead: at every site score the patch, and score a patch built ONE MONOMER OUT OF
REGISTER (``--monomer-shift 1``), and take the difference. The two templates are matched
to the atom -- same 13 PF, same 40.835 A rise, same 85.8 A axis offset, same 67,390 atoms
-- and differ only in which monomer is alpha. The per-site difference is then alpha/beta
discrimination with everything else cancelled.

Run on the curved synthetic FIRST, where the register is known: the unpaired test reads
it at 8.5-9.5% of score with t from -7.4 to -11.9. If the paired design does not at least
match that, the design is wrong and the GDP number would mean nothing.

Usage:
    python run_scripts/paired_register_test.py                 # synthetic
    python run_scripts/paired_register_test.py --frame gdp     # the real data
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry
from leopard_em.pydantic_models.managers import RefineTemplateManager
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from continuous_unwrap_curved import continuous_sites  # noqa: E402

from frame_geometry import (  # noqa: E402
    LEGACY_PIXEL_SIZE,
    legacy_frame_geometry,
    legacy_map,
)

SEARCH_STEM = "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB"
# Each entry is (in-register stem, one-monomer-out stem). The two must be matched to the
# atom; only which monomer is alpha may differ.
PAIRS = {
    "patch": ("6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB",
              "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB_shift1"),
    "patch4": ("6dpu_atoms_6dpv_lattice_13pf_patch5_4rings_flatB",
               "6dpu_atoms_6dpv_lattice_13pf_patch5_4rings_flatB_shift1"),
    "patch3": ("6dpu_atoms_6dpv_lattice_13pf_patch3_2rings_flatB",
               "6dpu_atoms_6dpv_lattice_13pf_patch3_2rings_flatB_shift1"),
    "patch3_4": ("6dpu_atoms_6dpv_lattice_13pf_patch3_4rings_flatB",
                 "6dpu_atoms_6dpv_lattice_13pf_patch3_4rings_flatB_shift1"),
    "ring": ("6dpu_atoms_6dpv_lattice_13pf_2rings_flatB",
             "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB_shift1"),
}

FRAMES = {
    "synthetic": {
        "table": "results_curved/output_correlation_table_clean_patch13.h5",
        "results": "results_curved",
        "micrograph": "Frames/synthetic_curved_13pf_6dpv_sag100_clean.mrc",
        "map_tag": "clean_patch13",
        "optics": {"defocus_u": 8390.503906, "defocus_v": 8035.022461,
                   "astigmatism_angle": -0.864719, "pixel_size": LEGACY_PIXEL_SIZE,
                   "voltage": 300.0, "spherical_aberration": 2.7,
                   "amplitude_contrast_ratio": 0.07, "phase_shift": 0.0,
                   "ctf_B_factor": 60.0},
        "known": "the unpaired test reads this frame at 8.5-9.5% of score, t -7.4 to -11.9",
        # The template that MADE the table -- not the pair being scored, which may be
        # the 3-PF patches. Only consulted because this table predates recorded geometry.
        "search_stem": SEARCH_STEM,
    },
    "gdp": {
        "table": "results_gdp_curved/output_correlation_table_gdp_curved_13pf_patch.h5",
        "results": "results_gdp_curved",
        "micrograph": "Frames/GDP_curved_mgraph.mrc",
        "map_tag": "gdp_curved_13pf_patch",
        "optics": None,   # read from the PICASSO YAML
        "known": "the unpaired test gave 3.0% at t = +2.35, vanishing under any cut",
        "search_stem": SEARCH_STEM,
    },
}


def build_stack(clean, optics, micrograph, template, results, map_tag, frame):
    """Particle stack at the sites, for whichever template is being scored.

    ``frame`` is the ``frame_geometry`` of the table the sites came from.
    """
    stack = pd.DataFrame({
        "particle_index": np.arange(len(clean)),
        "mip": clean["correlation_value"].to_numpy(),
        "scaled_mip": clean["z_score"].to_numpy(),
        "correlation_mean": clean["correlation_mean"].to_numpy(),
        "correlation_variance": clean["correlation_variance"].to_numpy(),
        "total_correlations": frame.correlations_per_pixel,
        "pos_x": clean["x"].astype(int).to_numpy(),
        "pos_y": clean["y"].astype(int).to_numpy(),
        "phi": clean["phi"].to_numpy(), "theta": clean["theta"].to_numpy(),
        "psi": clean["psi"].to_numpy(),
        "relative_defocus": clean["relative_defocus"].to_numpy(),
    })
    stack["pos_x_img"] = stack["pos_x"] + frame.half_width
    stack["pos_y_img"] = stack["pos_y"] + frame.half_width
    stack["pos_x_img_angstrom"] = stack["pos_x_img"] * frame.pixel_size
    stack["pos_y_img_angstrom"] = stack["pos_y_img"] * frame.pixel_size
    for key, value in optics.items():
        stack[key] = value
    stack["refined_pixel_size"] = optics["pixel_size"]
    for key in ("even_zernikes", "odd_zernikes", "mag_matrix"):
        stack[key] = np.nan
    stack["micrograph_path"] = micrograph
    stack["template_path"] = template
    for key, name in (
        ("mip_path", "output_mip"), ("scaled_mip_path", "output_scaled_mip"),
        ("psi_path", "output_orientation_psi"),
        ("theta_path", "output_orientation_theta"),
        ("phi_path", "output_orientation_phi"),
        ("defocus_path", "output_relative_defocus"),
        ("correlation_average_path", "output_correlation_average"),
        ("correlation_variance_path", "output_correlation_variance"),
    ):
        stack[key] = f"{results}/{name}_{map_tag}.mrc"
    return stack


def score(stem, stack, tag, gpus, batch):
    """Refine one template over the stack; returns refined z per site."""
    stack_path = MT_ROOT / "results_curved" / f"paired_stack_{tag}.csv"
    stack = stack.copy()
    stack["template_path"] = f"maps/{stem}_0.9194_bscale0.5.mrc"
    stack.to_csv(stack_path, index=True)
    config = {
        "template_volume_path": f"maps/{stem}_0.9194_bscale0.5.mrc",
        "particle_stack": {"df_path": str(stack_path.relative_to(MT_ROOT)),
                           "extracted_box_size": [640, 640],
                           "original_template_size": [600, 600]},
        "defocus_refinement_config": {"enabled": True, "defocus_min": -100.0,
                                      "defocus_max": 100.0, "defocus_step": 25.0},
        "pixel_size_refinement_config": {"enabled": False, "pixel_size_min": -0.01,
                                         "pixel_size_max": 0.01, "pixel_size_step": 0.01},
        "orientation_refinement_config": {
            "enabled": True, "phi_step_coarse": 2.5, "phi_step_fine": 0.4,
            "theta_step_coarse": 3.5, "theta_step_fine": 0.4,
            "psi_step_coarse": 2.5, "psi_step_fine": 0.4,
            "base_grid_method": "uniform"},
        "preprocessing_filters": {
            "whitening_filter": {"enabled": True, "do_power_spectrum": True,
                                 "max_freq": 1.0},
            "bandpass_filter": {"enabled": False}},
        "computational_config": {"gpu_ids": gpus, "num_cpus": 8},
        "movie_config": {"enabled": False, "movie_path": "",
                         "deformation_field_path": "", "pre_exposure": 0.0,
                         "fluence_per_frame": 1.0},
    }
    config_path = MT_ROOT / "configs" / f"paired_{tag}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    out = MT_ROOT / "results_curved" / f"paired_refined_{tag}.csv"
    manager = RefineTemplateManager.from_yaml(str(config_path))
    start = time.time()
    manager.run_refine_template(str(out), batch, allow_file_overwrite=True)
    print(f"    {tag}: {len(stack)} sites in {time.time() - start:.0f}s")
    return pd.read_csv(out, index_col=0)["refined_scaled_mip"].to_numpy()


def main() -> None:
    """Score both templates at the same sites and report the paired difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default="synthetic", choices=sorted(FRAMES))
    parser.add_argument("--pair", default="patch", choices=sorted(PAIRS),
                        help="which matched template pair to compete")
    parser.add_argument("--min-z", type=float, default=7.0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    spec = FRAMES[args.frame]
    optics = spec["optics"]
    if optics is None:
        with open(MT_ROOT / "results_ctf" / "GDP_curved_mgraph_optics.yaml",
                  encoding="utf-8") as handle:
            optics = yaml.safe_load(handle)["optics_group"]
        optics.pop("label", None)
        optics["ctf_B_factor"] = 60.0

    in_stem, out_stem = PAIRS[args.pair]
    geometry = TemplateLatticeGeometry.from_pdb(
        str(MT_ROOT / "models" / f"{in_stem}.pdb"))
    # A ring's phi is N-fold degenerate so it cannot identify a protofilament; resolving
    # them would be meaningless for it.
    resolve = args.pair != "ring"
    table = MT_ROOT / spec["table"]
    detections = detections_from_hdf5(
        str(table), min_z_score=args.min_z).reset_index(drop=True)
    frame = legacy_frame_geometry(table, spec["search_stem"])
    print(frame.describe())
    boot = A.pick_bootstrap(detections["z_score"].to_numpy(), want=20)
    sites = continuous_sites(detections, geometry, boot, pixel_size=frame.pixel_size,
                             resolve_protofilaments=resolve)
    clean = detections.iloc[sites["detection_index"]].reset_index(drop=True)
    print(f"{args.frame} / {args.pair}: {len(clean)} sites "
          f"({'protofilament-resolved' if resolve else 'one per axial index'}), "
          f"rise {sites['rise']:.3f} Å")
    print(f"  for reference, {spec['known']}\n")

    stack = build_stack(clean, optics, spec["micrograph"],
                        str(legacy_map(in_stem).relative_to(MT_ROOT)),
                        spec["results"], spec["map_tag"], frame)
    in_reg = score(in_stem, stack, f"{args.frame}_{args.pair}_in",
                   args.gpus, args.batch_size)
    out_reg = score(out_stem, stack, f"{args.frame}_{args.pair}_out",
                    args.gpus, args.batch_size)

    # SIGN BY PARITY. The two templates each win at alternate sites -- the in-register
    # one at even axial indices, the shifted one at odd -- so the raw difference averages
    # to ZERO by construction. It has to be signed by which template should win where.
    # (Averaging it unsigned was the first version of this test and it reported +0.362
    # from nothing but the even/odd count imbalance.)
    index = sites["axial_index"].astype(int)
    raw = in_reg - out_reg
    sign = np.where(index % 2 == 0, 1.0, -1.0)
    signed = raw * sign
    # WHICH parity is alpha is arbitrary -- it is set by the lattice phase origin -- so a
    # global flip is meaningless and the magnitude is what counts. Report the agreement
    # fraction, which is above 50% for either convention when discrimination is real.
    agreement = float(np.mean(np.sign(raw) == sign))
    if agreement < 0.5:
        signed, agreement = -signed, 1.0 - agreement

    n = len(signed)
    sem = signed.std(ddof=1) / np.sqrt(n)
    print(f"\n  in register   mean z {in_reg.mean():.3f}")
    print(f"  one monomer out      {out_reg.mean():.3f}")
    print(f"  sites {n}   even {int((index % 2 == 0).sum())}  "
          f"odd {int((index % 2 == 1).sum())}")
    print(f"\n  PAIRED delta, signed by parity  {signed.mean():+.3f} +/- {sem:.3f}"
          f"   t = {signed.mean() / sem:+.2f}")
    print(f"    = {100 * abs(signed.mean()) / in_reg.mean():.1f}% of score")
    print(f"    the correct template wins at {agreement:.1%} of sites "
          f"(50% = no discrimination)")
    print(f"    per-site sd of the paired difference {signed.std(ddof=1):.3f} z")
    print(f"\n  unsigned mean would be {raw.mean():+.3f} -- meaningless, it cancels")

    pd.DataFrame({"axial_index": index, "in_register": in_reg,
                  "out_of_register": out_reg, "signed_delta": signed}).to_csv(
        MT_ROOT / spec["results"] / f"paired_register_{args.frame}_{args.pair}.csv", index=True)


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
