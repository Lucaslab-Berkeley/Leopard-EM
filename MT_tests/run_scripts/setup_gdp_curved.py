"""Write every search config for the curved GDP micrograph.

GDP means the COMPACTED lattice, so the protofilament competition is run at 6DPV
throughout -- competing a 13-PF expanded model against a 14-PF compacted one would
confound the two questions. The expanded pair is included separately, at whichever N
wins, so lattice spacing can be asked on its own.

Two microtubules were drawn on this frame, and the constraint holds both. A search
covers them together; the analysis separates them by region.

Usage:
    python run_scripts/setup_gdp_curved.py
    python run_scripts/setup_gdp_curved.py --micrograph Frames/other.mrc --prefix other
"""

import argparse
import pathlib

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Ordered by what to run first: protofilament number decides which lattice pair and
# which patch are worth running at all.
SEARCHES = [
    # tag, template stem, what it is for
    ("12pf", "6dpu_atoms_6dpv_lattice_12pf_2rings_flatB", "PF competition"),
    ("13pf", "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB", "PF competition"),
    ("14pf", "6dpu_atoms_6dpv_lattice_14pf_2rings_flatB", "PF competition"),
    ("15pf", "6dpu_atoms_6dpv_lattice_15pf_2rings_flatB", "PF competition"),
    ("13pf_expanded", "6dpu_13pf_2rings_flatB", "lattice spacing, if N = 13"),
    ("14pf_expanded", "6dpu_atoms_6dpu_lattice_2rings_flatB",
     "lattice spacing, if N = 14"),
    ("13pf_patch", "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB",
     "register / seam, if N = 13"),
    ("14pf_patch", "6dpu_atoms_6dpv_lattice_14pf_patch5_2rings_flatB",
     "register / seam, if N = 14"),
]

CONFIG = """micrograph_path: {micrograph}
template_volume_path: maps/{stem}_0.9194_bscale0.5.mrc
computational_config:
  gpu_ids:
  - 0
  - 1
  - 2
  - 3
  num_cpus: 8
defocus_search_config:
  enabled: true
  defocus_max: 600.0
  defocus_min: -600.0
  defocus_step: 200.0
match_template_result:
  allow_file_overwrite: true
  correlation_average_path: {out}/output_correlation_average_{tag}.mrc
  correlation_variance_path: {out}/output_correlation_variance_{tag}.mrc
  mip_path: {out}/output_mip_{tag}.mrc
  orientation_phi_path: {out}/output_orientation_phi_{tag}.mrc
  orientation_psi_path: {out}/output_orientation_psi_{tag}.mrc
  orientation_theta_path: {out}/output_orientation_theta_{tag}.mrc
  relative_defocus_path: {out}/output_relative_defocus_{tag}.mrc
  scaled_mip_path: {out}/output_scaled_mip_{tag}.mrc
optics_group:
  label: micrograph_1
  amplitude_contrast_ratio: 0.07
  ctf_B_factor: 60.0
  astigmatism_angle: -0.864719
  defocus_u: 8390.503906
  defocus_v: 8035.022461
  phase_shift: 0.0
  pixel_size: 0.9194
  spherical_aberration: 2.7
  voltage: 300.0
orientation_search_config:
  base_grid_method: uniform
  psi_step: 2.5      # in degrees
  theta_step: 3.5  # in degrees
preprocessing_filters:
  bandpass_filter:
    enabled: false
    falloff: 0.05
    high_freq_cutoff: 0.5
    low_freq_cutoff: 0.0
  whitening_filter:
    enabled: true
    do_power_spectrum: true
    max_freq: 1.0
    num_freq_bins: null
"""


def main() -> None:
    """Write one config per template, and report what is missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micrograph", default="Frames/GDP_curved_mgraph.mrc")
    parser.add_argument("--prefix", default="gdp_curved")
    parser.add_argument("--results", default="results_gdp_curved")
    args = parser.parse_args()

    (MT_ROOT / args.results).mkdir(exist_ok=True)
    missing = []
    print(f"micrograph {args.micrograph}\nresults -> {args.results}/\n")
    for tag, stem, purpose in SEARCHES:
        volume = MT_ROOT / "maps" / f"{stem}_0.9194_bscale0.5.mrc"
        full_tag = f"{args.prefix}_{tag}"
        path = MT_ROOT / "configs" / f"match_tm_{full_tag}.yaml"
        path.write_text(
            CONFIG.format(micrograph=args.micrograph, stem=stem,
                          tag=full_tag, out=args.results)
        )
        mark = " " if volume.exists() else " MAP MISSING"
        if not volume.exists():
            missing.append(stem)
        print(f"  {full_tag:<28} {purpose:<28}{mark}")

    if missing:
        print("\nSimulate the missing maps first:")
        print("  python run_scripts/simulate_lattice_competition.py --stems "
              + " ".join(missing))

    # The tag already carries the prefix, so the runner must not add its own.
    print(f"\nRun one with:\n  python run_scripts/run_match_curved.py --prefix '' "
          f"\\\n      --tag {args.prefix}_13pf \\\n"
          f"      --constraint configs/filament_constraint_{args.prefix}.yaml \\\n"
          f"      --results {args.results}")


if __name__ == "__main__":
    main()
