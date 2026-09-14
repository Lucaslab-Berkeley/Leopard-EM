"""Simulate the 2x2 lattice-competition templates built by models/build_mt_templates.py.

Identical simulation parameters to process_GMPCPP.ipynb and simulate_patch_centred.py --
0.9194 A, 600^3, b_factor_scaling 0.5, dose 0-50, k3 MTF -- because the whole point of
the 2x2 is that nothing except the model differs between the four searches.

``center_atoms=False`` is deliberate and matches the existing templates: the builder
already puts the tube axis on the box centre, so the templates are axis-centred and the
analysis reads an axis offset near zero.

Usage:  python run_scripts/simulate_lattice_competition.py [--flat-b-only] [--device 0]
"""

import argparse
import pathlib
import time

from ttsim3d.models import Simulator, SimulatorConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MAP_DIR = ROOT / "maps"

PIXEL_SIZE = 0.9194
VOLUME_SHAPE = (600, 600, 600)
B_FACTOR_SCALING = 0.5

# The 2x2 of {atom source} x {lattice}, each with native and flattened B-factors.
TEMPLATES = [
    f"{atoms}_atoms_{lattice}_lattice_2rings{suffix}"
    for suffix in ("_flatB", "")
    for atoms in ("6dpu", "6dpv")
    for lattice in ("6dpu", "6dpv")
]

SIM_CONFIG = SimulatorConfig(
    voltage=300.0,
    apply_dose_weighting=True,
    dose_start=0.0,
    dose_end=50.0,
    dose_filter_modify_signal="rel_diff",
    upsampling=-1,
    mtf_reference="k3_300kV_FL2",
)


def simulate(stem: str, device: str) -> pathlib.Path:
    """Simulate one template and write it next to the existing maps."""
    pdb_path = MODEL_DIR / f"{stem}.pdb"
    mrc_path = MAP_DIR / f"{stem}_{PIXEL_SIZE}_bscale{B_FACTOR_SCALING}.mrc"
    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path} -- run models/build_mt_templates.py first")

    simulator = Simulator(
        pdb_filepath=str(pdb_path),
        pixel_spacing=PIXEL_SIZE,
        volume_shape=VOLUME_SHAPE,
        center_atoms=False,
        remove_hydrogens=True,
        b_factor_scaling=B_FACTOR_SCALING,
        additional_b_factor=0,
        simulator_config=SIM_CONFIG,
    )
    simulator.export_to_mrc(str(mrc_path), device=device)
    return mrc_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-b-only", action="store_true", help="skip native-B variants")
    parser.add_argument(
        "--stems",
        nargs="+",
        metavar="STEM",
        help="simulate these template stems instead of the 2x2, e.g. 6dpu_13pf_2rings_flatB",
    )
    # CPU by default: it is what the existing templates were simulated on, and the
    # GPU path JIT-compiles a reduction kernel that needs an nvrtc this box lacks.
    parser.add_argument("--device", default="cpu", help="'cpu' or e.g. 'cuda:0'")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    wanted = args.stems or [
        t for t in TEMPLATES if not args.flat_b_only or t.endswith("_flatB")
    ]

    print(f"Simulating {len(wanted)} templates on {args.device}")
    for i, stem in enumerate(wanted, start=1):
        mrc_path = MAP_DIR / f"{stem}_{PIXEL_SIZE}_bscale{B_FACTOR_SCALING}.mrc"
        if mrc_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(wanted)}] {stem}: exists, skipping")
            continue
        start = time.monotonic()
        simulate(stem, args.device)
        print(
            f"  [{i}/{len(wanted)}] {stem}: {time.monotonic() - start:6.1f} s"
            f"  -> {mrc_path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
