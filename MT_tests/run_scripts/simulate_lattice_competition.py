"""Simulate the 2x2 lattice-competition templates built by models/build_mt_templates.py.

Identical simulation parameters to process_GMPCPP.ipynb and simulate_patch_centred.py --
0.9194 A, 600^3, b_factor_scaling 0.5, dose 0-50, k3 MTF -- because the whole point of
the 2x2 is that nothing except the model differs between the four searches.

``center_atoms=False`` is deliberate and matches the existing templates: the builder
already puts the tube axis on the box centre, so the templates are axis-centred and the
analysis reads an axis offset near zero.

Usage:  python run_scripts/simulate_lattice_competition.py [--flat-b-only] [--device 0]

For another dataset, pass ``--pixel-size`` and ``--box``: keep the PHYSICAL box the same
(600 x 0.9194 = 552 A), e.g. the in situ set at 1.056 A/px takes ``--box 528``. The map
name carries the pixel size, so the in vitro maps are never overwritten. A dataset at
another voltage also needs ``--voltage`` and ``--mtf``: the 16Sep2026 kinesin + GMPCPP set
(Talos Arctica, K3 CDS) is ``--pixel-size 1.13 --box 488 --voltage 200 --mtf
k3_200kV_CDS``. Its maps are the only ones at 1.13 A/px, so the name stays unambiguous.

    python run_scripts/simulate_lattice_competition.py --pixel-size 1.056 --box 528 \
        --stems 6dpu_atoms_6dpv_lattice_13pf_2rings_flatB
"""

import argparse
import pathlib
import time

import torch
import ttsim3d.simulate3d
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

def damage_equivalent_dose(exposure: float, voltage: float) -> float:
    """Exposure on the 300 kV damage scale that ttsim3d's 3-D dose filter uses.

    The filter is Grant & Grigorieff's critical-exposure curve, measured at 300 kV, and
    ttsim3d does not scale it by voltage. Damage per e-/A^2 is higher at lower voltage;
    RELION -- and torch-fourier-filter's 2-D dose weighting, which RIPPLE applied to the
    movies -- divide the dose by 0.8 at 200 kV and 0.64 at 100 kV. Doing the same here
    keeps the template's damage filter matched to the sum's.
    """
    if abs(voltage - 200.0) <= 2.0:
        return exposure / 0.8
    if abs(voltage - 100.0) <= 2.0:
        return exposure / 0.64
    return exposure


def sim_config(voltage: float = 300.0, mtf: str = "k3_300kV_FL2", exposure: float = 50.0,
               apply_dqe: bool = True) -> SimulatorConfig:
    """The simulation settings; voltage, detector, DQE and dose follow the dataset."""
    return SimulatorConfig(
        voltage=voltage,
        apply_dose_weighting=True,
        dose_start=0.0,
        dose_end=damage_equivalent_dose(exposure, voltage),
        dose_filter_modify_signal="rel_diff",
        apply_dqe=apply_dqe,
        upsampling=-1,
        mtf_reference=mtf,
    )


SIM_CONFIG = sim_config()


def map_path(stem: str, pixel_size: float = PIXEL_SIZE,
             neighborhood_b: float | None = None) -> pathlib.Path:
    """Where the map for ``stem`` at ``pixel_size`` lives (``_nb<B>`` if fixed)."""
    tag = "" if neighborhood_b is None else f"_nb{neighborhood_b:g}"
    return MAP_DIR / f"{stem}_{pixel_size}_bscale{B_FACTOR_SCALING}{tag}.mrc"


def fix_neighborhood(neighborhood_b: float) -> None:
    """Make ttsim3d compute every atom out to the radius for B = ``neighborhood_b``.

    ttsim3d sizes each atom's neighbourhood from the MEAN B of the whole model (cisTEM's
    rule) and cuts the potential off beyond it. Models that differ only by added atoms
    of another B then truncate their SHARED atoms differently: 14-PF tubulin alone (B
    31.18) comes out 3.2% weaker than the same tubulin simulated with kinesin motors (B
    61.18) beside it, so "decorated = bare + motors" fails by 3%. One fixed radius for
    every model of a comparison restores exact additivity; take the largest B present
    (raw, before ``b_factor_scaling``), so no atom is cut shorter than its own B needs.
    """
    original = ttsim3d.simulate3d.get_voxel_neighborhood_offsets
    scaled = torch.tensor(float(neighborhood_b) * B_FACTOR_SCALING)

    def fixed(mean_b_factor, upsampled_pixel_size):  # pylint: disable=unused-argument
        return original(scaled.to(getattr(mean_b_factor, "device", "cpu")),
                        upsampled_pixel_size)

    ttsim3d.simulate3d.get_voxel_neighborhood_offsets = fixed


def simulate(stem: str, device: str, pixel_size: float = PIXEL_SIZE,
             volume_shape: tuple[int, int, int] = VOLUME_SHAPE,
             config: SimulatorConfig = SIM_CONFIG,
             neighborhood_b: float | None = None) -> pathlib.Path:
    """Simulate one template and write it next to the existing maps."""
    pdb_path = MODEL_DIR / f"{stem}.pdb"
    mrc_path = map_path(stem, pixel_size, neighborhood_b)
    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path} -- run models/build_mt_templates.py first")

    simulator = Simulator(
        pdb_filepath=str(pdb_path),
        pixel_spacing=pixel_size,
        volume_shape=volume_shape,
        center_atoms=False,
        remove_hydrogens=True,
        b_factor_scaling=B_FACTOR_SCALING,
        additional_b_factor=0,
        simulator_config=config,
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
    # CPU by default: it is what the existing templates were simulated on. The GPU
    # path once failed here for want of nvrtc; it now runs (2-7 s per map) and
    # reproduces the CPU maps to cc 1.000000.
    parser.add_argument("--device", default="cpu", help="'cpu' or e.g. 'cuda:0'")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pixel-size", type=float, default=PIXEL_SIZE,
                        help="Angstroms per voxel; must match the micrograph searched")
    parser.add_argument("--box", type=int, default=VOLUME_SHAPE[0],
                        help="cubic box edge in voxels; keep ~552 A physical")
    parser.add_argument("--voltage", type=float, default=300.0,
                        help="kV of the dataset (the Talos Arctica set is 200)")
    parser.add_argument("--mtf", default="k3_300kV_FL2",
                        help="ttsim3d MTF reference, e.g. k3_200kV_CDS for a K3 in CDS "
                             "mode at 200 kV")
    parser.add_argument("--exposure", type=float, default=50.0,
                        help="total exposure of the movies, e-/A^2 (converted to the "
                             "300 kV damage scale for other voltages)")
    parser.add_argument("--neighborhood-b", type=float, default=None,
                        help="fix every atom's simulated neighbourhood at this raw B "
                             "(see fix_neighborhood); maps get an _nb<B> suffix")
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="YAML with pixel_size, box, voltage, mtf, exposure, "
                             "apply_dqe and stems; overrides the options above")
    args = parser.parse_args()
    apply_dqe = True
    if args.config is not None:
        import yaml  # pylint: disable=import-outside-toplevel

        settings = yaml.safe_load(args.config.read_text())
        args.pixel_size = float(settings["pixel_size"])
        args.box = int(settings["box"])
        args.voltage = float(settings["voltage"])
        args.mtf = str(settings["mtf"])
        args.exposure = float(settings["exposure"])
        apply_dqe = bool(settings.get("apply_dqe", True))
        args.stems = list(settings["stems"])
        args.overwrite = bool(settings.get("overwrite", args.overwrite))
        args.neighborhood_b = settings.get("neighborhood_b", args.neighborhood_b)
    config = sim_config(args.voltage, args.mtf, args.exposure, apply_dqe)
    if args.neighborhood_b is not None:
        fix_neighborhood(args.neighborhood_b)
    shape = (args.box,) * 3

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    wanted = args.stems or [
        t for t in TEMPLATES if not args.flat_b_only or t.endswith("_flatB")
    ]

    print(f"Simulating {len(wanted)} templates on {args.device} at {args.pixel_size} A/px, "
          f"{args.box}^3 ({args.box * args.pixel_size:.0f} A), {args.voltage:.0f} kV, "
          f"MTF/DQE {args.mtf} (DQE {'on' if apply_dqe else 'off'}), exposure "
          f"{args.exposure:g} e-/A^2 -> dose filter 0-{config.dose_end:g} on the 300 kV "
          f"scale")
    for i, stem in enumerate(wanted, start=1):
        mrc_path = map_path(stem, args.pixel_size, args.neighborhood_b)
        if mrc_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(wanted)}] {stem}: exists, skipping")
            continue
        start = time.monotonic()
        simulate(stem, args.device, args.pixel_size, shape, config, args.neighborhood_b)
        print(
            f"  [{i}/{len(wanted)}] {stem}: {time.monotonic() - start:6.1f} s"
            f"  -> {mrc_path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
