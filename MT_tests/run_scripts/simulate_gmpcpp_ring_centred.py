"""Simulate the GMPCPP 2-ring template with its mass centred in the box.

``maps/GMPCPP_2rings_0.9194_bscale0.5.mrc`` (process_GMPCPP.ipynb, from
``models/6dpu_2rings_aligned_zero.pdb``: the deposited 14-PF 6DPU, two rings) has its
axis on the box centre but its mass ~94 px (86 A) below it ALONG the axis. The SVD search
(``svd_match_filament.py``) refuses such a template: Panther-EM rotates about the box
centre, and re-centring the map would move every detection along the tube.

This translates the same atoms along the axis only, by the map's own centre of positive
density rounded to whole voxels (the quantity the SVD search checks), and simulates them
with the notebook's parameters (the same as ``simulate_patch_centred.py``). Nothing else
changes.

Good for SEARCHING (the SVD autotrace with it lands 2 A from the hand-checked axis), NOT
as the ring of ``run_svd_pipeline.sh``: its 56 chains are a different slice of the
lattice from those of the templates ``models/build_mt_templates.py`` makes (same
polarity and hand, but ~19 A out of axial register with the built 14-PF ring, plus one
non-helical step, its seam), and the pipeline scores those templates -- the PF rivals,
both lattices, the patch -- at the ring's own poses. A run with it as the ring scored
every built rival at z ~5 against its own 10.4, and Stage 4's sites deviated 8.7 A rms
(823: 4-5 A). The pipeline's GMPCPP ring is the expanded14 set's own reference,
``6dpu_14pf_2rings_flatB`` (6DPU atoms and lattice, already centred), as 823's is the
compacted13 reference.

Writes
    models/GMPCPP_2rings_centred.pdb               the ring geometry for Stages 3-5
    maps/GMPCPP_2rings_centred_0.9194_bscale0.5.mrc

and checks that
    1. the UNtranslated atoms, simulated the same way, match the existing map shell by
       shell (FSC >= 0.999 at every resolution: same atoms, same geometry). The radial
       WEIGHTING differs: the existing map was made in June 2025 with an older ttsim3d
       (and is mean-subtracted); today's ttsim3d with the same call -- what
       simulate_patch_centred.py and every rival map at 0.9194 A were made with --
       weights 4-8 A ~0.6x relative to low resolution. Reported, not corrected;
    2. the new map's centre of positive density rounds to the box centre on every axis;
    3. the new map is the re-simulated old one moved along the axis (cc after the
       whole-voxel roll).

Usage (leopard-em env, from MT_tests/):
    python run_scripts/simulate_gmpcpp_ring_centred.py [--device cuda:0] [--overwrite]
"""

import argparse
import pathlib

import mrcfile
import numpy as np
from ttsim3d.models import Simulator, SimulatorConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_PDB = ROOT / "models" / "6dpu_2rings_aligned_zero.pdb"
OLD_MAP = ROOT / "maps" / "GMPCPP_2rings_0.9194_bscale0.5.mrc"
STEM = "GMPCPP_2rings_centred"
PIXEL = 0.9194
SHAPE = (600, 600, 600)


def simulator(pdb: pathlib.Path) -> Simulator:
    """process_GMPCPP.ipynb's settings."""
    config = SimulatorConfig(
        voltage=300.0,
        apply_dose_weighting=True,
        dose_start=0.0,
        dose_end=50.0,
        dose_filter_modify_signal="rel_diff",
        upsampling=-1,
        mtf_reference="k3_300kV_FL2",
    )
    return Simulator(
        pdb_filepath=str(pdb),
        pixel_spacing=PIXEL,
        volume_shape=SHAPE,
        center_atoms=False,
        remove_hydrogens=True,
        b_factor_scaling=0.5,
        additional_b_factor=0,
        simulator_config=config,
    )


def com_offset(volume: np.ndarray) -> np.ndarray:
    """Centre of positive density, (z, y, x) voxels from the box centre n // 2 -- as
    ``svd_trace_maps.mass_radius`` computes it."""
    w = np.clip(volume, 0, None).astype(np.float64)
    total = w.sum()
    out = []
    for axis in range(3):
        profile = w.sum(axis=tuple(a for a in range(3) if a != axis))
        out.append(float((profile * (np.arange(volume.shape[axis]) - volume.shape[axis] // 2)).sum()
                         / total))
    return np.array(out)


def shells(a: np.ndarray, b: np.ndarray, edges_angstrom=(999, 40, 20, 12, 8, 6, 4, 3, 2.2)):
    """(FSC, amplitude ratio b/a) per resolution shell between ``edges_angstrom``."""
    import torch  # pylint: disable=import-outside-toplevel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fa = torch.fft.rfftn(torch.from_numpy(a - a.mean()).to(dev))
    fb = torch.fft.rfftn(torch.from_numpy(b - b.mean()).to(dev))
    n = a.shape[0]
    f1 = torch.fft.fftfreq(n, device=dev)
    k = torch.sqrt(f1[:, None, None] ** 2 + f1[None, :, None] ** 2
                   + torch.fft.rfftfreq(n, device=dev)[None, None, :] ** 2) / PIXEL
    out = []
    for lo, hi in zip(edges_angstrom[:-1], edges_angstrom[1:]):
        m = (k >= 1 / lo) & (k < 1 / hi)
        x, y = fa[m], fb[m]
        fsc = float((x * y.conj()).real.sum() / torch.sqrt((x.abs() ** 2).sum()
                                                          * (y.abs() ** 2).sum()))
        out.append((lo, hi, fsc, float(torch.sqrt((y.abs() ** 2).sum() / (x.abs() ** 2).sum()))))
    return out


def cc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def translate_pdb(source: pathlib.Path, target: pathlib.Path, dz: float) -> None:
    """Copy ``source`` with every atom's z moved by ``dz`` Angstrom."""
    lines = []
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            z = float(line[46:54]) + dz
            line = f"{line[:46]}{z:8.3f}{line[54:]}"
        lines.append(line)
    target.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_pdb = ROOT / "models" / f"{STEM}.pdb"
    target_map = ROOT / "maps" / f"{STEM}_{PIXEL}_bscale0.5.mrc"
    if target_map.exists() and not args.overwrite:
        raise SystemExit(f"{target_map} exists; --overwrite to redo it")

    old = np.asarray(mrcfile.read(OLD_MAP), dtype=np.float32)
    old_com = com_offset(old)
    print(f"{OLD_MAP.name}: centre of density (z, y, x) {np.round(old_com, 2)} voxels")

    # 1. same atoms, same call -> the same structure as the existing map, shell by shell?
    again = simulator(SOURCE_PDB).run(device=args.device).cpu().numpy()
    print(f"  {SOURCE_PDB.name} re-simulated: cc with the existing map {cc(again, old):.4f}")
    table = shells(old, again)
    for lo, hi, fsc, ratio in table:
        print(f"    {hi:5.1f}-{lo:5.0f} A  FSC {fsc:.5f}  amplitude now / June 2025 {ratio:.3f}")
    if min(fsc for *_, fsc, _ in table) < 0.999:
        raise SystemExit("not the same structure as the existing map; not writing")

    # 2. translate along the axis (map axis 0 = PDB z) by the density's own offset, in
    #    WHOLE voxels: the map is then exactly the old one rolled, on the same voxel lattice
    dz = round(-old_com[0]) * PIXEL
    translate_pdb(SOURCE_PDB, target_pdb, dz)
    print(f"  wrote {target_pdb.relative_to(ROOT)}: atoms moved {dz:+.3f} A along z")
    new = simulator(target_pdb).run(device=args.device).cpu().numpy()
    new_com = com_offset(new)
    print(f"  new map: centre of density (z, y, x) {np.round(new_com, 2)} voxels")
    if np.any(np.round(new_com) != 0):
        raise SystemExit("still off-centre by a whole voxel; not writing the map")

    # 3. the re-simulated old map moved along the axis
    shift = int(round(-old_com[0]))
    check3 = cc(np.roll(again, shift, axis=0), new)
    print(f"  re-simulated old map rolled {shift:+d} voxels along z: cc with the new map "
          f"{check3:.6f}")
    if check3 < 0.999:
        raise SystemExit("the translated map is not the old one moved along the axis")

    with mrcfile.new(target_map, overwrite=True) as handle:
        handle.set_data(new.astype(np.float32))
        handle.voxel_size = PIXEL
    print(f"  wrote {target_map.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
