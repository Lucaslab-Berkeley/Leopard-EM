"""Build a PATCH template -- a few adjacent protofilaments, not the closed tube.

A ring template cannot read the monomer register: it is N-fold pseudo-symmetric, its
roll angle is degenerate, and it averages the whole circumference, which is precisely
what a seam breaks. The measurement that works on real data uses a patch, whose centre
of mass sits ~87 A off the tube axis so that rolling it genuinely changes the projection.

The existing ``6dpu_4_patches`` template is 14-protofilament and was built outside this
repository. This makes the matching object for any N and either lattice, from the same
generator as every other template here -- so a patch and the ring it came from differ in
nothing but which protofilaments were kept.

Usage:
    python models/build_patch_template.py --protofilaments 13 --lattice 6dpv --keep 5
    python models/build_patch_template.py --protofilaments 13 --keep 5 --rings 4
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import build_mt_templates as B


def main() -> None:
    """Build one patch template and report where its axis ends up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protofilaments", type=int, default=13,
                        help="protofilament number of the tube the patch is cut from")
    parser.add_argument("--keep", type=int, default=5,
                        help="how many adjacent protofilaments to keep")
    parser.add_argument("--first", type=int, default=0,
                        help="index of the first protofilament kept")
    parser.add_argument("--rings", type=int, default=2)
    parser.add_argument("--atoms", default="6dpu", choices=sorted(B.DEPOSITIONS))
    parser.add_argument("--lattice", default=None, choices=sorted(B.DEPOSITIONS),
                        help="repeat source; defaults to --atoms")
    parser.add_argument("--outdir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent)
    args = parser.parse_args()

    lattice_source = args.lattice or args.atoms
    print("Loading depositions")
    models = {pid: B.load(pid, args.outdir) for pid in B.DEPOSITIONS}
    keep_atoms = B.common_atoms(models)

    frames, all_b = {}, []
    for pid, model in models.items():
        point, axis = B.measure_axis(model, keep_atoms)
        rot, trans = B.canonical_transform(point, axis)
        frames[pid] = {
            "rot": rot,
            "trans": trans,
            "steps": B.measure_axial_steps(model, rot, trans, keep_atoms),
            "twist_deg": B.DEPOSITIONS[pid]["twist_deg"],
            "lateral_rise_a": B.DEPOSITIONS[pid]["lateral_rise_a"],
        }
        all_b += [
            a.b_iso
            for c in model
            for r in c
            for a in r
            if B.is_kept(r, a, keep_atoms[c.name])
        ]
    flat_b = float(np.mean(all_b))

    radius = B.measure_radius(
        models[args.atoms], frames[args.atoms]["rot"], frames[args.atoms]["trans"],
        keep_atoms,
    )
    lattice = B.protofilament_lattice(
        frames[lattice_source], radius, args.protofilaments
    )

    subset = [args.first + offset for offset in range(args.keep)]
    structure = B.build(
        models[args.atoms],
        frames[args.atoms]["rot"],
        frames[args.atoms]["trans"],
        lattice,
        keep_atoms,
        args.rings,
        flat_b,
        protofilament_subset=subset,
    )

    source_tag = (
        f"{args.atoms}"
        if lattice_source == args.atoms
        else f"{args.atoms}_atoms_{lattice_source}_lattice"
    )
    name = (
        f"{source_tag}_{args.protofilaments}pf_patch{args.keep}"
        f"_{args.rings}rings_flatB.pdb"
    )
    path = args.outdir / name
    structure.write_pdb(str(path))

    n_atoms = sum(len(r) for c in structure[0] for r in c)
    print(f"\nkept protofilaments {subset} of {args.protofilaments}, "
          f"spanning {args.keep * 360.0 / args.protofilaments:.0f}° of the circle")
    print(f"  {name}  chains {len(structure[0])}  atoms {n_atoms}")

    from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry

    geometry = TemplateLatticeGeometry.from_pdb(str(path))
    print(f"  measured: N {geometry.n_protofilaments}, "
          f"radius {geometry.subunit_radius_angstrom:.2f} Å, "
          f"rise {geometry.rise_angstrom:.3f} Å, "
          f"axis offset {geometry.axis_offset_magnitude_angstrom:.2f} Å")
    print("  (a ring sits at ~0 Å; the offset is what makes the roll angle readable)")
    print(f"\nSimulate with:\n  python run_scripts/simulate_lattice_competition.py "
          f"--stems {path.stem}")


if __name__ == "__main__":
    main()
