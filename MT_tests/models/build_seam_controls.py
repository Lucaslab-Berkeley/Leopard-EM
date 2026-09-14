"""Templates that separate the two things a seam signal could come from.

Locating a seam means telling alpha from beta, and a template's dimer-only (half-order)
Fourier power says how well it could. But two quite different properties feed that power
and they have opposite prospects:

* **density** -- alpha and beta are different proteins. This is the one that needs high
  resolution, and it is what `subunit_contrast` was meant to probe.
* **position** -- the intra-dimer and inter-dimer steps are not equal (41 vs 43 A). That
  is a POSITIONAL signature, present at every resolution, and it is seam route 4.

Observed templates contain both, so a measurement on them alone cannot say which is
carrying the signal, or whether the low-resolution part is real. These two controls
separate them, each changing exactly one thing:

    samemonomer  every monomer is the same chain, kept at the real alternating
                 heights          -> position only, no density difference
    equalsteps   the real alpha and beta chains, moved onto a perfectly regular
                 monomer lattice  -> density only, no positional alternation

Built at 13 protofilaments, which is what the competition and moiré routes now favour.

Usage:  python models/build_seam_controls.py [--protofilaments 13] [--rings 2]
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import build_mt_templates as B


def equal_step_lattice(lattice: dict) -> dict:
    """The same lattice with the 41/43 A alternation removed, dimer repeat preserved."""
    monomer = lattice["steps"]["dimer"] / 2.0
    return {
        **lattice,
        "steps": {
            "beta_alpha": monomer,
            "alpha_beta": monomer,
            "dimer": lattice["steps"]["dimer"],
        },
    }


def write_monomer_pair(models, frames, keep, flat_b, source, outdir):
    """Alpha and beta alone, each on the SAME lattice site, for a direct comparison.

    For an infinite filament of alternating monomers the correlation kept one monomer out
    of register is the normalised cross-correlation between the two monomers -- so per
    resolution shell it is just FSC(alpha, beta), whose null is exactly 1 with no
    template-length term to correct for.

    Both chains are written in one common frame and centred by the SAME vector, so the
    two volumes are directly comparable; centring each on its own centre of mass would
    superpose them by construction and destroy the measurement.
    """
    import gemmi

    model = models[source]
    rot, trans = frames[source]["rot"], frames[source]["trans"]
    beta_alpha = frames[source]["steps"]["beta_alpha"]

    # COLUMN is beta, alpha, beta, alpha bottom to top: lift the beta onto the alpha.
    beta_name, alpha_name = B.COLUMN[0], B.COLUMN[1]
    placements = {alpha_name: 0.0, beta_name: beta_alpha}

    centre = rot @ B.chain_centroid(model[alpha_name], keep[alpha_name]) + trans

    for label, name in (("alpha", alpha_name), ("beta", beta_name)):
        structure = gemmi.Structure()
        structure.spacegroup_hm = "P 1"
        out_model = gemmi.Model("1")
        out_chain = gemmi.Chain("A")
        for residue in model[name]:
            out_residue = gemmi.Residue()
            out_residue.name = residue.name
            out_residue.seqid = residue.seqid
            out_residue.het_flag = residue.het_flag
            for atom in residue:
                if not B.is_kept(residue, atom, keep[name]):
                    continue
                point = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                point = rot @ point + trans
                point = point + np.array([0.0, 0.0, placements[name]]) - centre
                out_atom = gemmi.Atom()
                out_atom.name = atom.name
                out_atom.element = atom.element
                out_atom.altloc = atom.altloc
                out_atom.occ = atom.occ
                out_atom.b_iso = flat_b
                out_atom.pos = gemmi.Position(*point)
                out_residue.add_atom(out_atom)
            if len(out_residue) > 0:
                out_chain.add_residue(out_residue)
        out_model.add_chain(out_chain)
        structure.add_model(out_model)
        structure.setup_entities()
        filename = f"{source}_seam_monomer_{label}.pdb"
        structure.write_pdb(str(outdir / filename))
        n_atoms = sum(len(r) for c in structure[0] for r in c)
        print(f"  {filename:44s} atoms {n_atoms}")


def main() -> None:
    """Build the observed template and its two single-variable controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protofilaments", type=int, default=13)
    parser.add_argument("--rings", type=int, default=2)
    parser.add_argument("--atoms", default="6dpu", choices=sorted(B.DEPOSITIONS))
    parser.add_argument("--outdir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent)
    parser.add_argument("--monomer-pair", action="store_true",
                        help="also write alpha and beta alone, on a common lattice site")
    args = parser.parse_args()

    print("Loading depositions")
    models = {pid: B.load(pid, args.outdir) for pid in B.DEPOSITIONS}
    keep = B.common_atoms(models)

    frames, all_b = {}, []
    for pid, model in models.items():
        point, axis = B.measure_axis(model, keep)
        rot, trans = B.canonical_transform(point, axis)
        frames[pid] = {
            "rot": rot,
            "trans": trans,
            "steps": B.measure_axial_steps(model, rot, trans, keep),
            "twist_deg": B.DEPOSITIONS[pid]["twist_deg"],
            "lateral_rise_a": B.DEPOSITIONS[pid]["lateral_rise_a"],
        }
        all_b += [
            a.b_iso for c in model for r in c for a in r if B.is_kept(r, a, keep[c.name])
        ]
    flat_b = float(np.mean(all_b))

    source = args.atoms
    steps = frames[source]["steps"]
    print(f"  {source}: beta->alpha {steps['beta_alpha']:.3f} A, "
          f"alpha->beta {steps['alpha_beta']:.3f} A, "
          f"alternation {abs(steps['beta_alpha'] - steps['alpha_beta']):.3f} A")
    print(f"  pooled mean B {flat_b:.2f}")

    radius = B.measure_radius(
        models[source], frames[source]["rot"], frames[source]["trans"], keep
    )
    base = B.protofilament_lattice(frames[source], radius, args.protofilaments)

    # The alpha chain of the column, used for every monomer in the samemonomer control.
    alpha = B.COLUMN[1]
    original_column = list(B.COLUMN)

    # The full 2x2. "periodic" is the floor: nothing distinguishes its two half-repeats,
    # so whatever register penalty it still shows is the template's finite length and
    # nothing else. Every variant is a real built template of identical length, which is
    # what makes the comparison clean -- a Fourier-symmetrised volume is not a valid
    # null, because averaging a finite object with a shifted copy smears its ends and
    # charges a penalty of its own.
    variants = [
        ("observed", base, original_column),
        ("samemonomer", base, [alpha] * 4),
        ("equalsteps", equal_step_lattice(base), original_column),
        ("periodic", equal_step_lattice(base), [alpha] * 4),
    ]

    tag = f"{args.protofilaments}pf_{args.rings}rings_flatB"
    print(f"\nBuilding {args.rings}-ring controls at "
          f"{args.protofilaments} protofilaments")
    for label, lattice, column in variants:
        B.COLUMN = column
        try:
            structure = B.build(
                models[source], frames[source]["rot"], frames[source]["trans"],
                lattice, keep, args.rings, flat_b,
            )
        finally:
            B.COLUMN = original_column

        name = f"{source}_seam_{label}_{tag}.pdb"
        structure.write_pdb(str(args.outdir / name))
        n_atoms = sum(len(r) for c in structure[0] for r in c)
        print(f"  {name:44s} chains {len(structure[0]):3d}  atoms {n_atoms}")

    if args.monomer_pair:
        print("\nWriting the monomer pair")
        write_monomer_pair(models, frames, keep, flat_b, source, args.outdir)

    print("\nSimulate with:\n  python run_scripts/simulate_lattice_competition.py "
          "--stems " + " ".join(f"{source}_seam_{label}_{tag}"
                                for label, _, _ in variants))


if __name__ == "__main__":
    main()
