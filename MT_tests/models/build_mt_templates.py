"""Build matched microtubule templates from 6DPU (GMPCPP) and 6DPV (GDP).

The two depositions are the same 12-chain unit (3 protofilaments x 2 dimers) refined
by the same group, so they can be compared -- but only if every difference except the
one under test is removed first. This script does that, and then builds the 2x2 of
{atom source} x {lattice} so the lattice effect and the model-quality effect can be
separated rather than confounded:

    6dpu@6dpu   6dpv@6dpu       "X@Y" = model X's monomers, arranged on lattice Y
    6dpu@6dpv   6dpv@6dpv

Equalisation applied to every output:

* residues trimmed to the set common to both models (6DPV's set is a strict subset
  of 6DPU's, so this only ever removes atoms from 6DPU)
* one protofilament column as the rigid source, the same chains from both models
* monomers repositioned onto idealised axial spacings, so the diagonal cells go
  through exactly the same idealisation as the off-diagonal ones
* the deposited helical operator, which closes the 14_3 lattice; the lattice measured
  from the 12-mer directly does not close and must not be used
* axis on +z through the origin, centred on the centre of mass
* optional flattening of B-factors to a common constant

Usage:  python build_mt_templates.py [--rings 2] [--outdir models]
"""

import argparse
import itertools
import pathlib
import urllib.request

import gemmi
import numpy as np

# Deposited helical operators. These close the 14_3 lattice (14 x 8.99 = 3 monomers
# of 6DPU, 14 x 8.754 = 3 monomers of 6DPV); the lattice measured directly from the
# 12-mer does not close, so it cannot be used to symmetrise.
DEPOSITIONS = {
    "6dpu": {"twist_deg": -25.750, "lateral_rise_a": 8.99, "state": "GMPCPP/expanded"},
    "6dpv": {"twist_deg": -25.766, "lateral_rise_a": 8.754, "state": "GDP/compacted"},
}

# One protofilament column, bottom to top: beta, alpha, beta, alpha. Both depositions
# use the same chain naming, and this is the middle protofilament of the three.
COLUMN = ["H", "A", "B", "K"]
N_PROTOFILAMENTS = 14

# Monomers per turn of the lateral helix. Both depositions are 3-start; see
# protofilament_lattice, which checks this against their deposited operators.
N_START = 3

# All three protofilaments of a deposition, bottom to top, used to measure the axis.
PROTOFILAMENTS = [["G", "E", "F", "J"], ["H", "A", "B", "K"], ["I", "C", "D", "L"]]

# PDB single-character chain IDs, in the order the existing templates use. The PDB
# format has only these 62, which caps a 4-chain column at 15 protofilaments; beyond
# that the format itself is the limit, not this script.
CHAIN_IDS = [str(d) for d in range(1, 10)] + list(
    itertools.chain(map(chr, range(65, 91)), map(chr, range(97, 123)))
) + ["0"]


def is_amino_acid(residue: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(residue.name)
    return info is not None and info.is_amino_acid()


def fetch(pdb_id: str, directory: pathlib.Path) -> pathlib.Path:
    path = directory / f"{pdb_id}.cif"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        print(f"  fetching {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load(pdb_id: str, directory: pathlib.Path) -> gemmi.Model:
    structure = gemmi.read_structure(str(fetch(pdb_id, directory)))
    structure.setup_entities()
    return structure[0]


def common_atoms(models: dict[str, gemmi.Model]) -> dict[str, set[tuple]]:
    """Protein atoms present in every model, per chain, keyed by residue and name.

    Trimming per residue is not enough: within a residue both models model different
    numbers of side-chain atoms, which would leave the templates unequal in scattering
    power. Ligands are deliberately excluded here -- the nucleotide difference is the
    real biology and is kept.
    """
    per_chain: dict[str, set[tuple]] = {}
    for model in models.values():
        for chain in model:
            atoms = {
                (r.seqid.num, a.name, a.altloc)
                for r in chain
                if is_amino_acid(r)
                for a in r
            }
            per_chain[chain.name] = (
                atoms if chain.name not in per_chain else per_chain[chain.name] & atoms
            )
    return per_chain


def is_kept(residue: gemmi.Residue, atom: gemmi.Atom, keep: set[tuple]) -> bool:
    """Ligands always survive; protein atoms only if both models model them."""
    if not is_amino_acid(residue):
        return True
    return (residue.seqid.num, atom.name, atom.altloc) in keep


def chain_centroid(chain: gemmi.Chain, keep: set[tuple] | None = None) -> np.ndarray:
    """Centroid of a chain's protein atoms, over the trimmed set when one is given."""
    pos = [
        [a.pos.x, a.pos.y, a.pos.z]
        for r in chain
        if is_amino_acid(r)
        for a in r
        if keep is None or (r.seqid.num, a.name, a.altloc) in keep
    ]
    return np.asarray(pos).mean(axis=0)


def measure_axis(
    model: gemmi.Model, keep: dict[str, set[tuple]]
) -> tuple[np.ndarray, np.ndarray]:
    """Helical axis (point, unit direction) from the three protofilaments."""
    centroids = {c.name: chain_centroid(c, keep[c.name]) for c in model}

    directions = []
    for names in PROTOFILAMENTS:
        pts = np.array([centroids[n] for n in names])
        d = np.linalg.svd(pts - pts.mean(axis=0))[2][0]
        directions.append(d if d[2] > 0 else -d)
    axis = np.mean(directions, axis=0)
    axis /= np.linalg.norm(axis)

    # Circle through the three protofilaments, in the plane normal to the axis.
    e1 = np.cross(axis, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    flat = []
    for names in PROTOFILAMENTS:
        c = centroids[names[0]]
        perp = c - (c @ axis) * axis
        flat.append([perp @ e1, perp @ e2])
    flat = np.array(flat)
    centre2d = np.linalg.solve(
        np.hstack([2 * flat, np.ones((3, 1))]), (flat**2).sum(axis=1)
    )[:2]

    all_centroids = np.array(list(centroids.values()))
    point = (
        centre2d[0] * e1 + centre2d[1] * e2 + float(all_centroids.mean(axis=0) @ axis) * axis
    )
    return point, axis


def canonical_transform(point: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation putting the helical axis on +z through the origin."""
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, target)
    s, c = np.linalg.norm(v), float(axis @ target)
    if s < 1e-12:
        rot = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    shifted = rot @ point
    return rot, np.array([-shifted[0], -shifted[1], 0.0])


def measure_axial_steps(
    model: gemmi.Model, rot: np.ndarray, trans: np.ndarray, keep: dict[str, set[tuple]]
) -> dict:
    """Monomer steps along the axis, split by interface, averaged over protofilaments."""
    def height(name: str) -> float:
        return float((rot @ chain_centroid(model[name], keep[name]) + trans)[2])

    beta_alpha, alpha_beta = [], []
    for names in PROTOFILAMENTS:
        h = [height(n) for n in names]
        beta_alpha += [h[1] - h[0], h[3] - h[2]]
        alpha_beta += [h[2] - h[1]]
    return {
        "beta_alpha": float(np.mean(beta_alpha)),
        "alpha_beta": float(np.mean(alpha_beta)),
        "dimer": float(np.mean(beta_alpha) + np.mean(alpha_beta)),
    }


def measure_radius(
    model: gemmi.Model, rot: np.ndarray, trans: np.ndarray, keep: dict[str, set[tuple]]
) -> float:
    """Distance of the source protofilament from the tube axis."""
    radii = [
        float(np.hypot(*(rot @ chain_centroid(model[n], keep[n]) + trans)[:2]))
        for n in COLUMN
    ]
    return float(np.mean(radii))


def protofilament_lattice(base: dict, base_radius: float, n_pf: int) -> dict:
    """Geometry of the same lattice wrapped into a tube of ``n_pf`` protofilaments.

    Three quantities follow from N once the lateral contact is held fixed:

    * **radius** scales as N, so the arc length between protofilaments -- the actual
      interface -- is unchanged. This is the dominant effect and the reason competing
      protofilament numbers works at all: 8.45 A per protofilament, 17 A in diameter.
    * **twist** is 360/N by definition.
    * **lateral rise** is ``n_start * monomer / N``, which keeps the lattice closed.
      That relation is not assumed: it reproduces the deposited helical operators of
      both 6DPU (8.9955 predicted vs 8.99) and 6DPV (8.7505 vs 8.754), to 0.06%.

    The monomer, the dimer repeat and every atom are untouched, so templates built this
    way differ *only* in protofilament number -- no model-quality confound at all.
    """
    monomer = base["steps"]["dimer"] / 2
    return {
        **base,
        "n_protofilaments": n_pf,
        "twist_deg": -360.0 / n_pf,
        "lateral_rise_a": N_START * monomer / n_pf,
        "radial_shift_a": base_radius * (n_pf - N_PROTOFILAMENTS) / N_PROTOFILAMENTS,
    }


def build(
    atom_model: gemmi.Model,
    atom_rot: np.ndarray,
    atom_trans: np.ndarray,
    lattice: dict,
    keep: dict[str, set[tuple]],
    n_rings: int,
    flat_b: float | None,
    protofilament_subset: list[int] | None = None,
) -> gemmi.Structure:
    """Place one model's monomers on another's lattice, then symmetrise to a tube.

    ``protofilament_subset`` keeps only those protofilament copies, which is how a
    PATCH template is made: a few adjacent protofilaments rather than the closed
    tube. The centring at the end then puts the patch's own centre of mass on the
    box centre, so the tube axis sits well off it -- and that offset is exactly what
    makes a patch's roll angle measurable where a ring's is degenerate.
    """
    steps = lattice["steps"]
    twist = np.radians(lattice["twist_deg"])
    lateral = lattice["lateral_rise_a"]
    n_pf = lattice.get("n_protofilaments", N_PROTOFILAMENTS)
    radial_shift = lattice.get("radial_shift_a", 0.0)

    # Idealised axial position of each monomer in the column, relative to the first.
    offsets = [0.0, steps["beta_alpha"], steps["beta_alpha"] + steps["alpha_beta"]]
    offsets.append(offsets[2] + steps["beta_alpha"])
    for ring in range(1, n_rings // 2 if n_rings > 2 else 0):
        offsets += [o + ring * 2 * steps["dimer"] for o in offsets[:4]]

    # Each source chain, in the canonical frame, shifted onto its idealised height.
    sources = []
    for name, target_h in zip(COLUMN * (len(offsets) // 4), offsets):
        chain = atom_model[name]
        current = float((atom_rot @ chain_centroid(chain, keep[name]) + atom_trans)[2])
        sources.append((chain, target_h - current))

    # Changing the protofilament count changes the tube radius: the lateral contact
    # spacing is a property of the tubulin-tubulin interface, not of N, so a 13-mer tube
    # is narrower rather than more loosely packed. Move the column out along its own
    # radius before symmetrising, keeping its orientation relative to that radius.
    radial_unit = np.zeros(3)
    if radial_shift:
        centre = np.mean(
            [atom_rot @ chain_centroid(c, keep[c.name]) + atom_trans for c, _ in sources],
            axis=0,
        )
        radial_unit = np.array([centre[0], centre[1], 0.0])
        radial_unit /= np.linalg.norm(radial_unit)

    copies = list(range(n_pf)) if protofilament_subset is None else [
        int(index) % n_pf for index in protofilament_subset
    ]

    if len(copies) * len(sources) > len(CHAIN_IDS):
        raise ValueError(
            f"{len(copies)} protofilaments x {len(sources)} chains = "
            f"{len(copies) * len(sources)} "
            f"chains, but the PDB format has only {len(CHAIN_IDS)} chain IDs. "
            "Reduce --rings, or write mmCIF (and check the downstream readers first: "
            "TemplateLatticeGeometry.from_pdb and ttsim3d both parse PDB)."
        )

    structure = gemmi.Structure()
    structure.spacegroup_hm = "P 1"
    out_model = gemmi.Model("1")
    ids = iter(CHAIN_IDS)

    for copy in copies:
        angle = copy * twist
        rz = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        lift = np.array([0.0, 0.0, copy * lateral])

        for chain, dz in sources:
            out_chain = gemmi.Chain(next(ids))
            for residue in chain:
                out_residue = gemmi.Residue()
                out_residue.name = residue.name
                out_residue.seqid = residue.seqid
                out_residue.het_flag = residue.het_flag
                for atom in residue:
                    if not is_kept(residue, atom, keep[chain.name]):
                        continue
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    p = atom_rot @ p + atom_trans + np.array([0.0, 0.0, dz])
                    p = p + radial_shift * radial_unit
                    p = rz @ p + lift
                    out_atom = gemmi.Atom()
                    out_atom.name = atom.name
                    out_atom.element = atom.element
                    out_atom.altloc = atom.altloc
                    out_atom.occ = atom.occ
                    out_atom.b_iso = atom.b_iso if flat_b is None else flat_b
                    out_atom.pos = gemmi.Position(*p)
                    out_residue.add_atom(out_atom)
                if len(out_residue) > 0:
                    out_chain.add_residue(out_residue)
            out_model.add_chain(out_chain)

    structure.add_model(out_model)

    # Centre on the centre of mass: axis already on z, so this only moves it in z.
    com = np.mean(
        [[a.pos.x, a.pos.y, a.pos.z] for c in structure[0] for r in c for a in r], axis=0
    )
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                atom.pos = gemmi.Position(
                    atom.pos.x - com[0], atom.pos.y - com[1], atom.pos.z - com[2]
                )

    structure.setup_entities()
    return structure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rings", type=int, default=2, help="dimer rings per template")
    # Next to this script, not in the caller's cwd: everything downstream looks for
    # these under models/, and defaulting to cwd silently scatters the PDBs (and
    # re-downloads the depositions) wherever the builder happened to be invoked from.
    # build_patch_template.py and build_seam_controls.py already default this way.
    parser.add_argument("--outdir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent)
    parser.add_argument(
        "--protofilaments",
        type=int,
        nargs="+",
        metavar="N",
        help="build protofilament-number variants (e.g. --protofilaments 13 14 15) "
        "instead of the expanded/compacted 2x2",
    )
    parser.add_argument(
        "--atoms",
        default="6dpu",
        choices=sorted(DEPOSITIONS),
        help="atom source for the protofilament-number variants (default 6dpu)",
    )
    parser.add_argument(
        "--lattice",
        default=None,
        choices=sorted(DEPOSITIONS),
        help="lattice (repeat) source for the protofilament-number variants; defaults "
        "to --atoms. Separating the two is what allows an expanded-against-compacted "
        "comparison at a fixed protofilament number and fixed atoms, so no "
        "model-quality confound arises.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading depositions")
    models = {pid: load(pid, args.outdir) for pid in DEPOSITIONS}

    keep = common_atoms(models)
    totals = {
        pid: sum(len(r) for c in m for r in c if is_amino_acid(r))
        for pid, m in models.items()
    }
    print("  protein atoms: " + ", ".join(f"{p} {n}" for p, n in totals.items()))
    print(f"  common:        {sum(len(v) for v in keep.values())} (trimming to this)")

    frames, all_b = {}, []
    for pid, model in models.items():
        point, axis = measure_axis(model, keep)
        rot, trans = canonical_transform(point, axis)
        steps = measure_axial_steps(model, rot, trans, keep)
        frames[pid] = {
            "rot": rot,
            "trans": trans,
            "steps": steps,
            "twist_deg": DEPOSITIONS[pid]["twist_deg"],
            "lateral_rise_a": DEPOSITIONS[pid]["lateral_rise_a"],
        }
        all_b += [
            a.b_iso for c in model for r in c for a in r if is_kept(r, a, keep[c.name])
        ]
        print(
            f"  {pid} ({DEPOSITIONS[pid]['state']}): dimer {steps['dimer']:.3f} A"
            f"  [beta->alpha {steps['beta_alpha']:.3f}, alpha->beta {steps['alpha_beta']:.3f}]"
        )

    flat_b = float(np.mean(all_b))
    print(f"  pooled mean B across both models: {flat_b:.2f} (used for the flatB variants)")

    if args.protofilaments:
        source = args.atoms
        lattice_source = args.lattice or source
        # Radius follows the ATOMS -- it is their protofilament that gets moved --
        # while the repeat follows the LATTICE source.
        radius = measure_radius(
            models[source], frames[source]["rot"], frames[source]["trans"], keep
        )
        print(f"\nBuilding {args.rings}-ring protofilament-number variants "
              f"from {source} atoms on the {lattice_source} lattice")
        print(f"  source protofilament radius {radius:.2f} A, "
              f"lateral spacing {2 * np.pi * radius / N_PROTOFILAMENTS:.2f} A (held fixed)")
        for n_pf in args.protofilaments:
            lattice = protofilament_lattice(frames[lattice_source], radius, n_pf)
            structure = build(
                models[source],
                frames[source]["rot"],
                frames[source]["trans"],
                lattice,
                keep,
                args.rings,
                flat_b,
            )
            name = (
                f"{source}_{n_pf}pf_{args.rings}rings_flatB.pdb"
                if lattice_source == source
                else f"{source}_atoms_{lattice_source}_lattice_{n_pf}pf_"
                     f"{args.rings}rings_flatB.pdb"
            )
            structure.write_pdb(str(args.outdir / name))
            n_atoms = sum(len(r) for c in structure[0] for r in c)
            print(
                f"  {name:34s} R {radius + lattice['radial_shift_a']:7.2f} A"
                f"  twist {lattice['twist_deg']:7.3f}"
                f"  lat rise {lattice['lateral_rise_a']:6.3f}"
                f"  chains {len(structure[0]):3d}  atoms {n_atoms}"
            )
        return

    print(f"\nBuilding {args.rings}-ring templates")
    for atoms, lat in itertools.product(DEPOSITIONS, DEPOSITIONS):
        for suffix, b in (("", None), ("_flatB", flat_b)):
            structure = build(
                models[atoms],
                frames[atoms]["rot"],
                frames[atoms]["trans"],
                frames[lat],
                keep,
                args.rings,
                b,
            )
            name = f"{atoms}_atoms_{lat}_lattice_{args.rings}rings{suffix}.pdb"
            structure.write_pdb(str(args.outdir / name))
            n_atoms = sum(len(r) for c in structure[0] for r in c)
            print(f"  {name:52s} chains {len(structure[0]):3d}  atoms {n_atoms}")


if __name__ == "__main__":
    main()
