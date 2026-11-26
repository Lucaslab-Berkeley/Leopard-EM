"""Decorate microtubule templates with nucleotide-free kinesin-1 (KIF5B) from 3J8X.

3J8X (Shang et al. 2014, eLife; 5.0 A cryo-EM) is the KIF5B motor domain (residues
1-349, chain K; 5-320 modelled -- the undocked neck linker is not) bound to one
alpha/beta tubulin dimer (chains A, B) of a microtubule, with no nucleotide: rigor.
Its tubulin is the straight lattice tubulin, so it places a motor on a lattice model
without the curved-tubulin error a crystal complex would bring.

For every beta-tubulin of a lattice template (beta found by SEQUENCE, so a register-
shifted patch gets its motors on its real beta chains, not on beta positions) the
complex's beta is superposed onto it (Kabsch on matched C-alpha) and the motor carried
across. Checks printed: the beta fit, how well the complex's alpha then lands on the
lattice's intradimer alpha, and clashes between motors and with other tubulin.

Three outputs per lattice template ``<stem>``:

* ``<stem>_kin3j8x.pdb``             the decorated template: every beta carries a motor
                                     (saturated);
* ``<stem>_kin3j8x_motorsonly.pdb``  the motors alone, at their lattice positions --
                                     scored at known tube poses its z is linear in
                                     occupancy;
* ``<stem>_kin3j8x_motors/motor_NN.pdb`` (``--split``) one file per motor, for building
                                     synthetic tubes of chosen occupancy;
* ``<stem>_kin3j8x[_motorsonly]_shift.pdb`` (``--register-shift``) the same motors moved one
                                     monomer along the axis (half the beta-to-beta repeat)
                                     -- onto the ALPHA positions, the other register. The
                                     mean of the two motors-only maps is motors at half
                                     weight on every monomer: its lattice repeats every
                                     monomer, so it has no seam and no register, and scores
                                     decoration without knowing either.

Coordinates are never re-centred: the tube axis stays on the box centre, so a pose found
with any template is valid for all of them. Motor atoms get B = the lattice's flat B +
``--motor-extra-b`` (kinesin density is always weaker than tubulin's: mobility relative
to the lattice and partial occupancy). Each motor is appended to its host beta chain as
residues 1005-1320, because 56 tubulin chains + 28 motors exceed the PDB format's 62
chain IDs and the simulator reads only coordinates, elements and B.

Usage:
    python models/build_kinesin_templates.py --split \\
        6dpu_14pf_2rings_flatB 6dpu_atoms_6dpv_lattice_14pf_2rings_flatB \\
        6dpu_14pf_patch5_2rings_flatB 6dpu_14pf_patch5_2rings_flatB_shift1
"""

import argparse
import pathlib
import urllib.request

import gemmi
import numpy as np
from scipy.spatial import cKDTree

MODEL_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_ID = "3j8x"
MOTOR_CHAIN, ALPHA_CHAIN, BETA_CHAIN = "K", "A", "B"
RESIDUE_OFFSET = 1000      # motor residue n is written as n + 1000 in its host chain
CLASH_A = 2.5              # heavy-atom distance counted as a clash


def fetch(pdb_id: str) -> pathlib.Path:
    """The deposition's mmCIF, downloaded once into the models directory."""
    path = MODEL_DIR / f"{pdb_id}.cif"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        print(f"  fetching {url}")
        urllib.request.urlretrieve(url, path)
    return path


def calpha(chain: gemmi.Chain) -> dict[tuple[int, str], np.ndarray]:
    """C-alpha positions keyed by (residue number, residue name)."""
    out = {}
    for residue in chain:
        atom = residue.find_atom("CA", "*")
        if atom is not None:
            out[(residue.seqid.num, residue.name)] = np.array(atom.pos.tolist())
    return out


def kabsch(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |R @ moving + t - fixed|."""
    mc, fc = moving.mean(axis=0), fixed.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (fixed - fc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, fc - rotation @ mc


def matched(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """C-alpha coordinates of the residues two chains share, in the same order."""
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def identity(chain_ca: dict, reference_ca: dict) -> float:
    """Fraction of the chain's residue numbers carrying the reference's residue name."""
    numbers = {num: name for num, name in reference_ca}
    hits = sum(numbers.get(num) == name for num, name in chain_ca)
    return hits / max(len(chain_ca), 1)


def motor_atoms(source: gemmi.Model) -> list[tuple[gemmi.Residue, np.ndarray]]:
    """The motor's residues with their heavy-atom coordinates."""
    out = []
    for residue in source[MOTOR_CHAIN].get_polymer():
        out.append((residue, np.array([a.pos.tolist() for a in residue
                                       if a.element.name != "H"])))
    return out


def dimer_repeat(model: gemmi.Model, betas: list[str]) -> float:
    """The beta-to-beta repeat along z: consecutive betas of one protofilament column."""
    columns: dict[tuple[float, float], list[float]] = {}
    for name in betas:
        xyz = np.array([a.pos.tolist() for r in model[name] for a in r])
        centre = xyz.mean(axis=0)
        key = (round(centre[0] / 5.0), round(centre[1] / 5.0))
        columns.setdefault(key, []).append(centre[2])
    steps = [np.diff(sorted(z)) for z in columns.values() if len(z) > 1]
    return float(np.median(np.concatenate(steps)))


def decorate(stem: str, source: gemmi.Model, extra_b: float, split: bool,
             register_shift: bool = False) -> None:
    """Write the decorated, motors-only and (optionally) per-motor PDBs for one lattice."""
    lattice = gemmi.read_structure(str(MODEL_DIR / f"{stem}.pdb"))
    lattice.setup_entities()
    model = lattice[0]
    alpha_ref, beta_ref = calpha(source[ALPHA_CHAIN]), calpha(source[BETA_CHAIN])
    flat_b = float(np.mean([a.b_iso for c in model for r in c for a in r]))
    motor_b = flat_b + extra_b

    chains_ca = {c.name: calpha(c) for c in model}
    betas = [n for n, ca in chains_ca.items() if identity(ca, beta_ref) > identity(ca, alpha_ref)]
    alphas = [n for n in chains_ca if n not in betas]
    motor = motor_atoms(source)
    source_alpha = np.array(list(alpha_ref.values()))

    placed = []      # (host chain, rotation, translation)
    report = []
    for name in betas:
        moving, fixed = matched(beta_ref, chains_ca[name])
        rotation, shift = kabsch(moving, fixed)
        rmsd = float(np.sqrt(np.mean(np.sum((moving @ rotation.T + shift - fixed) ** 2, 1))))
        # where the complex's alpha lands: on the lattice's intradimer alpha if the
        # two lattices agree; report the best-matching lattice alpha's RMSD
        moved_alpha = {k: v @ rotation.T + shift for k, v in alpha_ref.items()}
        alpha_rmsd = min(
            float(np.sqrt(np.mean(np.sum(np.subtract(*matched(moved_alpha, chains_ca[a])) ** 2, 1))))
            for a in alphas)
        placed.append((name, rotation, shift))
        report.append((name, rmsd, alpha_rmsd, len(moving)))
    del source_alpha

    # clashes: each motor against the other motors and against tubulin other than the
    # dimer it sits on (its host beta and the alpha its complex's alpha matched)
    motor_xyz = [np.vstack([xyz for _, xyz in motor]) @ r.T + t for _, r, t in placed]
    tubulin = {c.name: np.array([a.pos.tolist() for r in c for a in r
                                 if a.element.name != "H"]) for c in model}
    worst_motor, worst_tubulin, clashes = np.inf, np.inf, 0
    for i, xyz in enumerate(motor_xyz):
        tree = cKDTree(xyz)
        for j, other in enumerate(motor_xyz):
            if j > i:
                d = tree.query(other)[0]
                worst_motor = min(worst_motor, float(d.min()))
                clashes += int((d < CLASH_A).sum())
        host = placed[i][0]
        for chain, xyz_t in tubulin.items():
            d = tree.query(xyz_t)[0]
            if chain == host or d.min() > 6.0:
                continue
            # tubulin chains the motor genuinely touches (its dimer's alpha) are fine;
            # count only atoms that interpenetrate
            worst_tubulin = min(worst_tubulin, float(d.min()))

    print(f"{stem}: {len(betas)} beta chains (by sequence), {len(alphas)} alpha; "
          f"motor B {motor_b:.1f} (lattice {flat_b:.2f} + {extra_b:g})")
    fits = np.array([[r, a] for _, r, a, _ in report])
    print(f"  beta fit RMSD {fits[:, 0].min():.2f}-{fits[:, 0].max():.2f} A over "
          f"{report[0][3]} C-alpha; complex alpha on lattice alpha "
          f"{fits[:, 1].min():.2f}-{fits[:, 1].max():.2f} A")
    print(f"  closest motor-motor {worst_motor:.1f} A ({clashes} atom pairs < {CLASH_A} A); "
          f"closest motor-to-other-tubulin {worst_tubulin:.1f} A")

    def write(path: pathlib.Path, with_tubulin: bool, which: list[int],
              placements: list | None = None) -> None:
        placements = placed if placements is None else placements
        structure = gemmi.Structure()
        structure.cell = lattice.cell
        out_model = gemmi.Model("1")
        hosts = {placements[i][0]: i for i in which}
        for chain in model:
            if with_tubulin:
                out_chain = chain.clone()
            elif chain.name in hosts:
                out_chain = gemmi.Chain(chain.name)
            else:
                continue
            if chain.name in hosts:
                _, rotation, shift = placements[hosts[chain.name]]
                for residue, _ in motor:
                    new = gemmi.Residue()
                    new.name = residue.name
                    new.seqid = gemmi.SeqId(residue.seqid.num + RESIDUE_OFFSET, " ")
                    new.het_flag = "A"
                    for atom in residue:
                        if atom.element.name == "H":
                            continue
                        moved = atom.clone()
                        xyz = rotation @ np.array(atom.pos.tolist()) + shift
                        moved.pos = gemmi.Position(*xyz)
                        moved.b_iso = motor_b
                        moved.occ = 1.0
                        new.add_atom(moved)
                    out_chain.add_residue(new)
            out_model.add_chain(out_chain)
        structure.add_model(out_model)
        structure.setup_entities()
        structure.write_pdb(str(path))

    everything = list(range(len(placed)))
    write(MODEL_DIR / f"{stem}_kin3j8x.pdb", True, everything)
    write(MODEL_DIR / f"{stem}_kin3j8x_motorsonly.pdb", False, everything)
    print(f"  wrote {stem}_kin3j8x.pdb, {stem}_kin3j8x_motorsonly.pdb")
    if register_shift:
        half = 0.5 * dimer_repeat(model, betas)
        moved = [(name, rotation, shift + np.array([0.0, 0.0, half]))
                 for name, rotation, shift in placed]
        write(MODEL_DIR / f"{stem}_kin3j8x_shift.pdb", True, everything, moved)
        write(MODEL_DIR / f"{stem}_kin3j8x_motorsonly_shift.pdb", False, everything, moved)
        print(f"  wrote {stem}_kin3j8x[_motorsonly]_shift.pdb (motors +{half:.3f} A along "
              f"the axis: the alpha positions)")
    if split:
        directory = MODEL_DIR / f"{stem}_kin3j8x_motors"
        directory.mkdir(exist_ok=True)
        for i in everything:
            write(directory / f"motor_{i:02d}.pdb", False, [i])
        print(f"  wrote {len(everything)} single-motor PDBs in {directory.name}/")


def main() -> None:
    """Decorate each named lattice template."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stems", nargs="+", help="lattice templates in models/ (no .pdb)")
    parser.add_argument("--motor-extra-b", type=float, default=30.0)
    parser.add_argument("--split", action="store_true",
                        help="also write one PDB per motor (for synthetics)")
    parser.add_argument("--register-shift", action="store_true",
                        help="also write the motors moved one monomer (alpha positions)")
    args = parser.parse_args()

    source = gemmi.read_structure(str(fetch(SOURCE_ID)))
    source.setup_entities()
    source.remove_hydrogens()
    for stem in args.stems:
        decorate(stem, source[0], args.motor_extra_b, args.split, args.register_shift)


if __name__ == "__main__":
    main()
