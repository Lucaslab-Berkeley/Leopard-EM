"""How kinesin pose errors and modelling errors cost the occupancy readout -- synthetic.

The occupancy synthetic of ``simulate_kinesin_occupancy.py`` (same four tubes, optics,
noise and scale), but the SPECIMEN's motors are no longer exactly the template's: the
templates stay the ideal 3J8X placement, as on real data, and the specimen is perturbed.

Perturbations of each motor (rigid, about its pivot -- the centroid of its atoms within
5 A of tubulin, i.e. where it holds on):

* ``rotation``      random axis, fixed angle (deg)            -- heterogeneous heads
* ``tilt``          about the tube's tangent at the motor, +angle, the same for every
                    motor (deg)                                 -- a different bound state
* ``shift``         random direction, fixed distance (A)       -- loose seating
* ``axial``         +distance along the tube axis, every motor (A)
* ``atoms``         every motor atom displaced at random, given 3-D RMS (A) -- model error
* ``atoms_all``     the same for every atom, tubulin included  -- model error everywhere

Random kinds use ``--sets`` independent perturbed sets, assigned at random block by
block, so neighbouring sites do not share one error.

Each condition is scored as the occupancy curve is: z per 16.8 nm block, from ideal
templates (bare, decorated, motors, motors on both registers) at the true pose, on the
bare tube (p = 0) and the fully decorated one (p = 1). The block-to-block scatter barely
moves with occupancy (1.00-1.01 on the original synthetic), so slope = mean z(1) - mean
z(0) and the pooled SD give the occupancy SD per block, and the tube length to tell
occupancy p from bare at 3 SD, 2 (3 sigma / p)^2 x 16.8 nm, as on the plot.

Writes ``results_kinesin_synthetic/model_errors/``: ``model_errors.csv`` and a figure.

Usage:
    python run_scripts/kinesin_model_errors.py
"""

import argparse
import json
import pathlib
import sys

import gemmi
import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from simulate_curved_microtubule import phase_randomised  # noqa: E402
from simulate_kinesin_occupancy import (  # noqa: E402
    BOX,
    END_UNITS,
    LATTICE,
    MT_ROOT,
    NEIGHBORHOOD_B,
    PIXEL,
    correlate,
    ctf_filter,
    paste,
    project,
    template_rfft,
)
from simulate_lattice_competition import (  # noqa: E402
    B_FACTOR_SCALING,
    fix_neighborhood,
    sim_config,
)
from ttsim3d.models import Simulator  # noqa: E402

from leopard_em.pydantic_models.config.correlation_filters import (  # noqa: E402
    WhiteningFilterConfig,
)

BLOCK_NM = 16.8
SCORED = ["bare", "decorated", "motors", "motors_avg"]
CONDITIONS = {                       # kind: magnitudes
    "rotation": (2, 5, 10, 15, 20),
    "tilt": (2, 5, 10, 15, 20),
    "shift": (1, 2, 4, 6, 10),
    "axial": (1, 2, 4, 6, 10),
    "atoms": (0.5, 1, 2, 3, 5),
    "atoms_all": (0.5, 1, 2, 3, 5),
}
RANDOM_KINDS = {"rotation", "shift", "atoms", "atoms_all"}
UNITS = {"rotation": "deg", "tilt": "deg", "shift": "A", "axial": "A", "atoms": "A RMS",
         "atoms_all": "A RMS"}


def rotation_matrix(axis, angle_deg):
    """Rodrigues rotation about a unit axis."""
    axis = axis / np.linalg.norm(axis)
    a = np.radians(angle_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def read_atoms(path):
    """A structure and its atom coordinates (N, 3)."""
    structure = gemmi.read_structure(str(path))
    xyz = np.array([a.pos.tolist() for c in structure[0] for r in c for a in r])
    return structure, xyz


def write_atoms(structure, xyz, path):
    """The structure with new coordinates."""
    i = 0
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                atom.pos = gemmi.Position(*xyz[i])
                i += 1
    structure.write_pdb(str(path))


def simulate(path, device):
    """One volume, simulated exactly as every template of the kinesin set."""
    simulator = Simulator(pdb_filepath=str(path), pixel_spacing=PIXEL, volume_shape=(BOX,) * 3,
                          center_atoms=False, remove_hydrogens=True,
                          b_factor_scaling=B_FACTOR_SCALING, additional_b_factor=0,
                          simulator_config=sim_config(200.0, "k3_200kV_CDS", 50.0, True))
    return simulator.run(device=device).float().cpu()


def perturbed_sets(kind, magnitude, motors, pivots, tubulin_xyz, rng, n_sets):
    """``n_sets`` lists of perturbed motor coordinates (and tubulin, for atoms_all)."""
    out = []
    for _ in range(n_sets if kind in RANDOM_KINDS else 1):
        moved, tubulin = [], None
        for xyz, pivot in zip(motors, pivots):
            if kind == "rotation":
                axis = rng.normal(size=3)
                new = (xyz - pivot) @ rotation_matrix(axis, magnitude).T + pivot
            elif kind == "tilt":
                radial = np.array([pivot[0], pivot[1], 0.0])
                tangent = np.cross([0.0, 0.0, 1.0], radial / np.linalg.norm(radial))
                new = (xyz - pivot) @ rotation_matrix(tangent, magnitude).T + pivot
            elif kind == "shift":
                direction = rng.normal(size=3)
                new = xyz + magnitude * direction / np.linalg.norm(direction)
            elif kind == "axial":
                new = xyz + np.array([0.0, 0.0, magnitude])
            else:                               # atoms, atoms_all: 3-D RMS = magnitude
                new = xyz + rng.normal(scale=magnitude / np.sqrt(3), size=xyz.shape)
            moved.append(new)
        if kind == "atoms_all":
            tubulin = tubulin_xyz + rng.normal(scale=magnitude / np.sqrt(3),
                                               size=tubulin_xyz.shape)
        out.append((moved, tubulin))
    return out


def main() -> None:
    """Score bare and decorated synthetic tubes whose motors differ from the template."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=pathlib.Path,
                        default=MT_ROOT / "results_kinesin_synthetic")
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    outdir = args.synthetic / "model_errors"
    work = outdir / "pdb"
    work.mkdir(parents=True, exist_ok=True)
    fix_neighborhood(NEIGHBORHOOD_B)
    rng = np.random.default_rng(5)

    truth = json.loads((args.synthetic / "truth.json").read_text())
    tubes = truth["tubes"]
    poses = [(t["phi"], t["psi"]) for t in tubes]
    stored = np.load(args.synthetic / f"projections_seed{truth['seed']}.npz")
    tubulin_ideal, motors_ideal = stored["tubulin"], stored["motors"]
    templates = {name: stored[f"t_{name}"] for name in SCORED}
    optics = yaml.safe_load((args.synthetic / "optics.yaml").read_text())["optics_group"]
    with mrcfile.open(str(MT_ROOT / "results_kinesin" / "20260916_lysates_2_0031_DW.mrc"),
                      permissive=True) as handle:
        source = np.asarray(handle.data, dtype=np.float64)
    shape = source.shape
    noise = phase_randomised(source - source.mean(), truth["seed"])
    filt = ctf_filter(shape, optics, device)
    noise_dft = torch.fft.rfftn(torch.tensor(noise, dtype=torch.float32))
    noise_dft[0, 0] = 0
    white = WhiteningFilterConfig(enabled=True, max_freq=1.0, do_power_spectrum=True) \
        .calculate_whitening_filter(noise_dft).to(device)
    tf = {(n, t): template_rfft(templates[n][t], shape, filt * white, device)
          for n in SCORED for t in range(len(tubes))}
    noise_w = torch.fft.rfft2(torch.tensor(noise, dtype=torch.float32, device=device)) * white
    inner = (slice(BOX, shape[0] - BOX), slice(BOX, shape[1] - BOX))
    noise_cc, noise_sd = {}, {}
    for key, t in tf.items():
        cc = correlate(noise_w, t, shape)
        noise_cc[key], noise_sd[key] = cc.cpu().numpy(), float(cc[inner].std())
    sites = [[tuple(int(round(c)) for c in u)
              for u in t["units"][END_UNITS:len(t["units"]) - END_UNITS]] for t in tubes]

    def score(blocks_by_tube):
        """Mean and SD of z per template over all scored blocks, for one canvas."""
        canvas = np.zeros(shape)
        for t, tube in enumerate(tubes):
            for centre, block in zip(tube["units"], blocks_by_tube[t]):
                paste(canvas, block, centre)
        signal = torch.fft.irfft2(torch.fft.rfft2(torch.tensor(canvas, dtype=torch.float32,
                                                               device=device)) * filt, s=shape)
        signal_w = torch.fft.rfft2(signal) * white
        z = {n: [] for n in SCORED}
        for (n, t), template in tf.items():
            cc = correlate(signal_w, template, shape).cpu().numpy()
            for y, x in sites[t]:
                z[n].append((truth["scale"] * cc[y, x] + noise_cc[(n, t)][y, x])
                            / noise_sd[(n, t)])
        z = {n: np.array(v) for n, v in z.items()}
        z["dz"] = z["decorated"] - z["bare"]
        return z

    def tube_blocks(tubulin_proj, motor_proj_sets, occupied):
        """Per tube, per block: tubulin plus (if occupied) one randomly chosen motor set."""
        out = []
        for t, tube in enumerate(tubes):
            blocks = []
            for _ in tube["units"]:
                block = tubulin_proj[t].copy()
                if occupied:
                    block = block + motor_proj_sets[rng.integers(len(motor_proj_sets))][t]
                blocks.append(block)
            out.append(blocks)
        return out

    # the ideal: cached projections, all 28 motors summed per tube
    ideal_motors = [motors_ideal.sum(axis=1)]
    z0_ideal = score(tube_blocks(tubulin_ideal, None, False))
    z1_ideal = score(tube_blocks(tubulin_ideal, ideal_motors, True))

    # per-motor atoms and pivots (motor atoms within 5 A of tubulin)
    motor_dir = MT_ROOT / "models" / f"{LATTICE}_kin3j8x_motors"
    motor_files = sorted(motor_dir.glob("motor_*.pdb"))
    structures, motors = zip(*(read_atoms(p) for p in motor_files))
    ring_structure, ring_xyz = read_atoms(MT_ROOT / "models" / f"{LATTICE}.pdb")
    ring_tree = cKDTree(ring_xyz)
    pivots = [xyz[ring_tree.query(xyz)[0] < 5.0].mean(axis=0) for xyz in motors]
    template_set = gemmi.read_structure(str(motor_files[0]))

    rows = [{"kind": "ideal", "magnitude": 0.0, "z0": z0_ideal, "z1": z1_ideal}]
    for kind, magnitudes in CONDITIONS.items():
        for magnitude in magnitudes:
            sets = perturbed_sets(kind, magnitude, motors, pivots, ring_xyz, rng, args.sets)
            motor_proj_sets, tubulin_proj = [], tubulin_ideal
            tubulin_projs = []
            for k, (moved, tubulin) in enumerate(sets):
                # one PDB with all 28 perturbed motors (distinct chain IDs per motor)
                joined = gemmi.Structure()
                model = gemmi.Model("1")
                for structure, xyz in zip(structures, moved):
                    single = structure.clone()
                    i = 0
                    for chain in single[0]:
                        for residue in chain:
                            for atom in residue:
                                atom.pos = gemmi.Position(*xyz[i])
                                i += 1
                        model.add_chain(chain)
                joined.add_model(model)
                joined.setup_entities()
                path = work / f"{kind}_{magnitude:g}_{k}.pdb"
                joined.write_pdb(str(path))
                motor_proj_sets.append(project(simulate(path, device), poses, device))
                if tubulin is not None:
                    tpath = work / f"{kind}_{magnitude:g}_{k}_tubulin.pdb"
                    write_atoms(ring_structure, tubulin, tpath)
                    tubulin_projs.append(project(simulate(tpath, device), poses, device))
            if tubulin_projs:
                # atoms_all: the bare tube is perturbed too -- one tubulin set per block
                tubulin_proj = tubulin_projs[0]
            z0 = score(tube_blocks(tubulin_proj, None, False))
            z1 = score(tube_blocks(tubulin_proj, motor_proj_sets, True))
            rows.append({"kind": kind, "magnitude": magnitude, "z0": z0, "z1": z1})
            print(f"  {kind:<9} {magnitude:>4g} {UNITS[kind]:<5}  motors contrast "
                  f"{z1['motors'].mean() - z0['motors'].mean():+.2f} (ideal "
                  f"{z1_ideal['motors'].mean() - z0_ideal['motors'].mean():+.2f}), both "
                  f"registers {z1['motors_avg'].mean() - z0['motors_avg'].mean():+.2f}, bare "
                  f"z {z0['bare'].mean():.2f}", flush=True)
    del template_set

    table = []
    for row in rows:
        out = {"kind": row["kind"], "magnitude": row["magnitude"],
               "unit": UNITS.get(row["kind"], ""), "z_bare_on_bare": row["z0"]["bare"].mean()}
        for name, column in (("decorated - bare", "dz"), ("motors only", "motors"),
                             ("both registers", "motors_avg")):
            slope = row["z1"][column].mean() - row["z0"][column].mean()
            ideal = z1_ideal[column].mean() - z0_ideal[column].mean()
            sd = np.sqrt(0.5 * (row["z0"][column].var(ddof=1) + row["z1"][column].var(ddof=1)))
            occ_sd = sd / slope if slope > 0 else np.inf
            out[f"{name}: contrast kept"] = slope / ideal
            out[f"{name}: occupancy SD/block"] = occ_sd
            for p in (0.25, 0.5):
                out[f"{name}: nm for p={p}"] = 2 * (3 * occ_sd / p) ** 2 * BLOCK_NM
        table.append(out)
    table = pd.DataFrame(table)
    table.to_csv(outdir / "model_errors.csv", index=False)
    show = ["kind", "magnitude", "unit", "z_bare_on_bare", "motors only: contrast kept",
            "both registers: contrast kept", "motors only: nm for p=0.5",
            "both registers: nm for p=0.5"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    plot(table, outdir / "model_errors.png")
    print(f"wrote {outdir}/model_errors.csv and model_errors.png")


def plot(table, out_path):
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    ideal = table[table.kind == "ideal"].iloc[0]
    groups = [("rotation", "tilt", "kinesin rotation (deg)"),
              ("shift", "axial", "kinesin shift (A)"),
              ("atoms", "atoms_all", "atomic model error (A RMS)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    styles = {"motors only": "#2e7d32", "both registers": "#6a1b9a", "decorated - bare": "#c62828"}
    for col, (random_kind, other_kind, label) in enumerate(groups):
        for kind, dash in ((random_kind, "-"), (other_kind, "--")):
            part = pd.concat([ideal.to_frame().T, table[table.kind == kind]])
            x = part.magnitude.astype(float)
            for name, colour in styles.items():
                axes[0, col].plot(x, part[f"{name}: contrast kept"].astype(float), dash,
                                  marker="o", ms=4, color=colour,
                                  label=f"{name}, {kind}")
                axes[1, col].plot(x, part[f"{name}: nm for p=0.5"].astype(float), dash,
                                  marker="o", ms=4, color=colour, label=f"{name}, {kind}")
        axes[0, col].axhline(0, color="grey", lw=0.8)
        axes[0, col].set_ylabel("motor contrast kept (vs ideal)")
        axes[0, col].set_xlabel(label)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylim(30, 3e4)
        axes[1, col].axhline(500, color="#ef6c00", lw=1, ls=":")
        axes[1, col].set_ylabel("tube (nm) to tell 50% from bare, 3 SD")
        axes[1, col].set_xlabel(label)
        axes[0, col].legend(fontsize=7)
    fig.suptitle("Specimen motors differ from the 3J8X template: contrast kept and tube needed "
                 "(solid: random per motor; dashed: systematic)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    main()
