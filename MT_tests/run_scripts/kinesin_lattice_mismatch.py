"""Does a wrong lattice spacing inside the template cost the kinesin readout? Synthetic.

The templates are the EXPANDED 14-PF ring (83.96 A per dimer) at 1.13 A/px. Here the
SPECIMEN tubes are built differently and scored with those templates at the right site
positions (block centres -- on data the sites come from a lattice fitted to the data, so
spacing errors do not accumulate along the tube; only the template's internal spacing is
wrong):

* ``expanded``         the template's own lattice (the reference)
* ``compacted``        the 6DPV lattice, 81.67 A per dimer (2.7% shorter), same atoms
* ``scale -1.2%`` / ``scale +1.2%``  the expanded tube seen at 1.117 / 1.143 A/px: a
                       pixel-size error of the size the ring fit suggested (1.117 vs 1.13)

Same optics, noise and scale as the occupancy synthetic; bare (p = 0) and fully
decorated (p = 1) tubes; motor contrast and tube length to tell 50% from bare, as in
``kinesin_model_errors.py``.

Usage:
    python run_scripts/kinesin_lattice_mismatch.py
"""

import json
import pathlib
import sys

import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from simulate_curved_microtubule import phase_randomised  # noqa: E402
from simulate_kinesin_occupancy import (  # noqa: E402
    BOX,
    END_UNITS,
    LATTICE,
    MT_ROOT,
    NEIGHBORHOOD_B,
    PIXEL,
    TAG,
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
COMPACTED = "6dpu_atoms_6dpv_lattice_14pf_2rings_flatB"
REPEAT = {"expanded": 2 * 83.9582, "compacted": 2 * 81.6709}


def load(stem):
    with mrcfile.open(str(MT_ROOT / "maps" / f"{stem}{TAG}.mrc"), permissive=True) as handle:
        return torch.tensor(np.asarray(handle.data, dtype=np.float32))


def simulate_at(stem, pixel, device):
    """The expanded model at another pixel spacing (fixed neighbourhood, as every map)."""
    simulator = Simulator(pdb_filepath=str(MT_ROOT / "models" / f"{stem}.pdb"),
                          pixel_spacing=pixel, volume_shape=(BOX,) * 3, center_atoms=False,
                          remove_hydrogens=True, b_factor_scaling=B_FACTOR_SCALING,
                          additional_b_factor=0,
                          simulator_config=sim_config(200.0, "k3_200kV_CDS", 50.0, True))
    return simulator.run(device=device).float().cpu()


def main() -> None:
    device = "cuda:3"
    synthetic = MT_ROOT / "results_kinesin_synthetic"
    fix_neighborhood(NEIGHBORHOOD_B)
    truth = json.loads((synthetic / "truth.json").read_text())
    tubes = truth["tubes"]
    poses = [(t["phi"], t["psi"]) for t in tubes]
    stored = np.load(synthetic / f"projections_seed{truth['seed']}.npz")
    templates = {name: stored[f"t_{name}"] for name in SCORED}
    optics = yaml.safe_load((synthetic / "optics.yaml").read_text())["optics_group"]
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

    # specimen blocks: (bare projection, motors projection, block step in px) per variant
    variants = {}
    bare, decorated = load(LATTICE), load(f"{LATTICE}_kin3j8x")
    pb, pd_ = project(bare, poses, device), project(decorated, poses, device)
    variants["expanded"] = (pb, pd_ - pb, REPEAT["expanded"] / PIXEL)
    bare_c, decorated_c = load(COMPACTED), load(f"{COMPACTED}_kin3j8x")
    pb, pd_ = project(bare_c, poses, device), project(decorated_c, poses, device)
    variants["compacted (2.7% shorter)"] = (pb, pd_ - pb, REPEAT["compacted"] / PIXEL)
    for label, pixel in (("scale -1.2% (1.117 A/px)", 1.117), ("scale +1.2% (1.143 A/px)", 1.143)):
        pb = project(simulate_at(LATTICE, pixel, device), poses, device)
        pd_ = project(simulate_at(f"{LATTICE}_kin3j8x", pixel, device), poses, device)
        variants[label] = (pb, pd_ - pb, REPEAT["expanded"] / pixel)

    def units_for(tube, step):
        """Block centres along the tube at this lattice's step, about its middle block."""
        units = np.array(tube["units"])
        centre = units[len(units) // 2]
        angle = np.radians(tube["angle_deg"])
        direction = np.array([np.sin(angle), np.cos(angle)])
        n = int((shape[1] / 2 - BOX / 2 - 20) / abs(np.cos(angle)) // step)
        return [centre + k * step * direction for k in range(-n, n + 1)]

    def score(bare_proj, motor_proj, step, occupied):
        canvas = np.zeros(shape)
        sites = []
        for t, tube in enumerate(tubes):
            units = units_for(tube, step)
            for centre in units:
                paste(canvas, bare_proj[t] + (motor_proj[t] if occupied else 0.0), centre)
            sites.append([tuple(int(round(c)) for c in u)
                          for u in units[END_UNITS:len(units) - END_UNITS]])
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

    rows, ideal = [], None
    for label, (bare_proj, motor_proj, step) in variants.items():
        z0 = score(bare_proj, motor_proj, step, False)
        z1 = score(bare_proj, motor_proj, step, True)
        row = {"specimen": label, "bare z on bare tube": z0["bare"].mean()}
        for name, column in (("motors only", "motors"), ("both registers", "motors_avg"),
                             ("decorated - bare", "dz")):
            slope = z1[column].mean() - z0[column].mean()
            sd = np.sqrt(0.5 * (z0[column].var(ddof=1) + z1[column].var(ddof=1)))
            if ideal is None or column not in ideal:
                ideal = ideal or {}
                ideal.setdefault(column, slope)
            occ_sd = sd / slope
            row[f"{name}: contrast kept"] = slope / ideal[column]
            row[f"{name}: nm for 50%"] = 2 * (3 * occ_sd / 0.5) ** 2 * BLOCK_NM
        rows.append(row)
        print(f"  {label}: done", flush=True)
    table = pd.DataFrame(rows)
    out = synthetic / "model_errors" / "lattice_mismatch.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
