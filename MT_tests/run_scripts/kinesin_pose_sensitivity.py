"""Could pose errors or a wrong polarity hide kinesin from the templates? On the synthetic.

kinesin_2_0031's tubes score no 3J8X-like motor (``score_kinesin_sites.py``) while their
8 nm layer line says ~75% occupancy. Two pose explanations are tested here, on the
synthetic of ``simulate_kinesin_occupancy.py`` (same tubes, optics, noise and scale),
where the truth is known: a bare tube (p = 0) and a saturated one (p = 1).

At each perturbed pose every template is scored at every site, and two things are
compared with the TRUE pose:

* the bare ring's z (what sets the neutral poses on data), and
* the decoration contrast of each motors template, z(p = 1) - z(p = 0).

If a pose error that leaves the bare ring nearly untouched removes most of the motor
contrast, pose errors could hide kinesin; if the motor contrast survives as well as the
bare ring does, they cannot. ``read occ`` is the occupancy a SATURATED tube would read at
that pose through the true-pose calibration (the real-data recipe: ratio of motors-avg z
to bare z).

Perturbations (the neutral poses on data are built from the bare ring, so wherever a pose
is re-fitted it is re-fitted by the bare ring alone, per tube, as the data were):

1. position: the template off the true site, along or across the axis;
2. roll: true roll + d, the axial position then re-fitted (+/- half a monomer) by the
   bare ring;
3. seam: roll moved by k whole protofilament steps (360/14), roll fine-tuned +/-3 deg and
   axial position re-fitted by the bare ring -- the lattice lands on itself, the seam
   does not: the bare ring cannot tell these poses apart, register-specific motors can;
4. polarity: psi + 180, roll searched over 360 deg and axial position re-fitted by the
   bare ring, then the motors scored there (both registers).

Writes ``results_kinesin_synthetic/pose_sensitivity.csv`` and prints the tables.

Usage:
    python run_scripts/kinesin_pose_sensitivity.py
"""

import argparse
import json
import pathlib
import sys

import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_kinesin_sites import occupancy_from_ratio  # noqa: E402
from simulate_curved_microtubule import phase_randomised  # noqa: E402
from simulate_kinesin_occupancy import (  # noqa: E402
    BOX,
    END_UNITS,
    MT_ROOT,
    PIXEL,
    TEMPLATES,
    ctf_filter,
    load_map,
    paste,
)

from leopard_em.pydantic_models.config.correlation_filters import (  # noqa: E402
    WhiteningFilterConfig,
)
from leopard_em.utils.fourier_slice import (  # noqa: E402
    get_real_space_projections_from_volume,
)

PF_STEP = 360.0 / 14
MONOMER_A = 83.9582 / 2
SCORED = ["bare", "motors", "motors_shift", "motors_avg"]


class Scorer:
    """Correlate any template at any pose against a fixed whitened image."""

    def __init__(self, image, shape, filt, white, device):
        self.shape, self.filt, self.white, self.device = shape, filt, white, device
        self.image_w = torch.fft.rfft2(torch.tensor(image, dtype=torch.float32,
                                                    device=device)) * white
        self.volumes = {name: load_map(TEMPLATES[name]).to(device) for name in SCORED}

    def cc_map(self, name, phi, psi):
        """Correlation map of template ``name`` at (phi, 90, psi), unit-norm template."""
        with torch.no_grad():
            angles = [torch.tensor([float(a)], dtype=torch.float32, device=self.device)
                      for a in (phi, 90.0, psi)]
            proj = get_real_space_projections_from_volume(self.volumes[name], *angles)[0]
            padded = torch.zeros(self.shape, device=self.device)
            padded[:BOX, :BOX] = proj
            padded = torch.roll(padded, shifts=(-(BOX // 2), -(BOX // 2)), dims=(0, 1))
            spectrum = torch.fft.rfft2(padded) * self.filt * self.white
            spectrum[0, 0] = 0
            spectrum = spectrum / torch.linalg.norm(torch.fft.irfft2(spectrum, s=self.shape))
            return torch.fft.irfft2(self.image_w * torch.conj(spectrum), s=self.shape)


def sample(cc, points):
    """Bilinear sample of a (H, W) map at float (y, x) points, (..., 2) tensor."""
    y, x = points[..., 0], points[..., 1]
    y0, x0 = torch.floor(y).long(), torch.floor(x).long()
    fy, fx = y - y0, x - x0
    return ((1 - fy) * (1 - fx) * cc[y0, x0] + (1 - fy) * fx * cc[y0, x0 + 1]
            + fy * (1 - fx) * cc[y0 + 1, x0] + fy * fx * cc[y0 + 1, x0 + 1])


def fit_axial(cc_bare, sites, direction, device, half_range_a=MONOMER_A / 2):
    """Axial shift (A) maximising the tube-mean bare correlation, +/- half a monomer."""
    shifts = torch.arange(-half_range_a, half_range_a + 0.01, 0.5, device=device)
    pts = sites[None, :, :] + (shifts[:, None, None] / PIXEL) * direction[None, None, :]
    means = sample(cc_bare, pts).mean(dim=1)
    best = int(torch.argmax(means))
    return float(shifts[best]), float(means[best])


def main() -> None:
    """Score bare and saturated synthetic tubes at perturbed poses."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=pathlib.Path,
                        default=MT_ROOT / "results_kinesin_synthetic")
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"

    truth = json.loads((args.synthetic / "truth.json").read_text())
    tubes = truth["tubes"]
    for tube in tubes:
        tube["units"] = np.array(tube["units"])
    stored = np.load(args.synthetic / f"projections_seed{truth['seed']}.npz")
    tubulin, motors = stored["tubulin"], stored["motors"]
    optics = yaml.safe_load((args.synthetic / "optics.yaml").read_text())["optics_group"]
    noise_source = MT_ROOT / "results_kinesin" / "20260916_lysates_2_0031_DW.mrc"
    with mrcfile.open(str(noise_source), permissive=True) as handle:
        source = np.asarray(handle.data, dtype=np.float64)
    shape = source.shape
    noise = phase_randomised(source - source.mean(), truth["seed"])
    filt = ctf_filter(shape, optics, device)
    noise_dft = torch.fft.rfftn(torch.tensor(noise, dtype=torch.float32))
    noise_dft[0, 0] = 0
    white = WhiteningFilterConfig(enabled=True, max_freq=1.0, do_power_spectrum=True) \
        .calculate_whitening_filter(noise_dft).to(device)
    calibration = pd.read_csv(args.synthetic / "occupancy_summary.csv")

    # the two micrographs: bare tubes and saturated tubes, same noise
    images = {}
    for p in (0.0, 1.0):
        canvas = np.zeros(shape)
        for t, tube in enumerate(tubes):
            block = tubulin[t] + (motors[t].sum(axis=0) if p else 0.0)
            for centre in tube["units"]:
                paste(canvas, block, centre)
        signal = torch.fft.irfft2(torch.fft.rfft2(torch.tensor(canvas, dtype=torch.float32,
                                                               device=device)) * filt, s=shape)
        images[p] = truth["scale"] * signal.cpu().numpy() + noise
    scorers = {p: Scorer(img, shape, filt, white, device) for p, img in images.items()}
    noise_scorer = Scorer(noise, shape, filt, white, device)
    inner = (slice(BOX, shape[0] - BOX), slice(BOX, shape[1] - BOX))

    sd_cache = {}

    def noise_sd(name, t, phi, psi):
        key = (name, t, round(phi, 2), round(psi, 2))
        if key not in sd_cache:
            sd_cache[key] = float(noise_scorer.cc_map(name, phi, psi)[inner].std())
        return sd_cache[key]

    site_sets = []
    for t, tube in enumerate(tubes):
        units = tube["units"][END_UNITS:len(tube["units"]) - END_UNITS]
        angle = np.radians(tube["angle_deg"])
        site_sets.append((torch.tensor(np.round(units), dtype=torch.float32, device=device),
                          torch.tensor([np.sin(angle), np.cos(angle)], dtype=torch.float32,
                                       device=device)))

    def score_pose(t, phi, psi, offset_a=(0.0, 0.0), refit=False):
        """Mean z per template and occupancy, at one pose of tube t, both micrographs."""
        sites, along = site_sets[t]
        across = torch.stack([along[1], -along[0]])
        out = {}
        shift = offset_a[0]
        if refit:
            # the bare ring re-fits the axial position on the saturated micrograph (data
            # are what the poses come from; the bare ring barely sees the motors)
            shift, _ = fit_axial(scorers[1.0].cc_map("bare", phi, psi), sites, along, device)
        pts = sites + (shift / PIXEL) * along + (offset_a[1] / PIXEL) * across
        for p, scorer in scorers.items():
            for name in SCORED:
                cc = sample(scorer.cc_map(name, phi, psi), pts)
                out[(p, name)] = float(cc.mean()) / noise_sd(name, t, phi, psi)
        out["shift_A"] = shift
        return out

    def tabulate(label, per_tube_results):
        """Average the per-tube results of one condition into one row."""
        keys = [k for k in per_tube_results[0] if k != "shift_A"]
        mean = {k: float(np.mean([r[k] for r in per_tube_results])) for k in keys}
        row = {"condition": label}
        for name in SCORED:
            row[f"z0_{name}"] = mean[(0.0, name)]
            row[f"z1_{name}"] = mean[(1.0, name)]
        row["read_occ"] = float(occupancy_from_ratio(
            np.array([mean[(1.0, "motors_avg")] / mean[(1.0, "bare")]]), calibration,
            "motors_avg")[0])
        row["mean_abs_shift_A"] = float(np.mean([abs(r["shift_A"]) for r in per_tube_results]))
        return row

    rows = []
    truth_rows = [score_pose(t, tube["phi"], tube["psi"]) for t, tube in enumerate(tubes)]
    rows.append(tabulate("true pose", truth_rows))
    print("true pose done")

    for along_a in (2, 5, 10, 15, 20):
        rows.append(tabulate(f"position +{along_a} A along",
                             [score_pose(t, tb["phi"], tb["psi"], (along_a, 0.0))
                              for t, tb in enumerate(tubes)]))
    for across_a in (2, 5, 10):
        rows.append(tabulate(f"position +{across_a} A across",
                             [score_pose(t, tb["phi"], tb["psi"], (0.0, across_a))
                              for t, tb in enumerate(tubes)]))
    for d in (1, 2, 3, 5, 8):
        rows.append(tabulate(f"roll +{d} deg, axial re-fit",
                             [score_pose(t, tb["phi"] + d, tb["psi"], refit=True)
                              for t, tb in enumerate(tubes)]))
    for d in (0.5, 1, 2):
        rows.append(tabulate(f"psi +{d} deg",
                             [score_pose(t, tb["phi"], tb["psi"] + d)
                              for t, tb in enumerate(tubes)]))
    print("position / roll / psi done")

    # seam: whole protofilament steps, fine roll and axial re-fit by the bare ring
    for k in range(14):
        best = []
        for t, tube in enumerate(tubes):
            sites, along = site_sets[t]
            trials = []
            for d in np.arange(-3.0, 3.01, 0.5):
                phi = tube["phi"] + k * PF_STEP + d
                _, value = fit_axial(scorers[1.0].cc_map("bare", phi, tube["psi"]), sites,
                                     along, device)
                trials.append((value, phi))
            best.append(score_pose(t, max(trials)[1], tube["psi"], refit=True))
        rows.append(tabulate(f"seam moved {k} PF", best))
    print("seam done")

    # polarity: psi + 180, roll over 360 deg (coarse then fine), axial re-fit, by the bare ring
    flipped = []
    for t, tube in enumerate(tubes):
        sites, along = site_sets[t]
        psi = (tube["psi"] + 180.0) % 360.0
        coarse = []
        for phi in np.arange(0.0, 360.0, 2.0):
            _, value = fit_axial(scorers[1.0].cc_map("bare", phi, psi), sites, along, device)
            coarse.append((value, phi))
        centre = max(coarse)[1]
        fine = []
        for phi in np.arange(centre - 2.0, centre + 2.01, 0.25):
            _, value = fit_axial(scorers[1.0].cc_map("bare", phi, psi), sites, along, device)
            fine.append((value, phi))
        flipped.append(score_pose(t, max(fine)[1], psi, refit=True))
    rows.append(tabulate("polarity flipped (bare-fitted roll, axial)", flipped))
    print("polarity done")

    table = pd.DataFrame(rows)
    true = table.iloc[0]
    table["bare_kept"] = table.z1_bare / true.z1_bare
    for name in ("motors", "motors_shift", "motors_avg"):
        contrast = table[f"z1_{name}"] - table[f"z0_{name}"]
        table[f"{name}_contrast"] = contrast
        table[f"{name}_kept"] = contrast / (true[f"z1_{name}"] - true[f"z0_{name}"])
    table.to_csv(args.synthetic / "pose_sensitivity.csv", index=False)
    show = ["condition", "z1_bare", "bare_kept", "motors_contrast", "motors_kept",
            "motors_shift_contrast", "motors_avg_contrast", "motors_avg_kept", "read_occ",
            "mean_abs_shift_A"]
    print(f"\nnoise as calibrated (bare ring on a bare tube z {truth['target_z']}); "
          "contrast = z(saturated) - z(bare tube), mean over 4 tubes x 31 sites")
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
