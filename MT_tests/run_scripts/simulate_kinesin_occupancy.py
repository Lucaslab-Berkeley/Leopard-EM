"""Kinesin occupancy on synthetic tubes: how each template's z moves with occupancy.

Synthetic micrographs of 14-PF expanded tubes carrying KIF5B rigor motors (3J8X, built by
``models/build_kinesin_templates.py``), every motor bound independently with probability
``p``, imaged with kinesin_2_0031's own optics (200 kV, its PICASSO defocus, the search's
CTF B 60) in noise with that micrograph's power spectrum (phase-randomised). At every
true site three templates are scored at the TRUE pose -- the neutral-pose situation of
Stage 6b, with no search and so no finder bias:

* bare 14-PF ring          (what every search so far used)
* decorated 14-PF ring     (a motor on every beta: saturated)
* motors only              (the decoration alone at its lattice positions)
* the same two with the motors one monomer along (the alpha positions: wrong register
  here, since the synthetic is built in the templates' own register), and with half
  weight on every monomer ("avg": no register, no seam)

How the tubes are made. Projection is linear, so a tube is the sum of 2-ring tubulin
blocks tiled every 4 monomers (167.92 A, the model's own repeat) plus, for each bound
motor, that motor's projection. All volumes are simulated with one fixed ttsim3d
neighbourhood (B 61.18, ``_nb61.18``), so decorated = bare + motors exactly and the
templates differ from the specimen only by occupancy.

How the scores are made. As Leopard scores: the image whitened by its own power
spectrum (``WhiteningFilterConfig``, max_freq 1.0), each template projected by Leopard's
projector, multiplied by CTF x envelope x the same whitening, cross-correlated over the
whole frame; z = correlation at the site / SD of the correlation map of noise alone.
Signal and noise are correlated separately, so z = (scale * signal + noise) / SD for any
noise level without regenerating anything; ``scale`` is set once, from p = 0, so the
bare ring on a bare tube scores ``--target-z`` (default 3.7: kinesin_2_0031's bare-ring
mean z at its neutral poses).

The same tubes, noise-free and before the CTF, calibrate the layer-line check: their 8 nm
/ 4 nm layer-line power (``layer_line_check.py``'s own measurement) at each occupancy is
what a CTF-corrected ratio on data estimates. (The noisy synthetics are no use for this:
the phase-randomised noise keeps the real frame's structured spectrum, and the per-tube
SNR swings 3-11 with it.)

Writes ``results_kinesin_synthetic/``: per-site z (``occupancy_sites.csv``), a summary,
a figure, and each synthetic micrograph with its paths and optics.

Usage:
    python run_scripts/simulate_kinesin_occupancy.py
"""

import argparse
import json
import pathlib
import shutil
import sys

import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import layer_line_check  # noqa: E402
from simulate_curved_microtubule import phase_randomised, psi_from_tangent_angle  # noqa: E402
from simulate_lattice_competition import (  # noqa: E402
    B_FACTOR_SCALING,
    fix_neighborhood,
    sim_config,
)
from torch_ctf import calculate_ctf_2d  # noqa: E402
from ttsim3d.models import Simulator  # noqa: E402

from leopard_em.pydantic_models.config.correlation_filters import (  # noqa: E402
    BandpassFilterConfig,
    WhiteningFilterConfig,
)
from leopard_em.utils.fourier_slice import (  # noqa: E402
    get_real_space_projections_from_volume,
)

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIXEL = 1.13
BOX = 488
NEIGHBORHOOD_B = 61.18
TAG = f"_{PIXEL}_bscale{B_FACTOR_SCALING}_nb{NEIGHBORHOOD_B:g}"
LATTICE = "6dpu_14pf_2rings_flatB"
TEMPLATES = {"bare": LATTICE, "decorated": f"{LATTICE}_kin3j8x",
             "motors": f"{LATTICE}_kin3j8x_motorsonly",
             # the other register (motors on alpha) and half weight on every monomer
             "decorated_shift": f"{LATTICE}_kin3j8x_shift",
             "motors_shift": f"{LATTICE}_kin3j8x_motorsonly_shift",
             "decorated_avg": f"{LATTICE}_kin3j8x_avg",
             "motors_avg": f"{LATTICE}_kin3j8x_motorsavg"}
UNIT_A = 2 * 83.9582        # the model's beta-to-beta repeat x 2: one 2-ring block
CTF_B = 60.0                # the search's CTF B-factor envelope
END_UNITS = 2               # blocks at each tube end not scored (template sees past the end)


def load_map(stem: str) -> torch.Tensor:
    """A template volume of the fixed-neighbourhood kinesin set."""
    path = MT_ROOT / "maps" / f"{stem}{TAG}.mrc"
    with mrcfile.open(str(path), permissive=True) as handle:
        return torch.tensor(np.asarray(handle.data, dtype=np.float32))


def motor_volumes(device: str):
    """Each single motor of the 2-ring block, simulated exactly as the templates."""
    fix_neighborhood(NEIGHBORHOOD_B)
    config = sim_config(200.0, "k3_200kV_CDS", 50.0, True)
    for path in sorted((MT_ROOT / "models" / f"{LATTICE}_kin3j8x_motors").glob("motor_*.pdb")):
        simulator = Simulator(pdb_filepath=str(path), pixel_spacing=PIXEL,
                              volume_shape=(BOX,) * 3, center_atoms=False,
                              remove_hydrogens=True, b_factor_scaling=B_FACTOR_SCALING,
                              additional_b_factor=0, simulator_config=config)
        yield simulator.run(device=device).float().cpu()


def project(volume: torch.Tensor, poses: list[tuple[float, float]], device: str) -> np.ndarray:
    """Leopard projections of ``volume`` at (phi, psi) side views (theta 90)."""
    phi = torch.tensor([p[0] for p in poses], device=device)
    psi = torch.tensor([p[1] for p in poses], device=device)
    theta = torch.full_like(phi, 90.0)
    with torch.no_grad():
        out = get_real_space_projections_from_volume(volume.to(device), phi, theta, psi)
    return out.cpu().numpy()


def layout(shape, n_tubes, rng):
    """Straight parallel tubes across the frame, each with its own roll and tilt."""
    height, width = shape
    step = UNIT_A / PIXEL
    tubes = []
    for t in range(n_tubes):
        angle = np.radians(rng.uniform(-6, 6))
        phi = float(rng.uniform(0, 360))
        centre = np.array([height * (t + 0.5) / n_tubes, width / 2])
        direction = np.array([np.sin(angle), np.cos(angle)])           # (y, x)
        half = (width / 2 - BOX / 2 - 20) / abs(np.cos(angle))
        n = int(half // step)
        units = [centre + k * step * direction for k in range(-n, n + 1)]
        tubes.append({"phi": phi, "psi": float(psi_from_tangent_angle(angle)),
                      "angle_deg": float(np.degrees(angle)), "units": np.array(units)})
    return tubes


def paste(canvas, image, centre):
    """Add a BOX x BOX projection with its centre pixel at round(centre)."""
    y0, x0 = int(round(centre[0])) - BOX // 2, int(round(centre[1])) - BOX // 2
    canvas[y0:y0 + BOX, x0:x0 + BOX] += image


def ctf_filter(shape, optics, device):
    """CTF x the search's B envelope, rfft layout, at frame size."""
    defocus = 0.5 * (optics["defocus_u"] + optics["defocus_v"])
    ctf = calculate_ctf_2d(
        defocus=defocus * 1e-4, astigmatism=0.5 * (optics["defocus_u"] - optics["defocus_v"]) * 1e-4,
        astigmatism_angle=optics["astigmatism_angle"], voltage=optics["voltage"],
        spherical_aberration=optics["spherical_aberration"],
        amplitude_contrast=optics["amplitude_contrast_ratio"], phase_shift=0.0,
        pixel_size=PIXEL, image_shape=shape, rfft=True, fftshift=False).squeeze().to(device)
    fy = torch.fft.fftfreq(shape[0], d=PIXEL, device=device)[:, None]
    fx = torch.fft.rfftfreq(shape[1], d=PIXEL, device=device)[None, :]
    return ctf * torch.exp(-CTF_B * (fy ** 2 + fx ** 2) / 4.0)


def template_rfft(projection, shape, filt, device):
    """A filtered, zero-mean, unit-norm template centred on pixel (0, 0), as an rfft."""
    padded = torch.zeros(shape, device=device)
    padded[:BOX, :BOX] = torch.tensor(projection, device=device)
    padded = torch.roll(padded, shifts=(-(BOX // 2), -(BOX // 2)), dims=(0, 1))
    spectrum = torch.fft.rfft2(padded) * filt
    spectrum[0, 0] = 0
    real = torch.fft.irfft2(spectrum, s=shape)
    return spectrum / torch.linalg.norm(real)


def correlate(image_rfft, template, shape):
    """Cross-correlation map of a whitened image with a filtered template."""
    return torch.fft.irfft2(image_rfft * torch.conj(template), s=shape)


def noise_free_layer_lines(canvas, tubes) -> float:
    """Mean 8 nm / 4 nm layer-line power over the tubes of a noise-free, CTF-free frame."""
    sys.path.insert(0, str(MT_ROOT.parent / "programs" / "constrained_search"))
    from napari_choose_filament_path import (  # pylint: disable=import-outside-toplevel
        spline_through_points,
    )
    window = int(round(58.0 * 10 / PIXEL))
    ratios = []
    for tube in tubes:
        curve, tangent = spline_through_points(tube["units"], smooth_px=1.0)
        strip = layer_line_check.straighten(canvas, curve, tangent,
                                            int(round(25.0 * 10 / PIXEL)))
        pieces = [strip[s:s + window] for s in range(BOX // 2, len(curve) - window - BOX // 2,
                                                       window // 4)]
        _, axial, profile = layer_line_check.spectrum(pieces, PIXEL, 1 / 80.0)
        resolution = 1.0 / (window * PIXEL)
        power = {name: layer_line_check.band_excess(profile, axial, repeat, 0.035,
                                                    resolution)[0]
                 for name, repeat in layer_line_check.LINES.items()}
        ratios.append(power["8 nm"] / power["4 nm"])
    return float(np.mean(ratios))


def main() -> None:
    """Build the synthetics, score every site with every template, summarise."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--occupancies", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--target-z", type=float, default=3.7)
    parser.add_argument("--n-tubes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--noise-source", type=pathlib.Path,
                        default=MT_ROOT / "results_kinesin" / "20260916_lysates_2_0031_DW.mrc")
    parser.add_argument("--optics", type=pathlib.Path,
                        default=MT_ROOT / "results_ctf_kinesin" / "20260916_lysates_2_0031_DW_optics.yaml")
    parser.add_argument("--outdir", type=pathlib.Path, default=MT_ROOT / "results_kinesin_synthetic")
    parser.add_argument("--lowpass-A", type=float, default=None,
                        help="low-pass image and templates at this resolution, as Leopard's "
                             "bandpass_filter does (high_freq_cutoff = pixel / A)")
    parser.add_argument("--no-mrc", action="store_true",
                        help="skip writing the synthetic micrographs")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    args.outdir.mkdir(parents=True, exist_ok=True)
    optics = yaml.safe_load(args.optics.read_text())["optics_group"]
    with mrcfile.open(str(args.noise_source), permissive=True) as handle:
        source = np.asarray(handle.data, dtype=np.float64)
    shape = source.shape
    rng = np.random.default_rng(args.seed)
    noise = phase_randomised(source - source.mean(), args.seed)
    tubes = layout(shape, args.n_tubes, rng)
    poses = [(t["phi"], t["psi"]) for t in tubes]

    # projections: tubulin block, each motor, each template, at every tube's pose
    cache = MT_ROOT / "results_kinesin_synthetic" / f"projections_seed{args.seed}.npz"
    stored = dict(np.load(cache)) if cache.exists() else {}
    if "tubulin" not in stored:
        stored["tubulin"] = project(load_map(LATTICE), poses, device)
        stored["motors"] = np.stack([project(v, poses, device)
                                     for v in motor_volumes(device)], axis=1)
    for name, stem in TEMPLATES.items():
        if f"t_{name}" not in stored:
            stored[f"t_{name}"] = project(load_map(stem), poses, device)
    np.savez(cache, **stored)
    tubulin, motors = stored["tubulin"], stored["motors"]
    templates = {name: stored[f"t_{name}"] for name in TEMPLATES}
    print(f"{len(tubes)} tubes, {sum(len(t['units']) for t in tubes)} blocks, "
          f"{motors.shape[1]} motors per block; frame {shape}")

    filt = ctf_filter(shape, optics, device)
    whitening = WhiteningFilterConfig(enabled=True, max_freq=1.0, do_power_spectrum=True)
    noise_t = torch.tensor(noise, dtype=torch.float32, device=device)
    # as match_template computes it: on the CPU, from the rfft with the DC term zeroed
    noise_dft = torch.fft.rfftn(torch.tensor(noise, dtype=torch.float32))
    noise_dft[0, 0] = 0
    white = whitening.calculate_whitening_filter(noise_dft).to(device)
    if args.lowpass_A is not None:
        bandpass = BandpassFilterConfig(enabled=True, low_freq_cutoff=0.0,
                                        high_freq_cutoff=PIXEL / args.lowpass_A, falloff=0.01)
        white = white * bandpass.calculate_bandpass_filter(noise_dft.shape).to(device)
    noise_w = torch.fft.rfft2(noise_t) * white
    template_ffts = {(name, t): template_rfft(templates[name][t], shape, filt * white, device)
                     for name in TEMPLATES for t in range(len(tubes))}
    inner = (slice(BOX, shape[0] - BOX), slice(BOX, shape[1] - BOX))
    noise_cc, noise_sd = {}, {}
    for key, tf in template_ffts.items():
        cc = correlate(noise_w, tf, shape)
        noise_cc[key] = cc.cpu().numpy()
        noise_sd[key] = float(cc[inner].std())

    rows, images, layer_ratio = [], {}, {}
    for p in args.occupancies:
        canvas = np.zeros(shape)
        bound = []
        for t, tube in enumerate(tubes):
            for k, centre in enumerate(tube["units"]):
                mask = rng.random(motors.shape[1]) < p
                block = tubulin[t] + np.tensordot(mask.astype(float), motors[t], axes=1)
                paste(canvas, block, centre)
                bound.append((t, k, int(mask.sum())))
        layer_ratio[p] = noise_free_layer_lines(canvas, tubes)
        signal = torch.fft.irfft2(torch.fft.rfft2(torch.tensor(canvas, dtype=torch.float32,
                                                               device=device)) * filt, s=shape)
        images[p] = signal.cpu().numpy()
        signal_w = torch.fft.rfft2(signal) * white
        signal_cc = {key: correlate(signal_w, tf, shape).cpu().numpy()
                     for key, tf in template_ffts.items()}
        for t, k, n_bound in bound:
            units = tubes[t]["units"]
            if k < END_UNITS or k >= len(units) - END_UNITS:
                continue
            y, x = (int(round(c)) for c in units[k])
            row = {"occupancy": p, "tube": t, "block": k, "motors_bound": n_bound}
            for name in TEMPLATES:
                row[f"s_{name}"] = float(signal_cc[(name, t)][y, x]) / noise_sd[(name, t)]
                row[f"n_{name}"] = float(noise_cc[(name, t)][y, x]) / noise_sd[(name, t)]
            rows.append(row)
    sites = pd.DataFrame(rows)

    zero = sites[sites.occupancy == 0.0]
    scale = args.target_z / zero["s_bare"].mean()
    for name in TEMPLATES:
        sites[f"z_{name}"] = scale * sites[f"s_{name}"] + sites[f"n_{name}"]
    sites["dz_decorated_bare"] = sites["z_decorated"] - sites["z_bare"]
    sites.to_csv(args.outdir / "occupancy_sites.csv", index=False)
    summary = summarise(sites)
    summary["layer_8_4_noise_free"] = summary.occupancy.map(layer_ratio)
    summary.to_csv(args.outdir / "occupancy_summary.csv", index=False)
    print(f"\nnoise scale set so the bare ring on a bare tube scores z {args.target_z} "
          f"(signal x {scale:.3g})")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # micrographs at that noise level, with paths and optics, for the layer-line check
    paths = [{"points": t["units"].tolist(), "width_px": 460.0, "polarity": "both"}
             for t in tubes]
    for p, image in ({} if args.no_mrc else images).items():
        out = args.outdir / f"synthetic_occ{p:g}.mrc"
        with mrcfile.new(str(out), overwrite=True) as handle:
            handle.set_data((scale * image + noise).astype(np.float32))
            handle.voxel_size = PIXEL
        with open(out.with_suffix(".paths.json"), "w", encoding="utf-8") as handle:
            json.dump({"micrograph_path": str(out), "paths": paths}, handle)
    shutil.copy(args.optics, args.outdir / "optics.yaml")
    (args.outdir / "truth.json").write_text(json.dumps(
        {"scale": scale, "target_z": args.target_z, "seed": args.seed,
         "tubes": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in t.items()} for t in tubes]}, indent=1))
    plot(sites, summary, args.outdir / "occupancy.png", args.target_z)
    print(f"wrote {args.outdir}/occupancy_sites.csv, occupancy_summary.csv, occupancy.png, "
          f"synthetic_occ*.mrc")


def summarise(sites: pd.DataFrame) -> pd.DataFrame:
    """Per occupancy: mean and SD of each z, and how many sites resolve it from bare.

    ``n_3sigma``: blocks needed to tell a stretch at this occupancy from an equally long
    bare stretch at 3 SD by the decorated-minus-bare difference (two independent means;
    neighbouring blocks' noise is uncorrelated, measured); ``n_3sigma_motors`` the same
    by the motors-only z. ``occupancy_sd_block`` is one block's occupancy error by either
    readout (per-site SD / slope of the mean against occupancy).
    """
    rows = []
    base = sites[sites.occupancy == 0.0]
    for p, group in sites.groupby("occupancy"):
        row = {"occupancy": p, "sites": len(group),
               "motors_per_block": group.motors_bound.mean()}
        for name in TEMPLATES:
            row[f"z_{name}"] = group[f"z_{name}"].mean()
            row[f"sd_{name}"] = group[f"z_{name}"].std(ddof=1)
            # noise-free part: ratios of these to the bare ring's do not depend on the
            # noise level, so they calibrate data whose noise scale is unknown
            row[f"signal_{name}"] = (group[f"z_{name}"] - group[f"n_{name}"]).mean()
        row["dz"] = group.dz_decorated_bare.mean()
        row["sd_dz"] = group.dz_decorated_bare.std(ddof=1)
        for key, column in (("n_3sigma", "dz_decorated_bare"), ("n_3sigma_motors", "z_motors")):
            effect = group[column].mean() - base[column].mean()
            spread = np.sqrt(0.5 * (group[column].var(ddof=1) + base[column].var(ddof=1)))
            row[key] = np.inf if effect <= 0 else float(np.ceil((3 * spread / effect) ** 2 * 2))
        rows.append(row)
    out = pd.DataFrame(rows)
    slope = np.polyfit(out.occupancy, out.z_motors, 1)[0]
    out["occupancy_sd_block"] = out.sd_motors / slope
    return out


def plot(sites, summary, out_path, target_z):
    """z of each template, and decorated-minus-bare, against occupancy."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    colours = {"bare": "#1f4e79", "decorated": "#c62828", "motors": "#2e7d32"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, colour in colours.items():
        axes[0].errorbar(summary.occupancy, summary[f"z_{name}"], yerr=summary[f"sd_{name}"],
                         color=colour, marker="o", capsize=3, label=f"{name} template")
    axes[0].set_xlabel("occupancy (fraction of beta-tubulins with a motor)")
    axes[0].set_ylabel("z at the true pose (mean +/- per-site SD)")
    axes[0].set_title(f"noise set: bare on bare tube = z {target_z}", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].errorbar(summary.occupancy, summary.dz, yerr=summary.sd_dz, color="black",
                     marker="o", capsize=3)
    axes[1].axhline(0, color="grey", lw=0.8, ls="--")
    axes[1].set_xlabel("occupancy")
    axes[1].set_ylabel("z decorated - z bare, per site")
    axes[1].set_title("competition at the true pose", fontsize=9)
    finite = summary[np.isfinite(summary.n_3sigma) & (summary.occupancy > 0)]
    axes[2].plot(finite.occupancy, finite.n_3sigma, "o-", color="black",
                 label="decorated - bare")
    finite_m = summary[np.isfinite(summary.n_3sigma_motors) & (summary.occupancy > 0)]
    axes[2].plot(finite_m.occupancy, finite_m.n_3sigma_motors, "o-", color=colours["motors"],
                 label="motors-only z")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("occupancy")
    axes[2].set_ylabel("sites to tell from bare at 3 SD")
    axes[2].set_title("1 site = one 2-ring block (16.8 nm of tube)", fontsize=9)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    main()
