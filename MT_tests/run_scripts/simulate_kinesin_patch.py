"""Kinesin on synthetic tubes, read with 5-protofilament PATCH templates.

The occupancy synthetic of ``simulate_kinesin_occupancy.py`` (same tubes, optics, noise and
scale: the bare RING scores z 3.7 on a bare tube), now scored with patches -- five adjacent
protofilaments, the template Stage 7 uses for the register and the seam -- at every
position round the tube:

* bare patch                 tubulin only
* decorated patch            + a motor on each beta (10 motors)
* motors-only patch          the 10 motors alone
* motors, both registers     half-weight motors on every monomer (no register)

The patches here are cut from the ring models WITHOUT re-centring (protofilaments 0-4,
axis on the box centre), so a patch is scored at exactly the ring's pose -- unlike the
search patches, which sit on their own centre of mass ~87 A off the axis. At known poses
the two are the same template, only boxed differently.

Round the tube: the roll + k protofilament steps (k = 0..13) with the axial shift that
keeps the lattice on itself (k x the lateral rise, 8.995 A; its sign is found from the
bare ring on the noise-free tube) puts the patch on protofilaments k..k+4. The model's
seam lies between protofilaments 13 and 0, so from k = 10 the patch straddles it.

The register: the same template one monomer (41.98 A) further along the axis has every
alpha where the beta was -- on a straight lattice that IS the other register, so the
register difference is z(at the site) - z(one monomer along), the mean of both directions.

Writes ``results_kinesin_synthetic/patch/``: per-site scores, summaries, a figure.

Usage:
    python run_scripts/simulate_kinesin_patch.py
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kinesin_pose_sensitivity import Scorer, sample  # noqa: E402
from simulate_curved_microtubule import phase_randomised  # noqa: E402
from simulate_kinesin_occupancy import (  # noqa: E402
    BOX,
    END_UNITS,
    MT_ROOT,
    NEIGHBORHOOD_B,
    PIXEL,
    TAG,
    ctf_filter,
    paste,
)
from simulate_lattice_competition import fix_neighborhood, sim_config, simulate  # noqa: E402

from leopard_em.pydantic_models.config.correlation_filters import (  # noqa: E402
    WhiteningFilterConfig,
)

LATTICE = "6dpu_14pf_2rings_flatB"
N_PF = 14
PF_STEP = 360.0 / N_PF
MONOMER_A = 83.9582 / 2
LATERAL_A = 3 * MONOMER_A / N_PF          # 3-start: 3 monomers per turn
KEEP = 5                                   # protofilaments in the patch
PATCH_SOURCES = {                          # patch name: (ring model it is cut from)
    "bare": LATTICE,
    "decorated": f"{LATTICE}_kin3j8x",
    "motors": f"{LATTICE}_kin3j8x_motorsonly",
    "motors_shift": f"{LATTICE}_kin3j8x_motorsonly_shift",
}
SCORED = ["bare", "decorated", "motors", "motors_avg"]


def patch_stem(source: str) -> str:
    return f"{source}_axpatch{KEEP}"


def ensure_patch_templates(device: str) -> None:
    """Cut protofilaments 0..KEEP-1 from each ring model (no re-centring), simulate them."""
    ring = gemmi.read_structure(str(MT_ROOT / "models" / f"{LATTICE}.pdb"))[0]
    keep = {ring[i].name for i in range(4 * KEEP)}         # 4 chains per protofilament
    fix_neighborhood(NEIGHBORHOOD_B)
    config = sim_config(200.0, "k3_200kV_CDS", 50.0, True)
    for source in PATCH_SOURCES.values():
        stem = patch_stem(source)
        pdb = MT_ROOT / "models" / f"{stem}.pdb"
        if not pdb.is_file():
            structure = gemmi.read_structure(str(MT_ROOT / "models" / f"{source}.pdb"))
            model = structure[0]
            for name in [c.name for c in model if c.name not in keep]:
                model.remove_chain(name)
            structure.setup_entities()
            structure.write_pdb(str(pdb))
            print(f"  wrote {pdb.name} ({len(model)} chains)")
        if not (MT_ROOT / "maps" / f"{stem}{TAG}.mrc").is_file():
            simulate(stem, device, PIXEL, (BOX,) * 3, config, NEIGHBORHOOD_B)
            print(f"  simulated {stem}{TAG}.mrc")
    average = MT_ROOT / "maps" / f"{patch_stem(PATCH_SOURCES['motors'])}avg{TAG}.mrc"
    if not average.is_file():
        parts = []
        for name in ("motors", "motors_shift"):
            with mrcfile.open(str(MT_ROOT / "maps" / f"{patch_stem(PATCH_SOURCES[name])}{TAG}.mrc"),
                              permissive=True) as handle:
                parts.append(np.asarray(handle.data, dtype=np.float32))
        with mrcfile.new(str(average), overwrite=True) as handle:
            handle.set_data(0.5 * (parts[0] + parts[1]))
            handle.voxel_size = PIXEL


class PatchScorer(Scorer):
    """``Scorer`` over the patch templates (and the ring, for the shift sign)."""

    def __init__(self, image, shape, filt, white, device):
        # pylint: disable=super-init-not-called
        self.shape, self.filt, self.white, self.device = shape, filt, white, device
        self.image_w = torch.fft.rfft2(torch.tensor(image, dtype=torch.float32,
                                                    device=device)) * white
        self.volumes = {}
        for name in SCORED:
            stem = (f"{patch_stem(PATCH_SOURCES['motors'])}avg" if name == "motors_avg"
                    else patch_stem(PATCH_SOURCES[name]))
            with mrcfile.open(str(MT_ROOT / "maps" / f"{stem}{TAG}.mrc"), permissive=True) as handle:
                self.volumes[name] = torch.tensor(np.asarray(handle.data, dtype=np.float32),
                                                  device=device)
        with mrcfile.open(str(MT_ROOT / "maps" / f"{LATTICE}{TAG}.mrc"), permissive=True) as handle:
            self.volumes["ring"] = torch.tensor(np.asarray(handle.data, dtype=np.float32),
                                                device=device)


def main() -> None:
    """Patch scores round every block of the synthetic tubes, at every occupancy."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=pathlib.Path,
                        default=MT_ROOT / "results_kinesin_synthetic")
    parser.add_argument("--occupancies", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    outdir = args.synthetic / "patch"
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_patch_templates(device)

    truth = json.loads((args.synthetic / "truth.json").read_text())
    tubes = truth["tubes"]
    stored = np.load(args.synthetic / f"projections_seed{truth['seed']}.npz")
    tubulin, motors = stored["tubulin"], stored["motors"]
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
    rng = np.random.default_rng(11)

    sites, directions = [], []
    for tube in tubes:
        units = np.array(tube["units"])[END_UNITS:len(tube["units"]) - END_UNITS]
        angle = np.radians(tube["angle_deg"])
        sites.append(torch.tensor(np.round(units), dtype=torch.float32, device=device))
        directions.append(torch.tensor([np.sin(angle), np.cos(angle)], dtype=torch.float32,
                                       device=device))

    def signal_image(p):
        canvas = np.zeros(shape)
        for t, tube in enumerate(tubes):
            for centre in tube["units"]:
                mask = rng.random(motors.shape[1]) < p
                paste(canvas, tubulin[t] + np.tensordot(mask.astype(float), motors[t], axes=1),
                      centre)
        signal = torch.fft.irfft2(torch.fft.rfft2(torch.tensor(canvas, dtype=torch.float32,
                                                               device=device)) * filt, s=shape)
        return truth["scale"] * signal.cpu().numpy()

    # the sign of the axial shift that goes with one protofilament step of roll: the bare
    # ring on the noise-free bare tube must land on itself
    clean = PatchScorer(signal_image(0.0), shape, filt, white, device)
    t0 = 0
    best = {}
    for sign in (+1.0, -1.0):
        cc = clean.cc_map("ring", tubes[t0]["phi"] + PF_STEP, tubes[t0]["psi"])
        pts = sites[t0] + sign * (LATERAL_A / PIXEL) * directions[t0]
        best[sign] = float(sample(cc, pts).mean())
    reference = float(sample(clean.cc_map("ring", tubes[t0]["phi"], tubes[t0]["psi"]),
                             sites[t0]).mean())
    sign = max(best, key=best.get)
    print(f"one protofilament step of roll goes with {sign * LATERAL_A:+.3f} A along the axis "
          f"(ring cc {best[sign]:.1f} vs {best[-sign]:.1f}; {reference:.1f} unrolled)")
    del clean

    noise_scorer = PatchScorer(noise, shape, filt, white, device)
    inner = (slice(BOX, shape[0] - BOX), slice(BOX, shape[1] - BOX))
    sd = {}
    rows = []
    for p in args.occupancies:
        scorer = PatchScorer(signal_image(p) + noise, shape, filt, white, device)
        for t, tube in enumerate(tubes):
            for k in range(N_PF):
                phi, psi = tube["phi"] + k * PF_STEP, tube["psi"]
                base = sites[t] + sign * k * (LATERAL_A / PIXEL) * directions[t]
                step = (MONOMER_A / PIXEL) * directions[t]
                for name in SCORED:
                    key = (name, t, k)
                    if key not in sd:
                        sd[key] = float(noise_scorer.cc_map(name, phi, psi)[inner].std())
                    cc = scorer.cc_map(name, phi, psi)
                    here = sample(cc, base).cpu().numpy() / sd[key]
                    other = 0.5 * (sample(cc, base + step) + sample(cc, base - step)) \
                        .cpu().numpy() / sd[key]
                    for b, (z, z_other) in enumerate(zip(here, other)):
                        rows.append({"occupancy": p, "tube": t, "block": b, "k": k,
                                     "template": name, "z": z, "z_other_register": z_other})
        print(f"  occupancy {p:g} scored")
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "patch_sites.csv", index=False)
    summarise(frame, outdir)


def summarise(frame: pd.DataFrame, outdir: pathlib.Path) -> None:
    """Occupancy precision per patch site and per block; the register, bare vs decorated."""
    wide = frame.pivot_table(index=["occupancy", "tube", "block", "k"], columns="template",
                             values=["z", "z_other_register"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide["dz"] = wide.z_decorated - wide.z_bare
    seam_free = wide.k <= N_PF - KEEP
    print(f"\nseam-free patch positions: k = 0..{N_PF - KEEP}; k = {N_PF - KEEP + 1}..{N_PF - 1} "
          "straddle the model's seam")

    def precision(data, column):
        groups = data.groupby("occupancy")[column]
        mean, spread = groups.mean(), groups.std(ddof=1)
        slope = np.polyfit(mean.index, mean.to_numpy(), 1)[0]
        block_sd = float(np.sqrt((spread ** 2).mean()))
        return slope, block_sd, block_sd / abs(slope), mean

    print("\nOCCUPANCY, one patch site (seam-free positions pooled):")
    lines = []
    for column, label in (("dz", "decorated - bare patch"), ("z_motors", "motors patch"),
                          ("z_motors_avg", "motors patch, both registers"),
                          ("z_bare", "bare patch")):
        slope, sd, occ_sd, mean = precision(wide[seam_free], column)
        lines.append({"readout": label, "per": "patch site", "slope": slope, "sd": sd,
                      "occupancy_sd": occ_sd})
        print(f"  {label:<30} slope {slope:+.2f} z/occ, SD {sd:.2f} -> occupancy +/-{occ_sd:.2f} "
              f"per site; mean z by occupancy {np.round(mean.to_numpy(), 2)}")
    # one block = all 14 positions round the tube averaged (they overlap: SD measured)
    block = wide.groupby(["occupancy", "tube", "block"])[["dz", "z_motors", "z_motors_avg"]] \
        .mean().reset_index()
    print("\nOCCUPANCY, one 16.8 nm block (all 14 patch positions averaged):")
    for column, label in (("dz", "decorated - bare patch"), ("z_motors", "motors patch"),
                          ("z_motors_avg", "motors patch, both registers")):
        slope, sd, occ_sd, _ = precision(block, column)
        lines.append({"readout": label, "per": "block (14 positions)", "slope": slope, "sd": sd,
                      "occupancy_sd": occ_sd})
        print(f"  {label:<30} slope {slope:+.2f}, SD {sd:.2f} -> occupancy +/-{occ_sd:.2f} per "
              "block")

    print("\nOCCUPANCY by position round the tube (motors patch, both registers; "
          "occupancy SD per site):")
    by_k = []
    for k, group in wide.groupby("k"):
        slope, sd, occ_sd, _ = precision(group, "z_motors_avg")
        by_k.append({"k": k, "slope": slope, "sd": sd, "occupancy_sd": occ_sd})
    by_k = pd.DataFrame(by_k)
    print("  " + "  ".join(f"k{int(r.k)} {r.occupancy_sd:.2f}" for _, r in by_k.iterrows()))

    print("\nREGISTER, one patch site: z(in register) - z(one monomer along); sites for a 3 SD "
          "call")
    register = []
    for name in ("bare", "decorated", "motors"):
        diff = wide[f"z_{name}"] - wide[f"z_other_register_{name}"]
        for p, group in diff.groupby(wide.occupancy):
            free = group[seam_free.loc[group.index]]
            mean, spread = free.mean(), free.std(ddof=1)
            need = (3 * spread / abs(mean)) ** 2 if mean else np.inf
            register.append({"template": name, "occupancy": p, "mean": mean, "sd": spread,
                             "sites_3sd": need})
    register = pd.DataFrame(register)
    for name, group in register.groupby("template", sort=False):
        print(f"  {name:<10}" + "  ".join(
            f"p{r.occupancy:g}: {r['mean']:+.2f}±{r.sd:.2f} ({r.sites_3sd:.0f} sites)"
            for _, r in group.iterrows()))
    by_k_register = (wide.assign(diff=wide.z_decorated - wide.z_other_register_decorated)
                     .groupby(["occupancy", "k"])["diff"].mean().unstack())
    print("\nREGISTER by position round the tube, decorated patch, mean difference:")
    print(by_k_register.round(2).to_string())

    pd.DataFrame(lines).to_csv(outdir / "patch_occupancy_precision.csv", index=False)
    by_k.to_csv(outdir / "patch_occupancy_by_position.csv", index=False)
    register.to_csv(outdir / "patch_register.csv", index=False)
    by_k_register.to_csv(outdir / "patch_register_by_position.csv")
    plot(wide, register, by_k, by_k_register, outdir / "patch.png")


def plot(wide, register, by_k, by_k_register, out_path):
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for name, colour in (("bare", "#1f4e79"), ("decorated", "#c62828"), ("motors", "#2e7d32")):
        group = register[register.template == name]
        axes[0].errorbar(group.occupancy, group["mean"], yerr=group.sd, marker="o", capsize=3,
                         color=colour, label=f"{name} patch")
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("occupancy")
    axes[0].set_ylabel("z(in register) − z(one monomer along)")
    axes[0].set_title("register signal per patch site (seam-free positions)", fontsize=10)
    axes[0].legend(fontsize=8)
    for p in by_k_register.index:
        axes[1].plot(by_k_register.columns, by_k_register.loc[p], marker="o", ms=3,
                     label=f"p = {p:g}")
    axes[1].axvspan(N_PF - KEEP + 0.5, N_PF - 0.5, color="grey", alpha=0.15)
    axes[1].set_xlabel("patch position k (protofilament steps); grey: straddles the seam")
    axes[1].set_ylabel("register difference, decorated patch")
    axes[1].legend(fontsize=8)
    axes[2].plot(by_k.k, by_k.occupancy_sd, marker="o", color="#6a1b9a")
    axes[2].set_xlabel("patch position k")
    axes[2].set_ylabel("occupancy SD per patch site (both registers)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    main()
