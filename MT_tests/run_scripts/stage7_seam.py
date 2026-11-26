"""Stage 7 per protofilament: the seam, and the register that is right at every site.

Stage 7 (``stage7_register.py``, ``stage7_every_site.py``) reads ONE register per tube --
half the even-minus-odd difference of (in - out) on the axial index. Two things flip that
parity around a tube, so pooling every protofilament partly cancels it:

* the LABELS. A site's index comes from the helical unwrap of its own roll
  (``continuous_sites``: s + lateral/spacing x phi), and a full turn adds N x the lateral
  rise = 3 monomers on a 14_3 or 13_3 lattice: odd. So a protofilament is indexed in one
  parity where its roll is just above 0 and the other just below 360 -- and with the
  supertwist a protofilament can cross 0/360 partway along a tube;
* the SEAM. Lateral neighbours are alpha-beta across it, so the register really flips.

Binned by roll, a real tube's register can sit at +t on one arc and -t on the other, so a
weak pooled t can be cancellation between arcs that are each clearly called (2026-11-17).

Per filament, from what ``lattice_fill.py sites`` and ``stage7_every_site.py`` wrote
(nothing is re-run; seconds, no GPU):

1. **One labelling.** Each site's roll (``roll_abs``, the roll its index was assigned at)
   less the tube's roll PHASE (roll modulo one protofilament step, circular-mean per
   index, smoothed over +-SMOOTH indices and unwrapped along the tube, so it follows the
   supertwist) gives its protofilament p (0..N-1) and how many turns its roll was counted
   from; the index moves by the turn's monomers to one labelling. There the label flip
   sits between p = N-1 and 0, on the same protofilaments all along the tube.
2. **The model.** in - out = offset + A (-1)^k g_s(p): g flips sign at that cut and at the
   seam s|s+1, and near the seam a patch is part in, part out of register:
   g = +-(n same side - n other side) / WIDTH over its WIDTH protofilaments. Least squares
   for every s; the best is the seam. WIDTH 5 (the patch) fits the 32 tubes best (total
   RSS above it: width 1 +594, 3 +92, 7 +52, 9 +280). With an EVEN turn (e.g. 15_4) a
   B lattice needs no seam and only the seamless pattern is fitted.
3. **How sure.** Sites one index apart share pixels, so a block bootstrap over runs of
   BLOCK axial indices (all protofilaments together): how often a resample picks this
   seam (and within one protofilament), and A's SE. Also each half of the tube alone, and
   the decoys scored with the tube's own pattern (must be null).

Only the seam's place relative to the patch is identifiable: centring the patch's window
one protofilament off shifts the best s by one and fits identically. Labels assume the
patch is centred on its labelled protofilament. A seam inside a stretch of roll with no
sites (some rolls are never found) is placed only to within that gap; the bootstrap
support shows it.

Validation, 13-PF strong synthetic (known rolls 11/47/113/200 deg, a bundle with a crossing
tube): tubes 2 and 4 (same polarity) are 5.5 protofilament steps apart in roll and their
seams 5 apart; tubes 1 and 3 (3.7 apart) fit 2 apart -- tube 3 is the weakest, its runner
up (dRSS 6.7) is 3 apart, and both lie inside the bundle, where a neighbour's density can
reach the patch.

Writes, next to the table:
    stage7_seam_<tag>.csv          per filament
    stage7_seam_sites_<tag>.csv    per site: one-labelling PF and index, g, the register
                                   that is right there (in / out) and whether its patch
                                   straddles the seam -- what scoring "at the right
                                   register" needs
    stage7_seam_<tag>.png          per filament: register per protofilament and the fit

Usage:
    python run_scripts/stage7_seam.py --table results_svd_pipeline/output_correlation_table_<tag>_patch.h5
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage7_register import paired_register  # noqa: E402

WIDTH = 5          # protofilaments in the patch
SMOOTH = 8         # +- axial indices for the roll phase
BLOCK = 8          # axial indices per bootstrap block
N_BOOT = 400
MIN_SITES = 100


def load(table: pathlib.Path):
    """Every-site register scores with each site's roll, and the patch's geometry."""
    results = table.parent
    tag = table.stem.removeprefix("output_correlation_table_")
    scores = pd.read_csv(results / f"stage7_every_register_{tag}.csv", index_col=0)
    sites = pd.read_csv(results / f"lattice_fill_sites_{tag}.csv", index_col=0)
    frame = scores.join(sites[["roll_abs"]])
    if frame["roll_abs"].isna().any():
        raise SystemExit("the every-site scores and lattice_fill's sites do not match")
    path = pd.read_csv(results / f"stage7_refined_{tag}_every_in.csv", nrows=1,
                       usecols=["template_path"])["template_path"].iloc[0]
    stem = pathlib.Path(path).name.split("_bscale")[0].rsplit("_", 1)[0]
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    return results, tag, stem, frame, geometry


def turn_monomers(geometry) -> int:
    """Monomers a full turn adds to the unwrapped index (3 for 13_3 and 14_3)."""
    turn = geometry.n_protofilaments * geometry.lateral_axial_offset_angstrom \
        / geometry.rise_angstrom
    if abs(turn - round(turn)) > 0.1:
        raise SystemExit(f"a turn is {turn:.2f} monomers: not a closed lattice")
    return int(round(turn))


def roll_phase(index, roll, spacing):
    """Function of axial index: the tube's roll phase, smoothed and unwrapped."""
    ks = np.arange(index.min(), index.max() + 1)
    vectors = np.zeros(len(ks), dtype=complex)
    np.add.at(vectors, index - ks[0], np.exp(2j * np.pi * roll / spacing))
    smoothed = np.convolve(vectors, np.ones(2 * SMOOTH + 1), mode="same")
    phase = np.unwrap(np.angle(smoothed)) / (2 * np.pi) * spacing
    return lambda k: phase[np.asarray(k) - ks[0]]


def one_labelling(index, roll, phase, n_pf, spacing, turn):
    """(protofilament 0..N-1, index in one labelling) per site."""
    steps = np.round((roll - phase(index)) / spacing).astype(np.int64)
    turns = np.floor_divide(steps, n_pf)
    return np.mod(steps, n_pf), index - turn * turns


def pattern(pf, n_pf, seam, odd_turn):
    """g_seam(p): sign flips at the label cut and the seam, smeared over the patch."""
    if seam is None:                       # seamless (even turn)
        return np.ones(len(pf))
    if not odd_turn:
        raise ValueError("a single seam needs an odd turn")
    sign = np.where(pf <= seam, 1.0, -1.0)
    crossings = np.zeros(len(pf))
    for offset in range(-(WIDTH // 2), WIDTH // 2 + 1):
        other = pf + offset
        low, high = np.minimum(pf, other), np.maximum(pf, other)
        # seam positions at seam + 0.5 + n_pf * t: count those strictly between
        first = np.ceil((low - seam - 0.5) / n_pf)
        last = np.floor((high - seam - 0.5) / n_pf)
        crossings += np.clip(last - first + 1, 0, None)
    return sign * (1.0 - 2.0 * crossings / WIDTH)


def ols(d, x, weights=None):
    """(A, SE(A), RSS) of d = offset + A x, optionally weighted (bootstrap counts)."""
    w = np.ones(len(d)) if weights is None else weights
    sw = w.sum()
    xm, dm = (w * x).sum() / sw, (w * d).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, np.inf
    a = (w * (x - xm) * (d - dm)).sum() / sxx
    rss = (w * (d - dm - a * (x - xm)) ** 2).sum()
    se = np.sqrt(rss / max(sw - 2, 1) / sxx)
    return a, se, rss


def candidates(n_pf, odd_turn):
    """Seam positions to try: after each protofilament, or none for an even turn."""
    return list(range(n_pf)) if odd_turn else [None]


def fit_filament(part, decoys, geometry):
    """Seam, register and per-site register for one filament's real sites."""
    n_pf = geometry.n_protofilaments
    spacing = float(geometry.protofilament_angular_spacing_deg)
    turn = turn_monomers(geometry)
    odd = turn % 2 == 1
    index = part["axial_index"].to_numpy(dtype=np.int64)
    roll = part["roll_abs"].to_numpy(dtype=float)
    phase = roll_phase(index, roll, spacing)
    pf, k1 = one_labelling(index, roll, phase, n_pf, spacing, turn)
    d = (part["in"] - part["out"]).to_numpy(dtype=float)
    parity = np.where(k1 % 2 == 0, 1.0, -1.0)
    seams = candidates(n_pf, odd)
    designs = {s: parity * pattern(pf, n_pf, s, odd) for s in seams}
    fits = {s: ols(d, x) for s, x in designs.items()}
    rss = np.array([fits[s][2] for s in seams])
    order = np.argsort(rss)
    best = seams[order[0]]
    a, se, _ = fits[best]
    g = pattern(pf, n_pf, best, odd)

    # block bootstrap: per-block sums, resampled by counts
    blocks = (index - index.min()) // BLOCK
    n_blocks = blocks.max() + 1
    rng = np.random.default_rng(0)
    picks, a_boot = [], []
    for _ in range(N_BOOT):
        counts = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks))
        w = counts[blocks].astype(float)
        r = [ols(d, designs[s], w)[2] for s in seams]
        picks.append(seams[int(np.argmin(r))])
        a_boot.append(ols(d, designs[best], w)[0])
    picks = np.array([-1 if p is None else p for p in picks])
    b = -1 if best is None else best
    circular = np.minimum(np.abs(picks - b), n_pf - np.abs(picks - b))
    se_boot = float(np.nanstd(a_boot, ddof=1))

    halves = []
    middle = np.median(index)
    for half in (index < middle, index >= middle):
        r = [ols(d[half], designs[s][half])[2] for s in seams]
        halves.append(seams[int(np.argmin(r))])

    # Stage 7's estimator, pooled (raw index) and with each site's right register
    pooled = paired_register(index, part["in"], part["out"])
    clean = np.abs(g) == 1.0
    flipped = k1 + (g < 0)                       # +1 flips the parity where g < 0
    aware = paired_register(flipped[clean], part["in"].to_numpy()[clean],
                            part["out"].to_numpy()[clean])

    control = np.nan
    if len(decoys) >= 20:
        di = decoys["axial_index"].to_numpy(dtype=np.int64)
        dpf, dk = one_labelling(di, decoys["roll_abs"].to_numpy(dtype=float), phase,
                                n_pf, spacing, turn)
        dd = (decoys["in"] - decoys["out"]).to_numpy(dtype=float)
        da, dse, _ = ols(dd, np.where(dk % 2 == 0, 1.0, -1.0) * pattern(dpf, n_pf, best, odd))
        control = da / dse

    per_pf = pd.DataFrame({"pf": pf, "signed": d * parity}).groupby("pf")["signed"]
    covered = sorted(set(range(n_pf)) - set(pf.tolist()))
    summary = {
        "sites": len(part), "pooled_t": pooled["t"], "pooled_register_z": pooled["mean"],
        "seam": "none" if best is None else f"{best}|{(best + 1) % n_pf}",
        "seam_after_pf": best, "support": float(np.mean(picks == b)),
        "support_pm1": float(np.mean(circular <= 1)),
        "runner_up": None if len(seams) < 2 else seams[order[1]],
        "drss_runner_up": float((rss[order[1]] - rss[order[0]]) / (rss[order[0]] / (len(d) - 2)))
        if len(seams) > 1 else np.nan,
        "halves": "/".join("none" if h is None else str(h) for h in halves),
        "A": float(a), "t": float(abs(a) / se), "t_boot": float(abs(a) / se_boot),
        "register_t": aware["t"], "register_z": aware["mean"],
        "straddling_sites": int((~clean).sum()), "decoy_t": float(control),
        "empty_pfs": " ".join(map(str, covered)),
        "seam_roll_deg": float(np.nan if best is None else
                               (np.median(phase(index)) + (best + 0.5) * spacing) % 360.0),
    }
    sites = pd.DataFrame({
        "filament": part["filament"].to_numpy(), "axial_index": index,
        "pf_one_labelling": pf, "index_one_labelling": k1, "g": g,
        "right_register": np.where(a * parity * g > 0, "in", "out"),
        "straddles_seam": ~clean, "in": part["in"].to_numpy(), "out": part["out"].to_numpy(),
    }, index=part.index)
    return summary, sites, per_pf, g, parity, d, pf


def plot(panels, n_pf, out, tag):
    """Register per protofilament with the fitted pattern, one panel per filament."""
    cols = min(4, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows + 0.6),
                             squeeze=False, sharey=True)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (fil, summary, per_pf, pf, g) in zip(axes.ravel(), panels):
        mean, sem = per_pf.mean(), per_pf.sem()
        ax.axhline(0, color="#AEB6BF", lw=0.8)
        ax.errorbar(mean.index, mean.to_numpy(), yerr=sem.to_numpy(), fmt="o", ms=4,
                    color="#16202B", lw=1, capsize=2, label="mean (in − out) × parity")
        model = pd.Series(summary["A"] * g, index=pf).groupby(level=0).mean()
        ax.plot(model.index, model.to_numpy(), "-", color="#e34948", lw=1.6, label="fit")
        if summary["seam_after_pf"] is not None:
            ax.axvline(summary["seam_after_pf"] + 0.5, color="#e34948", ls=(0, (4, 3)),
                       lw=1)
        ax.set_xlim(-0.6, n_pf - 0.4)
        ax.set_xticks(range(n_pf))
        ax.tick_params(labelsize=8)
        ax.set_title(f"filament {fil}: seam {summary['seam']}, t {summary['register_t']:.1f} "
                     f"(pooled {summary['pooled_t']:.1f})", fontsize=9, loc="left")
        ax.set_xlabel("protofilament (one labelling; label cut between "
                      f"{n_pf - 1} and 0)", fontsize=8)
    axes[0, 0].set_ylabel("register z", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend([*handles, plt.Line2D([], [], color="#e34948", ls=(0, (4, 3)), lw=1)],
               [*labels, "seam"], fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.suptitle(f"{tag}: register per protofilament", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    """Seam and per-site register for every filament of one patch table."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=pathlib.Path, required=True,
                        help="the PATCH table the every-site steps ran on")
    args = parser.parse_args()
    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    results, tag, stem, frame, geometry = load(table)
    n_pf = geometry.n_protofilaments
    turn = turn_monomers(geometry)
    print(f"{tag}: patch {stem} ({n_pf} PF, a turn adds {turn} monomers: "
          f"{'odd, one seam' if turn % 2 else 'even, seamless'}); patch width {WIDTH} PF, "
          f"bootstrap blocks of {BLOCK} indices x {N_BOOT}")
    real, decoy = frame[~frame["decoy"].astype(bool)], frame[frame["decoy"].astype(bool)]
    rows, all_sites, panels = [], [], []
    for fil, part in real.groupby("filament"):
        if len(part) < MIN_SITES:
            print(f"  filament {fil}: {len(part)} sites, too few, left out")
            continue
        summary, sites, per_pf, g, _, _, pf = fit_filament(
            part, decoy[decoy["filament"] == fil], geometry)
        rows.append({"filament": fil, **summary})
        all_sites.append(sites)
        panels.append((fil, summary, per_pf, pf, g))
    if not rows:
        raise SystemExit("no filament with enough sites")
    table_out = pd.DataFrame(rows)
    table_out.to_csv(results / f"stage7_seam_{tag}.csv", index=False)
    pd.concat(all_sites).to_csv(results / f"stage7_seam_sites_{tag}.csv")
    plot(panels, n_pf, results / f"stage7_seam_{tag}.png", tag)

    print(f"\n{'filament':<9}{'sites':>6}{'pooled t':>10}{'seam':>7}{'support':>9}"
          f"{'(+-1)':>7}{'halves':>8}{'register t':>12}{'boot t':>8}{'straddle':>10}"
          f"{'decoy t':>9}  empty PFs")
    for r in rows:
        print(f"{r['filament']:<9}{r['sites']:>6}{r['pooled_t']:>10.1f}{r['seam']:>7}"
              f"{r['support']:>9.2f}{r['support_pm1']:>7.2f}{r['halves']:>8}"
              f"{r['register_t']:>12.1f}{r['t_boot']:>8.1f}{r['straddling_sites']:>10}"
              f"{r['decoy_t']:>9.1f}  {r['empty_pfs'] or '-'}")
    print()
    print("  pooled t: Stage 7's one register per tube; register t: the same estimator with each\n"
          "  site's own register, sites whose patch straddles the seam left out (both treat sites\n"
          "  as independent, as Stage 7 does); boot t: |A| over its block-bootstrap SE (neighbouring\n"
          "  sites share pixels). support: share of resamples choosing this seam (within one PF).\n"
          "  Seam labels: protofilaments counted in roll from the tube's roll phase, patch assumed\n"
          "  centred; only the seam's place relative to the patch is identifiable.")
    print(f"wrote stage7_seam_{tag}.csv, stage7_seam_sites_{tag}.csv and the .png in "
          f"{results.relative_to(MT_ROOT) if results.is_relative_to(MT_ROOT) else results}")


if __name__ == "__main__":
    main()
