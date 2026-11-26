"""Rebuild the per-pixel constraint from the RING search instead of the hand drawing.

The first constraint took psi_center from the tangent of a hand-drawn spline. That was
the only thing available before a search had run, but it is not the best estimate once
one has: the ring match_template measures psi and theta at every site it finds.

Checked on the GDP frame, the drawing's error is not noise but smooth structure in arc
length -- region 2's psi deviates from the drawn tangent with sd 3.01 degrees while the
site-to-site scatter is only 0.72 degrees. Nearly all of that 3 degrees is the drawing
wandering, and a smooth fit to the MEASURED angles removes it: residual sd 0.72 degrees,
max 1.54. Region 1 goes from sd 4.64 to 2.21.

That matters because the old box was CLIPPING real orientations. Refined theta reaches
79.0 and 99.1 degrees against a search box of [80, 100], and refine_template is not
constrained -- so at those sites the search was never allowed to consider the true
orientation. Psi likewise ran to 9.9 degrees against a 10 degree cone.

So this rebuilds the sidecar with:
  * psi_center from a smooth fit to the ring's own refined psi, per region;
  * theta_center re-centred on the measured median;
  * a cone wide enough that neither tail is clipped.

⚠️ One knob, two angles. ``cone_half_angle_deg`` gates BOTH theta (theta_center +/- cone)
and psi (psi_center +/- cone), so the cone cannot be tightened past what theta needs even
though region 2's psi would now be happy with 2 degrees. Decoupling them would need a
per-region or per-pixel theta bound, which the sidecar format does not carry. The gain
here is correctness -- nothing true is excluded -- not a smaller orientation count.

Usage:
    python run_scripts/refit_constraint_from_ring.py
"""

import argparse
import importlib.util
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = MT_ROOT.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402

POLE_AGREEMENT = 0.9  # restrict a region's pole only when this share of sites agree
SMOOTH_PSI = 200.0   # UnivariateSpline s = n * this; ~2 deg of slack per site


def _load_tool():
    path = REPO_ROOT / "programs" / "constrained_search" / "napari_choose_filament_path.py"
    spec = importlib.util.spec_from_file_location("_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def fitted_psi(curve, arc, psi_measured):
    """Smooth psi against arc length, fitted on the doubled angle so it cannot wrap."""
    order = np.argsort(arc)
    a, p = arc[order], psi_measured[order]
    doubled = np.degrees(np.unwrap(np.radians(2.0 * p))) / 2.0
    spline = UnivariateSpline(a, doubled, s=len(a) * SMOOTH_PSI)
    residual = doubled - spline(a)
    return spline, float(residual.std()), float(np.abs(residual).max())


def main() -> None:
    """Refit psi_center from the ring's measurements and write a new sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="gdp_curved_13pf")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--source", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--paths", type=pathlib.Path,
                        default=MT_ROOT / "configs" / "gdp_curved_pol.paths.json")
    parser.add_argument("--out", default="filament_constraint_gdp_curved_refit")
    parser.add_argument("--width-px", type=float, default=560.0,
                        help="band width; must match the run, not the JSON's stale value")
    parser.add_argument("--cone", type=float, default=None,
                        help="override the cone; default is set from the theta tails")
    parser.add_argument("--micrograph", type=pathlib.Path,
                        default=MT_ROOT / "Frames" / "GDP_curved_mgraph.mrc",
                        help="recorded in the sidecar")
    parser.add_argument("--psi-step", type=float, default=1.5,
                        help="match the search grid the new constraint will gate")
    parser.add_argument("--theta-step", type=float, default=2.5)
    parser.add_argument("--restrict-polarity", action="store_true",
                        help="set each region's pole from the ring's refined psi, when "
                             "at least POLE_AGREEMENT of its sites agree")
    args = parser.parse_args()

    refined = pd.read_csv(
        MT_ROOT / args.results / f"refined_sites_{args.tag}.csv", index_col=0)
    stack = pd.read_csv(
        MT_ROOT / args.results / f"stack_sites_{args.tag}.csv", index_col=0)
    # The refine stack carries the search's optics, so its pixel size is the one these
    # sites were measured at -- not a constant.
    pixel_size = float(stack["pixel_size"].iloc[0])
    print(f"pixel size {pixel_size:.4f} A (from the refine stack)")
    with h5py.File(args.source, "r") as handle:
        region_map = handle["maps/region_id"][:]
        eligible_old = handle["maps/eligible"][:]
        drawn = {int(name): handle[f"regions/{name}/path"][:]
                 for name in handle["regions"]}
        psi_old = handle["maps/psi_center"][:]
    shape = region_map.shape

    y = np.clip(stack["pos_y_img"].astype(int), 0, shape[0] - 1)
    x = np.clip(stack["pos_x_img"].astype(int), 0, shape[1] - 1)
    # Stage 5 stacks carry data-driven membership; older ones only the drawn region.
    region_of = (stack["filament"].to_numpy()[: len(refined)]
                 if "filament" in stack.columns else region_map[y, x])
    psi_measured = refined["refined_psi"].to_numpy()
    theta = refined["refined_theta"].to_numpy()

    micrograph_path, paths = TOOL.load_paths_json(str(args.paths))
    for path in paths:
        path["width_px"] = float(args.width_px)

    # theta: centre on the median and open the cone until neither tail is clipped.
    # MIDRANGE, not the median: the cone has to cover both tails, so the centre that
    # minimises the required width is the midpoint of the range. On this frame theta is
    # near-symmetric about 89.1 with a half-range of 10.0, so simply moving the centre
    # off 90.0 takes clipping from 3 sites to 0 at an unchanged cone width -- the
    # original box was the right size and the wrong place.
    centre = float(0.5 * (theta.min() + theta.max()))
    needed = float(np.max(np.abs(theta - centre)))
    cone = args.cone if args.cone is not None else float(np.round(needed + 0.2, 1))
    print(f"theta measured {theta.min():.1f}-{theta.max():.1f}, median {centre:.1f}")
    with h5py.File(args.source, "r") as handle:
        first = handle["regions"][sorted(handle["regions"])[0]].attrs
        old_centre = float(first.get("theta_center_deg", 90.0))
        old_cone = float(first.get("cone_half_angle_deg", 10.0))
    old_lo, old_hi = old_centre - old_cone, old_centre + old_cone
    print(f"  the search's box [{old_lo:.1f}, {old_hi:.1f}] clipped "
          f"{int((theta < old_lo).sum() + (theta > old_hi).sum())} of {len(theta)} sites")
    print(f"  new centre {centre:.1f}, cone {cone:.1f} -> "
          f"[{centre - cone:.1f}, {centre + cone:.1f}]\n")

    # psi_center: fit per region, then paint along each centreline. Painting goes
    # through the exporter's own layered rasterizer, so overlapping bands keep every
    # covering region here exactly as they do for a drawing.
    for index, path in enumerate(paths, start=1):
        curve_points, tangent = TOOL.spline_through_points(path["points"])
        curve = A.DrawnCurve(drawn[index], pixel_size=pixel_size)
        mask = region_of == index
        if mask.sum() < 10:
            print(f"region {index}: too few sites, keeping the drawn tangent")
            psi_along = TOOL.psi_from_tangent(tangent)
        else:
            arc, _ = curve.project(
                np.column_stack([stack["pos_x_img"][mask], stack["pos_y_img"][mask]]))
            # Region 1 holds sites at both poles; fit only the majority one or the
            # unwrap tears. Polarity is a property of the tube, not of the site.
            folded = psi_measured[mask] % 180.0
            spline, sd, peak = fitted_psi(curve, arc, folded)
            drawn_sd = float(np.std(((psi_measured[mask]
                                      - psi_old[y[mask], x[mask]] + 90) % 180) - 90))
            print(f"region {index}: {int(mask.sum())} sites, arc "
                  f"{arc.min():.0f}-{arc.max():.0f} of {curve.length_px:.0f} px")
            print(f"  drawn tangent residual sd {drawn_sd:.2f}°  ->  "
                  f"fitted psi residual sd {sd:.2f}°, max {peak:.2f}°")
            # Evaluate on the curve's own arc grid, clamped to the fitted span so the
            # unmeasured ends hold the nearest fitted value rather than extrapolating.
            arc_grid = np.clip(curve.arc, arc.min(), arc.max())
            psi_along = spline(arc_grid) % 360.0
            # Keep psi_center on the SAME POLE the drawn tangent used. Polarity is
            # carried separately by each region's pole_mask, which Leopard-EM derives
            # at load time as psi_center or psi_center+180 -- so folding the measured
            # pole into psi_center here would apply the flip twice.
            rows0 = np.clip(np.rint(curve.curve[:, 0]).astype(int), 0, shape[0] - 1)
            cols0 = np.clip(np.rint(curve.curve[:, 1]).astype(int), 0, shape[1] - 1)
            reference = psi_old[rows0, cols0]
            flip = np.abs(((psi_along - reference + 180) % 360) - 180) > 90
            psi_along = np.where(flip, (psi_along + 180.0) % 360.0, psi_along)
            curve_points = curve.curve
            if args.restrict_polarity:
                # Which pole of THIS psi_center the sites point along. Compare each
                # site's refined psi with psi_center at its own arc position.
                at_site = psi_along[np.clip(np.searchsorted(curve.arc, arc), 0,
                                            len(psi_along) - 1)]
                same = np.abs(((psi_measured[mask] - at_site + 180) % 360) - 180) < 90
                agree = max(same.mean(), 1 - same.mean())
                if agree >= POLE_AGREEMENT:
                    path["polarity"] = "positive" if same.mean() >= 0.5 else "negative"
                print(f"  polarity: {same.mean():.1%} of sites on psi_center, "
                      f"{1 - same.mean():.1%} opposite -> {path['polarity']}")

        psi_along = np.broadcast_to(np.asarray(psi_along, dtype=np.float64),
                                    (len(curve_points),))
        path["curve_yx"] = np.asarray(curve_points, dtype=np.float64)
        path["psi_deg"] = np.asarray(psi_along[: len(curve_points)], dtype=np.float64)

    layers = TOOL.paint_path_layers(shape, paths)
    eligible, region_new = layers["eligible"], layers["region_id"]
    psi_new, distance = layers["psi_center"], layers["distance"]

    n_orientations = TOOL.estimate_n_orientations(
        eligible, cone, args.psi_step, args.theta_step, theta_center_deg=centre)
    out_h5 = MT_ROOT / "configs" / f"{args.out}.h5"
    TOOL.write_constraint_hdf5(
        str(out_h5), eligible, region_new, psi_new, n_orientations,
        distance.astype(np.float32), paths, cone, centre, args.psi_step,
        args.theta_step, pixel_size,
        psi_center_extra=layers["psi_center_extra"],
        region_id_extra=layers["region_id_extra"],
    )

    # Keyword arguments, not positional: the signature is
    # (paths, cone, theta_center, psi_step, theta_step, micrograph, sidecar) and passing
    # them in the wrong order produced a silently scrambled YAML rather than an error.
    yaml_text = TOOL.dump_sidecar_yaml(TOOL.build_sidecar_payload(
        paths,
        cone_half_angle_deg=cone,
        theta_center_deg=centre,
        psi_step=args.psi_step,
        theta_step=args.theta_step,
        micrograph_path=str(args.micrograph if args.micrograph.is_absolute()
                            else MT_ROOT / args.micrograph),
        spatial_constraint_path=str(out_h5),
    ))
    (MT_ROOT / "configs" / f"{args.out}.yaml").write_text(yaml_text)

    changed = np.abs(((psi_new - psi_old + 180) % 360) - 180)[eligible > 0]
    print(f"\neligible {int(eligible.sum()):,} px ({eligible.sum() / eligible.size:.2%}), "
          f"was {int(eligible_old.sum()):,}")
    print(f"psi_center moved: median {np.median(changed):.2f}°, "
          f"90th pct {np.percentile(changed, 90):.2f}°, max {changed.max():.2f}°")
    print(f"wrote configs/{args.out}.yaml and .h5")


if __name__ == "__main__":
    main()
