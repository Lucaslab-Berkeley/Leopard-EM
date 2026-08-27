r"""Render opposite-pole template overlays for membrane polarity picking.

Run this in the Leopard-EM environment (torch + leopard_em). It does **not**
open napari. Output is a folder of numpy images plus a JSON manifest that
``napari_choose_membrane_polarity.py`` can load in a GUI-only environment.

``positive`` searches ``psi_center`` only; ``negative`` searches
``psi_center + 180``; ``both`` searches both Euler boxes.

Example
-------
::

    python prepare_membrane_polarity_overlays.py \\
        --micrograph /path/to/micrograph.mrc \\
        --template /path/to/template.mrc \\
        --constraint membrane_constraint.yaml \\
        --output polarity_preview
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_POLE_A_RGB = np.array([1.0, 0.0, 1.0], dtype=np.float32)  # magenta
_POLE_B_RGB = np.array([0.0, 1.0, 1.0], dtype=np.float32)  # cyan


def _progress(message: str) -> None:
    """Print a status line and flush so batch runs show progress immediately."""
    print(message, flush=True)


def path_relative_to(path: str | Path, base_dir: Path) -> str:
    """Return ``path`` as a portable relative path from ``base_dir``."""
    return os.path.relpath(str(Path(path).resolve()), str(base_dir.resolve()))


def wrap_360(angle_deg: float) -> float:
    """Wrap an angle into ``[0, 360)``."""
    return float(angle_deg % 360.0)


def normal_yx_from_psi(psi_deg: float) -> tuple[float, float]:
    """Unit membrane normal ``(ny, nx)`` from ``psi = atan2(-ny, nx)``."""
    rad = np.deg2rad(float(psi_deg))
    return float(-np.sin(rad)), float(np.cos(rad))


def load_mrc_image(path: str) -> tuple[np.ndarray, float | None]:
    """Load a 2D MRC as float32 ``(y, x)`` and pixel size in Angstroms if present."""
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        image = np.asarray(mrc.data, dtype=np.float32).squeeze()
        voxel = float(getattr(mrc.voxel_size, "x", 0.0) or 0.0)
        if voxel <= 1e-6:
            nx = int(mrc.header.nx)
            cella = float(mrc.header.cella.x)
            voxel = cella / nx if nx > 0 and cella > 1e-6 else 0.0
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D MRC micrograph, got shape {image.shape}.")
    return image, (voxel if voxel > 1e-6 else None)


def load_mrc_volume_numpy(path: str) -> np.ndarray:
    """Load a 3D MRC template as float32 ``(z, y, x)``."""
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        volume = np.asarray(mrc.data, dtype=np.float32).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRC template, got shape {volume.shape}.")
    return volume


def imagej_contrast_limits(
    image: np.ndarray, saturated_percent: float = 0.35
) -> tuple[float, float]:
    """Display range matching ImageJ Enhance Contrast (no Normalize)."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    if saturated_percent <= 0.0:
        return float(np.min(finite)), float(np.max(finite))
    tail = min(saturated_percent / 2.0, 50.0)
    low, high = np.percentile(finite, (tail, 100.0 - tail))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return float(np.min(finite)), float(np.max(finite))
    return float(low), float(high)


def normalize_display(image: np.ndarray, saturated_percent: float = 0.35) -> np.ndarray:
    """Scale an image to ``[0, 1]`` with ImageJ-style contrast."""
    low, high = imagej_contrast_limits(image, saturated_percent)
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    scaled = (np.asarray(image, dtype=np.float32) - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def load_constraint(yaml_path: str) -> tuple[dict[str, Any], str]:
    """Load the Euler-box YAML and resolve its spatial-constraint HDF5 path."""
    payload = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    hdf5_path = payload.get("spatial_constraint_path")
    if not hdf5_path:
        raise ValueError(f"{yaml_path} has no spatial_constraint_path.")
    hdf5_path = str(Path(hdf5_path))
    if not Path(hdf5_path).is_file():
        alt = Path(yaml_path).resolve().parent / Path(hdf5_path).name
        if alt.is_file():
            hdf5_path = str(alt)
        else:
            raise FileNotFoundError(f"Constraint HDF5 not found: {hdf5_path}")
    return payload, hdf5_path


def load_constraint_maps(hdf5_path: str) -> dict[str, Any]:
    """Read eligible, region_id, psi_center, and signed_distance from HDF5."""
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        maps = handle["maps"]
        eligible = np.asarray(maps["eligible"][:], dtype=np.uint8)
        region_id = np.asarray(maps["region_id"][:], dtype=np.int16)
        if "psi_center" not in maps:
            raise ValueError(f"{hdf5_path} has no maps/psi_center.")
        psi_center = np.asarray(maps["psi_center"][:], dtype=np.float32)
        signed_distance = (
            np.asarray(maps["signed_distance"][:], dtype=np.float32)
            if "signed_distance" in maps
            else np.zeros(eligible.shape, dtype=np.float32)
        )
        polarities: dict[int, str] = {}
        if "regions" in handle:
            for name in sorted(handle["regions"].keys()):
                attrs = handle["regions"][name].attrs
                rid = int(attrs.get("region_id", int(name)))
                polarities[rid] = str(attrs.get("polarity", "unset"))
    return {
        "eligible": eligible,
        "region_id": region_id,
        "psi_center": psi_center,
        "signed_distance": signed_distance,
        "polarities": polarities,
    }


def region_ids_from_maps(region_id: np.ndarray) -> list[int]:
    """Return sorted positive region ids present in the map."""
    ids = np.unique(region_id)
    return [int(rid) for rid in ids if int(rid) > 0]


def polarities_from_yaml(payload: dict[str, Any]) -> dict[int, str]:
    """Read per-region polarity from a sidecar dict."""
    default = str(payload.get("polarity", "unset"))
    mapping: dict[int, str] = {}
    for index, region in enumerate(payload.get("regions") or [], start=1):
        rid = int(region.get("region_id", index))
        mapping[rid] = str(region.get("polarity", default))
    return mapping


def sample_membrane_sites(
    region_id: np.ndarray,
    signed_distance: np.ndarray,
    psi_center: np.ndarray,
    rid: int,
    n_sites: int = 4,
) -> list[tuple[float, float, float]]:
    """Return ``(y, x, psi)`` sites spread along one membrane."""
    mask = region_id == int(rid)
    if not np.any(mask):
        return []
    abs_sd = np.abs(signed_distance)
    inside = abs_sd[mask]
    cutoff = float(np.percentile(inside, 25)) + 1e-3
    band = mask & (abs_sd <= cutoff)
    if not np.any(band):
        band = mask
    ys, xs = np.nonzero(band)
    psi = psi_center[ys, xs].astype(np.float64)
    ny = -np.sin(np.deg2rad(psi))
    nx = np.cos(np.deg2rad(psi))
    ty, tx = -nx, ny
    proj = ys.astype(np.float64) * np.mean(ty) + xs.astype(np.float64) * np.mean(tx)
    order = np.argsort(proj)
    n_sites = max(1, min(int(n_sites), order.size))
    picks = np.linspace(0, order.size - 1, n_sites).astype(int)
    sites = []
    for pick in picks:
        idx = int(order[pick])
        sites.append((float(ys[idx]), float(xs[idx]), wrap_360(float(psi[idx]))))
    return sites


def template_projections(
    volume: np.ndarray, psi_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Real-space projections at pole A (``psi``) and pole B (``psi + 180``)."""
    import torch

    from leopard_em.utils.fourier_slice import get_real_space_projections_from_volume

    vol = torch.as_tensor(volume, dtype=torch.float32)
    psi_a = wrap_360(psi_deg)
    psi_b = wrap_360(psi_deg + 180.0)
    phi = torch.zeros(2, dtype=torch.float32)
    theta = torch.full((2,), 90.0, dtype=torch.float32)
    psi = torch.tensor([psi_a, psi_b], dtype=torch.float32)
    projections = get_real_space_projections_from_volume(
        volume=vol, phi=phi, theta=theta, psi=psi, degrees=True
    )
    proj = projections.detach().cpu().numpy().astype(np.float32)
    return proj[0], proj[1]


def overlay_translate(
    center_y: float,
    center_x: float,
    psi_deg: float,
    template_width: int,
    sign: float,
) -> tuple[float, float]:
    """Top-left of a projection placed on one leaflet along the normal."""
    ny, nx = normal_yx_from_psi(psi_deg)
    offset = 0.5 * float(template_width)
    cy = center_y + sign * offset * ny
    cx = center_x + sign * offset * nx
    half = 0.5 * float(template_width)
    return cy - half, cx - half


def paste_colored_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    translate_yx: tuple[float, float],
    color_rgb: np.ndarray,
) -> None:
    """Add a grayscale patch, colored RGB, onto a ``(H, W, 3)`` canvas."""
    height, width = canvas.shape[:2]
    patch_h, patch_w = patch.shape
    y0 = int(round(translate_yx[0]))
    x0 = int(round(translate_yx[1]))
    y1 = y0 + patch_h
    x1 = x0 + patch_w
    cy0 = max(0, y0)
    cx0 = max(0, x0)
    cy1 = min(height, y1)
    cx1 = min(width, x1)
    if cy0 >= cy1 or cx0 >= cx1:
        return
    py0 = cy0 - y0
    px0 = cx0 - x0
    py1 = py0 + (cy1 - cy0)
    px1 = px0 + (cx1 - cx0)
    gray = np.clip(patch[py0:py1, px0:px1], 0.0, 1.0)[..., None]
    canvas[cy0:cy1, cx0:cx1] += gray * color_rgb[None, None, :]


def render_region_overlay(
    image_shape: tuple[int, int],
    sites: list[dict[str, Any]],
) -> np.ndarray:
    """Paint pole-A (magenta) and pole-B (cyan) projections onto an RGB canvas."""
    canvas = np.zeros((*image_shape, 3), dtype=np.float32)
    for site in sites:
        paste_colored_patch(canvas, site["proj_a"], site["translate_a"], _POLE_A_RGB)
        paste_colored_patch(canvas, site["proj_b"], site["translate_b"], _POLE_B_RGB)
    np.clip(canvas, 0.0, 1.0, out=canvas)
    return canvas


def write_overlay_package(
    output_dir: Path,
    micrograph: np.ndarray,
    maps: dict[str, Any],
    payload: dict[str, Any],
    yaml_path: str,
    hdf5_path: str,
    polarities: dict[int, str],
    overlays: dict[int, np.ndarray],
    site_counts: dict[int, int],
    pixel_size_angstrom: float | None,
) -> Path:
    """Write numpy images and a JSON manifest under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    _progress(f"Writing overlay package to {output_dir}")
    np.save(output_dir / "micrograph.npy", np.asarray(micrograph, dtype=np.float32))
    eligible = maps["eligible"].astype(bool)
    labels = maps["region_id"].astype(np.int32) * eligible.astype(np.int32)
    np.save(output_dir / "region_id.npy", labels)

    region_entries: list[dict[str, Any]] = []
    for rid in sorted(overlays):
        rel = f"overlays/region_{int(rid):04d}.npy"
        np.save(output_dir / rel, overlays[rid])
        region_entries.append(
            {
                "region_id": int(rid),
                "polarity": polarities.get(int(rid), "unset"),
                "overlay": rel,
                "n_sites": int(site_counts.get(int(rid), 0)),
            }
        )

    manifest = {
        "version": 1,
        "constraint_yaml": path_relative_to(yaml_path, output_dir),
        "constraint_hdf5": path_relative_to(hdf5_path, output_dir),
        "micrograph": "micrograph.npy",
        "region_id": "region_id.npy",
        "micrograph_shape": [int(micrograph.shape[0]), int(micrograph.shape[1])],
        "pixel_size_angstrom": pixel_size_angstrom,
        "regions": region_entries,
        "cone_half_angle_deg": payload.get("cone_half_angle_deg"),
        "theta_center_deg": payload.get("theta_center_deg"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _progress(f"Wrote {manifest_path}")
    return manifest_path


def prepare_overlays(
    micrograph_path: str,
    template_path: str,
    yaml_path: str,
    output_dir: str,
    n_sites: int = 4,
    pixel_size_angstrom: float | None = None,
) -> Path:
    """Load data, project the template, paint overlays, and save the package."""
    _progress(f"Loading constraint YAML: {yaml_path}")
    payload, hdf5_path = load_constraint(yaml_path)

    _progress(f"Loading constraint maps: {hdf5_path}")
    maps = load_constraint_maps(hdf5_path)

    _progress(f"Loading template volume: {template_path}")
    volume = load_mrc_volume_numpy(template_path)
    template_width = int(volume.shape[-1])
    _progress(f"  template shape (z, y, x) = {tuple(int(v) for v in volume.shape)}")

    _progress(f"Loading micrograph: {micrograph_path}")
    image, header_px = load_mrc_image(micrograph_path)
    image_shape = (int(image.shape[0]), int(image.shape[1]))
    _progress(f"  micrograph shape (y, x) = {image_shape}")
    pixel_size = pixel_size_angstrom or header_px
    if pixel_size is not None:
        source = "given" if pixel_size_angstrom else "MRC header"
        _progress(f"  pixel size {pixel_size:.4g} Å/px ({source})")

    region_ids = region_ids_from_maps(maps["region_id"])
    if not region_ids:
        raise ValueError("Constraint maps contain no membrane regions (region_id > 0).")
    _progress(f"Found {len(region_ids)} membrane region(s): {region_ids}")

    polarities = {rid: "unset" for rid in region_ids}
    polarities.update(maps["polarities"])
    polarities.update(polarities_from_yaml(payload))
    polarities = {rid: polarities.get(rid, "unset") for rid in region_ids}

    _progress(f"Sampling up to {n_sites} overlay site(s) per membrane...")
    overlays: dict[int, np.ndarray] = {}
    site_counts: dict[int, int] = {}
    for rid in region_ids:
        sites = sample_membrane_sites(
            maps["region_id"],
            maps["signed_distance"],
            maps["psi_center"],
            rid,
            n_sites=n_sites,
        )
        if not sites:
            _progress(f"  membrane {rid}: no eligible pixels (empty overlay)")
            overlays[rid] = np.zeros((*image_shape, 3), dtype=np.float32)
            site_counts[rid] = 0
            continue

        _progress(f"  membrane {rid}: {len(sites)} site(s)")
        painted: list[dict[str, Any]] = []
        for site_index, (cy, cx, psi) in enumerate(sites, start=1):
            _progress(
                f"    site {site_index}/{len(sites)} at "
                f"(y={cy:.1f}, x={cx:.1f}), psi={psi:.1f}° — projecting template..."
            )
            proj_a, proj_b = template_projections(volume, psi)
            painted.append(
                {
                    "proj_a": normalize_display(proj_a),
                    "proj_b": normalize_display(proj_b),
                    "translate_a": overlay_translate(
                        cy, cx, psi, template_width, sign=1.0
                    ),
                    "translate_b": overlay_translate(
                        cy, cx, psi, template_width, sign=-1.0
                    ),
                }
            )
        overlays[rid] = render_region_overlay(image_shape, painted)
        site_counts[rid] = len(sites)

    total_sites = sum(site_counts.values())
    _progress(
        f"Painted {total_sites} overlay site(s) across {len(region_ids)} membrane(s)."
    )
    return write_overlay_package(
        output_dir=Path(output_dir),
        micrograph=normalize_display(image),
        maps=maps,
        payload=payload,
        yaml_path=yaml_path,
        hdf5_path=hdf5_path,
        polarities=polarities,
        overlays=overlays,
        site_counts=site_counts,
        pixel_size_angstrom=pixel_size,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for overlay rendering."""
    parser = argparse.ArgumentParser(
        description=(
            "Project a membrane-protein template onto a micrograph at both "
            "poles and save RGB overlay images for the napari polarity picker."
        )
    )
    parser.add_argument("--micrograph", required=True, help="2D MRC micrograph.")
    parser.add_argument(
        "--template",
        required=True,
        help="3D MRC template (Z = membrane normal).",
    )
    parser.add_argument(
        "--constraint",
        required=True,
        help="YAML sidecar from export_membrane_constraint.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for the overlay package (manifest.json + npy images).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in Angstroms (stored in the manifest; MRC header if omitted).",
    )
    parser.add_argument(
        "--n-sites",
        type=int,
        default=4,
        help="Projection overlays sampled along each membrane.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for Leopard-EM overlay rendering."""
    args = parse_args(argv)
    for path, label in (
        (args.micrograph, "Micrograph"),
        (args.template, "Template"),
        (args.constraint, "Constraint YAML"),
    ):
        if not Path(path).is_file():
            print(f"{label} not found: {path}", file=sys.stderr)
            raise SystemExit(1)
    manifest_path = prepare_overlays(
        micrograph_path=args.micrograph,
        template_path=args.template,
        yaml_path=args.constraint,
        output_dir=args.output,
        n_sites=args.n_sites,
        pixel_size_angstrom=args.pixel_size,
    )
    _progress("Done. Open the overlays in a napari environment with:")
    _progress(
        "  python programs/constrained_search/napari_choose_membrane_polarity.py "
        f"--overlays {Path(args.output)}"
    )
    _progress(f"  (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
