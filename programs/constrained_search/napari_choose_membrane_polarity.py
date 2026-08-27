r"""Napari viewer to pick membrane-protein polarity from pre-rendered overlays.

This script does **not** import leopard_em or torch. Render overlays first
with ``prepare_membrane_polarity_overlays.py`` in the Leopard-EM environment,
then open this viewer in a napari-only environment.

Magenta (A) is ``psi_center`` on the +normal leaflet. Cyan (B) is
``psi_center + 180`` on the opposite leaflet.

``positive`` searches ``psi_center`` only; ``negative`` searches
``psi_center + 180``; ``both`` searches both Euler boxes.

Example
-------
::

    python napari_choose_membrane_polarity.py \\
        --overlays polarity_preview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_NAPARI_INSTALL_MESSAGE = (
    "This program needs napari, numpy, h5py, and PyYAML.\n"
    "Install with:  pip install napari numpy h5py pyyaml"
)
_ANGLE_DECIMALS = 4
_MANIFEST_NAME = "manifest.json"


def _progress(message: str) -> None:
    """Print a status line and flush so batch runs show progress immediately."""
    print(message, flush=True)


def dump_membrane_yaml(payload: dict[str, Any]) -> str:
    """Serialize a membrane-constraint sidecar, preserving Euler-box fields."""
    keys = (
        "cone_half_angle_deg",
        "theta_center_deg",
        "phi_min",
        "phi_max",
        "psi_step",
        "theta_step",
        "base_grid_method",
        "polarity",
        "spatial_constraint_path",
        "micrograph_path",
    )
    lines: list[str] = []
    for key in keys:
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if isinstance(value, float):
            text = f"{round(float(value), _ANGLE_DECIMALS):.4f}".rstrip("0").rstrip(".")
        elif isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (int, str)):
            text = json.dumps(value) if isinstance(value, str) else str(value)
        else:
            text = json.dumps(str(value))
        lines.append(f"{key}: {text}")
    regions = payload.get("regions") or []
    lines.append("regions:")
    for index, region in enumerate(regions, start=1):
        rid = int(region.get("region_id", index))
        polarity = str(region.get("polarity", payload.get("polarity", "unset")))
        lines.append(f"  - region_id: {rid}")
        lines.append(f"    polarity: {polarity}")
    return "\n".join(lines) + "\n"


def write_polarities(
    yaml_path: str,
    hdf5_path: str,
    payload: dict[str, Any],
    polarities: dict[int, str],
) -> None:
    """Write per-region polarity into the YAML sidecar and HDF5 region table."""
    import h5py

    default = str(payload.get("polarity", "unset"))
    regions = list(payload.get("regions") or [])
    known = {
        int(region.get("region_id", i)): dict(region)
        for i, region in enumerate(regions, start=1)
    }
    for rid, polarity in polarities.items():
        entry = known.get(int(rid), {"region_id": int(rid)})
        entry["region_id"] = int(rid)
        entry["polarity"] = polarity
        known[int(rid)] = entry
    payload["regions"] = [known[rid] for rid in sorted(known)]
    if polarities:
        unique = set(polarities.values())
        payload["polarity"] = unique.pop() if len(unique) == 1 else default
    Path(yaml_path).write_text(dump_membrane_yaml(payload), encoding="utf-8")

    with h5py.File(hdf5_path, "a") as handle:
        if "regions" not in handle:
            handle.create_group("regions")
        regions_group = handle["regions"]
        for rid, polarity in polarities.items():
            name = f"{int(rid):04d}"
            if name in regions_group:
                grp = regions_group[name]
            else:
                grp = regions_group.create_group(name)
            grp.attrs["region_id"] = int(rid)
            grp.attrs["polarity"] = str(polarity)


def resolve_overlay_dir(overlays: str) -> Path:
    """Accept a package directory or a path to ``manifest.json``."""
    path = Path(overlays)
    if path.is_file() and path.name == _MANIFEST_NAME:
        return path.parent
    if path.is_dir() and (path / _MANIFEST_NAME).is_file():
        return path
    raise FileNotFoundError(
        f"No {_MANIFEST_NAME} under {path}. Run "
        "prepare_membrane_polarity_overlays.py first."
    )


def resolve_manifest_path(path_str: str, overlay_dir: Path) -> Path:
    """Resolve a constraint path stored in the overlay manifest.

    New packages store paths relative to the overlay directory so the same
    tree works when the repo is mounted at a different absolute path.
    Older absolute paths are remapped by filename when the original host
    path is missing.
    """
    raw = Path(path_str)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((overlay_dir / raw).resolve())

    name = raw.name
    candidates.extend(
        [
            (overlay_dir / name).resolve(),
            (overlay_dir.parent / "configs" / name).resolve(),
            (overlay_dir.parent / name).resolve(),
        ]
    )
    # Absolute path from another machine: keep trailing repo-relative suffix.
    parts = raw.parts
    for marker in ("membrane_tests", "configs"):
        if marker in parts:
            idx = parts.index(marker)
            suffix = Path(*parts[idx:])
            candidates.append((overlay_dir.parent.parent / suffix).resolve())
            candidates.append((overlay_dir.parent / suffix).resolve())
            break

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    tried = "\n  ".join(str(path) for path in seen)
    raise FileNotFoundError(
        f"Could not resolve constraint path {path_str!r} relative to "
        f"{overlay_dir}. Tried:\n  {tried}"
    )


def load_overlay_package(overlay_dir: Path) -> dict[str, Any]:
    """Load the overlay package written by the Leopard-EM renderer."""
    overlay_dir = overlay_dir.resolve()
    manifest_path = overlay_dir / _MANIFEST_NAME
    _progress(f"Loading overlay package: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    yaml_path = resolve_manifest_path(str(manifest["constraint_yaml"]), overlay_dir)
    hdf5_path = resolve_manifest_path(str(manifest["constraint_hdf5"]), overlay_dir)
    _progress(f"Loading constraint YAML: {yaml_path}")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    micrograph = np.load(overlay_dir / manifest["micrograph"])
    region_id = np.load(overlay_dir / manifest["region_id"])
    region_overlays: dict[int, np.ndarray] = {}
    polarities: dict[int, str] = {}
    ids: list[int] = []
    for entry in manifest.get("regions") or []:
        rid = int(entry["region_id"])
        ids.append(rid)
        polarities[rid] = str(entry.get("polarity", "unset"))
        overlay_path = overlay_dir / entry["overlay"]
        _progress(f"  membrane {rid}: {overlay_path.name}")
        region_overlays[rid] = np.load(overlay_path)

    if not ids:
        raise ValueError(f"{manifest_path} lists no membrane regions.")
    return {
        "payload": payload,
        "yaml_path": str(yaml_path),
        "hdf5_path": str(hdf5_path),
        "micrograph": micrograph,
        "region_id": region_id,
        "ids": ids,
        "polarities": polarities,
        "overlays": region_overlays,
    }


def build_viewer(package: dict[str, Any]) -> None:
    """Open napari with pre-rendered opposite-pole overlays."""
    try:
        import napari
        from qtpy.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QListWidget,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit(_NAPARI_INSTALL_MESSAGE) from exc

    payload = package["payload"]
    hdf5_path = package["hdf5_path"]
    yaml_path = package["yaml_path"]
    ids: list[int] = package["ids"]
    polarities: dict[int, str] = dict(package["polarities"])
    overlays: dict[int, np.ndarray] = package["overlays"]
    image = package["micrograph"]
    labels = package["region_id"]

    _progress("Opening napari...")
    viewer = napari.Viewer(title="Membrane polarity")
    viewer.add_image(
        image,
        name="micrograph",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        blending="translucent",
    )
    viewer.add_labels(labels.astype(np.int32), name="membranes", opacity=0.25)

    overlay_layers: dict[str, Any] = {}

    def clear_overlays() -> None:
        for name in list(overlay_layers):
            layer = overlay_layers.pop(name)
            if layer in viewer.layers:
                viewer.layers.remove(layer)

    def current_region_id() -> int | None:
        row = region_list.currentRow()
        if row < 0 or row >= len(ids):
            return None
        return ids[row]

    def refresh_overlays() -> None:
        clear_overlays()
        rid = current_region_id()
        if rid is None:
            return
        rgb = overlays.get(rid)
        if rgb is None or not np.any(rgb):
            preview.setPlainText(f"No overlay image for membrane {rid}.")
            return
        layer = viewer.add_image(
            rgb,
            name=f"poles_region_{rid}",
            rgb=True,
            blending="additive",
            opacity=0.9,
        )
        overlay_layers[layer.name] = layer
        polarity = polarities.get(rid, "unset")
        preview.setPlainText(
            f"Membrane {rid}  polarity={polarity}\n"
            "Magenta (A) is psi_center on the +normal leaflet.\n"
            "Cyan (B) is psi_center+180 on the opposite leaflet.\n"
            "This pole (A) = positive; Other pole (B) = negative; Either = both."
        )

    def refresh_region_list() -> None:
        current = region_list.currentRow()
        region_list.blockSignals(True)
        region_list.clear()
        for rid in ids:
            region_list.addItem(f"{rid}  {polarities.get(rid, 'unset')}")
        if ids:
            region_list.setCurrentRow(min(max(current, 0), len(ids) - 1))
        region_list.blockSignals(False)

    def set_polarity(label: str) -> None:
        rid = current_region_id()
        if rid is None:
            preview.setPlainText("Select a membrane first.")
            return
        polarities[rid] = label
        refresh_region_list()
        refresh_overlays()

    def on_save() -> None:
        write_polarities(yaml_path, hdf5_path, payload, polarities)
        preview.setPlainText(f"Saved polarity to\n{yaml_path}\nand {hdf5_path}")

    panel = QWidget()
    layout = QVBoxLayout()
    panel.setLayout(layout)
    layout.addWidget(QLabel("Membranes"))
    region_list = QListWidget()
    region_list.setMinimumHeight(90)
    layout.addWidget(region_list)
    buttons = QHBoxLayout()
    button_a = QPushButton("This pole (A)")
    button_b = QPushButton("Other pole (B)")
    button_both = QPushButton("Either")
    buttons.addWidget(button_a)
    buttons.addWidget(button_b)
    buttons.addWidget(button_both)
    layout.addLayout(buttons)
    save_button = QPushButton("Save polarity")
    layout.addWidget(save_button)
    preview = QTextEdit()
    preview.setReadOnly(True)
    layout.addWidget(preview)
    viewer.window.add_dock_widget(panel, name="Membrane polarity", area="right")

    button_a.clicked.connect(lambda: set_polarity("positive"))
    button_b.clicked.connect(lambda: set_polarity("negative"))
    button_both.clicked.connect(lambda: set_polarity("both"))
    save_button.clicked.connect(on_save)
    region_list.currentRowChanged.connect(lambda _: refresh_overlays())

    refresh_region_list()
    if ids:
        region_list.setCurrentRow(0)
        refresh_overlays()
    napari.run()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the polarity viewer."""
    parser = argparse.ArgumentParser(
        description=(
            "View pre-rendered opposite-pole overlays and save polarity "
            "(positive / negative / both) per membrane region."
        )
    )
    parser.add_argument(
        "--overlays",
        required=True,
        help="Directory from prepare_membrane_polarity_overlays.py (or manifest.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the napari membrane-polarity viewer."""
    args = parse_args(argv)
    overlay_dir = resolve_overlay_dir(args.overlays)
    package = load_overlay_package(overlay_dir)
    build_viewer(package)


if __name__ == "__main__":
    main()
