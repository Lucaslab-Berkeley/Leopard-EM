"""Data class and helper functions for saving/loading peak-inspection results.

Peak inspection produces a large local score tensor per particle (cross-correlation
maps or FRC spectra over a grid of orientation/defocus/pixel-size hypotheses). This
module bundles that tensor with the axis metadata needed to interpret it into a
self-describing ``.npz`` file, so downstream analysis (e.g. notebooks) can reload the
array without knowing how the run was configured.


Shape of the cross-correlation tensor ``result.scores``
-------------------------------------------------------

For the cross-correlation mode, have a 6-D tensor with last two dimensions corresponding
to the spatial dimensions of the valid correlation map.
```
scores.shape = (N, n_px, n_def, n_orient, H, W)
               |   |     |      |         |  |
               |   |     |      |         |  +-- valid same-size CC map (x)
               |   |     |      |         +----- valid same-size CC map (y)
               |   |     |      +--------------- local Euler *offsets* (phi,theta,psi)
               |   |     +---------------------- relative defocus search index
               |   +---------------------------- relative pixel-size search index
               +-------------------------------- particle (row in stack / CSV)
```

For the FRC mode, have a 5-D tensor where the last dimension corresponds to the
FRC frequency bins.
```
scores.shape = (N, n_px, n_def, n_orient, num_freq)
               |   |     |      |         |
               |   |     |      |         +----- FRC frequency bins
               |   |     |      +--------------- local Euler *offsets* (phi,theta,psi)
               |   |     +---------------------- relative defocus search index
               |   +---------------------------- relative pixel-size search index
               +-------------------------------- particle (row in stack / CSV)
```

Per-frame inspection
--------------------

The per-frame manager scores each movie frame independently, inserting a ``frame``
axis immediately after ``particle`` (``per_frame=True``). The cross-correlation tensor
is then 7-D ``(N, T, n_px, n_def, n_orient, H, W)`` and the FRC tensor is 6-D
``(N, T, n_px, n_def, n_orient, num_freq)``, where ``T`` is the number of frames.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

# Bump when the on-disk layout changes in a backwards-incompatible way.
INSPECTION_FORMAT_VERSION = 1

# Axis labels for the stored tensors. These describe what each dimension of the
# main array means so downstream analysis (notebooks) does not have to hard-code
# the 6-D / 5-D layout.
CROSS_CORRELATION_AXES = (
    "particle",
    "pixel_size",
    "defocus",
    "orientation",
    "y",
    "x",
)
FRC_AXES = (
    "particle",
    "pixel_size",
    "defocus",
    "orientation",
    "frequency",
)

# Per-frame variants insert a ``frame`` axis right after ``particle``.
CROSS_CORRELATION_FRAME_AXES = (
    "particle",
    "frame",
    "pixel_size",
    "defocus",
    "orientation",
    "y",
    "x",
)
FRC_FRAME_AXES = (
    "particle",
    "frame",
    "pixel_size",
    "defocus",
    "orientation",
    "frequency",
)


@dataclass
class InspectionResult:
    """Self-describing container for a saved peak-inspection run.

    Attributes
    ----------
    output_mode : str
        Either ``"cross_correlation"`` or ``"frc"``.
    scores : np.ndarray
        The main score tensor. Shape is ``(N, n_px, n_defocus, n_orient, H, W)`` for
        cross-correlation mode and ``(N, n_px, n_defocus, n_orient, n_freq)`` for FRC
        mode. See :attr:`axes` for per-dimension labels.
    axes : tuple[str, ...]
        Label for each dimension of :attr:`scores`.
    euler_angle_offsets : np.ndarray
        ZYZ orientation offsets searched per particle, shape ``(n_orient, 3)``. Indexes
        the ``orientation`` axis of :attr:`scores`.
    defocus_offsets : np.ndarray
        Relative defocus offsets (Angstroms), shape ``(n_defocus,)``. Indexes the
        ``defocus`` axis.
    pixel_size_offsets : np.ndarray
        Relative pixel-size offsets, shape ``(n_px,)``. Indexes the ``pixel_size`` axis.
    base_euler_angles : np.ndarray
        Per-particle base ZYZ angles the offsets are relative to, shape ``(N, 3)``.
    base_defocus : np.ndarray
        Per-particle base astigmatic defocus the ``defocus_offsets`` are relative to,
        shape ``(N, 3)`` as ``(defocus_u, defocus_v, defocus_angle)``.
    particle_index : np.ndarray | None
        Global particle index for each row of the ``particle`` axis, shape ``(N,)``, or
        ``None`` if the source dataframe had no ``particle_index``
        column. Maps tensor rows back to the particle stack dataframe.
    frequency_bins : np.ndarray | None
        FRC frequency bins, shape ``(n_freq,)``, in FRC mode; ``None`` otherwise.
    frame_index : np.ndarray | None
        Movie frame index for each entry of the ``frame`` axis, shape ``(T,)``, when the
        result was produced by per-frame inspection; ``None`` otherwise.
    metadata : dict[str, Any]
        Free-form metadata stored alongside the arrays (includes the format version and
        any caller-supplied ``extra_metadata``).
    """

    output_mode: Literal["cross_correlation", "frc"]
    scores: np.ndarray
    axes: tuple[str, ...]
    euler_angle_offsets: np.ndarray
    defocus_offsets: np.ndarray
    pixel_size_offsets: np.ndarray
    base_euler_angles: np.ndarray
    base_defocus: np.ndarray
    particle_index: np.ndarray | None
    frequency_bins: np.ndarray | None
    frame_index: np.ndarray | None
    metadata: dict[str, Any]


def _to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a detached CPU numpy view of a tensor (passthrough for ndarrays)."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def save_inspection_result(
    output_path: str | Path,
    *,
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    output_mode: Literal["cross_correlation", "frc"],
    euler_angle_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    base_euler_angles: torch.Tensor,
    base_defocus: torch.Tensor,
    particle_index: torch.Tensor | np.ndarray | None = None,
    frame_index: torch.Tensor | np.ndarray | None = None,
    per_frame: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a peak-inspection result to a self-describing ``.npz`` file.

    Parameters
    ----------
    output_path : str | Path
        Destination path. A ``.npz`` suffix is appended if not present.
    result : torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        Output of the inspect backend. A tensor in ``"cross_correlation"`` mode, or
        ``(frc_tensor, frequency_bins)`` in ``"frc"`` mode. When ``per_frame`` is True
        the score tensor carries an extra ``frame`` axis after ``particle``.
    output_mode : Literal["cross_correlation", "frc"]
        Score mode used to produce ``result``.
    euler_angle_offsets : torch.Tensor
        Orientation offsets searched, shape ``(n_orient, 3)``.
    defocus_offsets : torch.Tensor
        Relative defocus offsets searched, shape ``(n_defocus,)``.
    pixel_size_offsets : torch.Tensor
        Relative pixel-size offsets searched, shape ``(n_px,)``.
    base_euler_angles : torch.Tensor
        Per-particle base ZYZ angles the offsets are relative to, shape ``(N, 3)``.
    base_defocus : torch.Tensor
        Per-particle base astigmatic defocus the offsets are relative to, shape
        ``(N, 3)`` as ``(defocus_u, defocus_v, defocus_angle)``.
    particle_index : torch.Tensor | np.ndarray | None, optional
        Global particle index for each tensor row, shape ``(N,)``.
    frame_index : torch.Tensor | np.ndarray | None, optional
        Movie frame index for each entry of the ``frame`` axis, shape ``(T,)``. Only
        meaningful when ``per_frame`` is True.
    per_frame : bool, optional
        If True, the score tensor carries a ``frame`` axis after ``particle`` and the
        stored axis labels use the per-frame variants.
    extra_metadata : dict[str, Any] | None, optional
        Additional JSON-serializable metadata to store alongside the arrays.

    Returns
    -------
    Path
        The path the result was written to (with ``.npz`` suffix).
    """
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axes: tuple[str, ...]

    if output_mode == "frc":
        if not (isinstance(result, tuple) and len(result) == 2):
            raise ValueError(
                "FRC mode expects a (frc_tensor, frequency_bins) tuple result."
            )
        scores_tensor, frequency_bins = result
        axes = FRC_FRAME_AXES if per_frame else FRC_AXES
    elif output_mode == "cross_correlation":
        if not isinstance(result, torch.Tensor):
            raise ValueError("Cross-correlation mode expects a single tensor result.")
        scores_tensor = result
        frequency_bins = None
        axes = CROSS_CORRELATION_FRAME_AXES if per_frame else CROSS_CORRELATION_AXES
    else:
        raise ValueError(f"Unknown output_mode: {output_mode!r}")

    arrays: dict[str, np.ndarray] = {
        "scores": _to_numpy(scores_tensor),
        "euler_angle_offsets": _to_numpy(euler_angle_offsets),
        "defocus_offsets": _to_numpy(defocus_offsets),
        "pixel_size_offsets": _to_numpy(pixel_size_offsets),
        "base_euler_angles": _to_numpy(base_euler_angles),
        "base_defocus": _to_numpy(base_defocus),
    }

    if frequency_bins is not None:
        arrays["frequency_bins"] = _to_numpy(frequency_bins)
    if particle_index is not None:
        arrays["particle_index"] = _to_numpy(particle_index)
    if per_frame and frame_index is not None:
        arrays["frame_index"] = _to_numpy(frame_index)

    metadata: dict[str, Any] = {
        "format_version": INSPECTION_FORMAT_VERSION,
        "output_mode": output_mode,
        "axes": list(axes),
        "per_frame": per_frame,
    }

    # Store metadata as a JSON string in a 0-d array so it survives the round trip.
    if extra_metadata:
        metadata.update(extra_metadata)
    arrays["metadata_json"] = np.array(json.dumps(metadata))

    np.savez(output_path, **arrays)

    return output_path


def load_inspection_result(path: str | Path) -> InspectionResult:
    """Load a ``.npz`` written by :func:`save_inspection_result`.

    Parameters
    ----------
    path : str | Path
        Path to the ``.npz`` file.

    Returns
    -------
    InspectionResult
        Self-describing container with the score tensor and its axis metadata.
    """
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        output_mode = metadata["output_mode"]
        return InspectionResult(
            output_mode=output_mode,
            scores=data["scores"],
            axes=tuple(metadata["axes"]),
            euler_angle_offsets=data["euler_angle_offsets"],
            defocus_offsets=data["defocus_offsets"],
            pixel_size_offsets=data["pixel_size_offsets"],
            base_euler_angles=data["base_euler_angles"],
            base_defocus=data["base_defocus"],
            particle_index=(
                data["particle_index"] if "particle_index" in data else None
            ),
            frequency_bins=(
                data["frequency_bins"] if "frequency_bins" in data else None
            ),
            frame_index=(data["frame_index"] if "frame_index" in data else None),
            metadata=metadata,
        )
