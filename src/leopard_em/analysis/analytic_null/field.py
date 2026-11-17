"""Covariance of a search's null correlation field, estimated from its templates.

For a phase-randomised (or any stationary Gaussian) image, the cross-correlation
CC(x, s) = <I, P_s(. - x)> of the image I with the filtered, normalised template
projections P_s is a Gaussian random field over position x and search point s
(orientation, defocus plane, ...). Its covariance depends only on the projections and
the image power spectrum |I(k)|^2:

    Cov(CC(x1, s1), CC(x2, s2))
        = sum_k |I(k)|^2 P_s1(k) conj(P_s2(k)) exp(2 pi i k (x1 - x2)).

Because the projections live in an h x w box, only image autocorrelation lags within
+-(h, w) enter, so every inner product is exact on a 2h x 2w "lag grid"
(:class:`PowerWeighting`). The functions here estimate the quantities the lattice
model (:mod:`.lattice`) needs, from projections supplied by the search engine:

* :func:`estimate_mean_template`: the mean projection M over the search and the field
  variance after the per-pixel mean is removed (the z-score subtracts <I, M>);
* :func:`estimate_kappa`: the mean squared correlation between random search points at
  one pixel, which sets how the per-pixel standard deviation thins the null tail;
* :func:`sample_lattice`: correlations at one and two grid steps along every search
  axis, at sampled grid points.

They are engine-agnostic: the engine supplies a ``project`` callable mapping a batch
of its search points ("hypotheses", in whatever representation it uses) to their
normalised projections on the lag grid, shape ``(planes, n, Ly, Lx // 2 + 1)``, via
:meth:`PowerWeighting.transform`.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

ProjectFn = Callable[[Any], torch.Tensor]


# pylint: disable=not-callable
class PowerWeighting:
    """Image power spectrum on the 2h x 2w lag grid, for field inner products.

    Parameters
    ----------
    image_dft : torch.Tensor
        RFFT of the searched (preprocessed) image, shape (H, W // 2 + 1).
    image_shape : tuple[int, int]
        Real-space shape (H, W) of the searched image.
    box_shape : tuple[int, int]
        Real-space shape (h, w) of the template projections.
    white : bool
        Replace the image power by its mean (a perfectly white image). Default False.
    """

    def __init__(
        self,
        image_dft: torch.Tensor,
        image_shape: tuple[int, int],
        box_shape: tuple[int, int],
        white: bool = False,
    ):
        height, width = image_shape
        h, w = box_shape
        power = image_dft.detach().to("cpu").abs().double() ** 2
        power[0, 0] = 0.0
        if white:
            weight = torch.full_like(power, 2.0)
            weight[:, 0] = 1.0
            weight[:, -1] = 1.0
            power = torch.full_like(power, float((power * weight).sum() / weight.sum()))
            power[0, 0] = 0.0
        autocorrelation = torch.fft.irfftn(power, s=(height, width))
        self.lag_shape = (2 * h, 2 * w)
        rows_lag = torch.cat([torch.arange(0, h), torch.arange(-h, 0)]) % height
        cols_lag = torch.cat([torch.arange(0, w), torch.arange(-w, 0)]) % width
        spectrum = torch.fft.rfftn(autocorrelation[rows_lag][:, cols_lag]).real
        spectrum = spectrum / spectrum.mean()
        column = torch.full((self.lag_shape[1] // 2 + 1,), 2.0, dtype=torch.float64)
        column[0] = 1.0
        column[-1] = 1.0
        self.weights = (spectrum * column[None, :]).float()
        self.negative_fraction = float((spectrum < 0).double().mean())
        self.ky = torch.fft.fftfreq(self.lag_shape[0]).float()[:, None]
        self.kx = torch.fft.rfftfreq(self.lag_shape[1]).float()[None, :]

    def to(self, device: torch.device | str) -> "PowerWeighting":
        """Move the weights to ``device`` (in place); returns self."""
        self.weights = self.weights.to(device)
        self.ky = self.ky.to(device)
        self.kx = self.kx.to(device)
        return self

    def transform(self, projections: torch.Tensor) -> torch.Tensor:
        """RFFT of real-space projections (..., h, w) on the lag grid, DC removed."""
        out = torch.fft.rfftn(projections, s=self.lag_shape, dim=(-2, -1))
        out[..., 0, 0] = 0
        return out

    def inner(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Power-weighted inner product over the last two (lag-grid) dimensions."""
        return (self.weights * (a * b.conj()).real).sum(dim=(-2, -1))

    def shift_correlation(self, q: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Covariance of the field with itself shifted by (dy, dx) pixels."""
        phase = torch.cos(2 * math.pi * (self.ky * dy + self.kx * dx))
        return (self.weights * q.abs() ** 2 * phase).sum(dim=(-2, -1))


@dataclass
class MeanTemplate:
    """Mean projection over the search and the field variance around it.

    Attributes
    ----------
    mean : torch.Tensor
        Mean projection M on the lag grid (over orientations and planes).
    mean_power : float
        Mean squared norm of the projections, mean_s ||P_s||^2.
    sigma2 : float
        Mean squared norm after the per-pixel mean is removed, mean_s ||P_s - M||^2.
    rho_bar : float
        Mean-template share ||M||^2 / mean_s ||P_s||^2 (unbiased for finite samples).
        A null map's Var_x(corr_mean) / mean(corr_std)^2 is about rho_bar /
        (1 - rho_bar).
    n_samples : int
        Number of projections (orientations x planes) averaged.
    """

    mean: torch.Tensor
    mean_power: float
    sigma2: float
    rho_bar: float
    n_samples: int


def estimate_mean_template(
    project: ProjectFn, weighting: PowerWeighting, batches: Iterable[Any]
) -> MeanTemplate:
    """Estimate the mean projection from batches of randomly sampled search points.

    Parameters
    ----------
    project : ProjectFn
        Maps a batch of hypotheses to projections (planes, n, Ly, Lx // 2 + 1).
    weighting : PowerWeighting
        Lag-grid power weighting of the image.
    batches : Iterable[Any]
        Batches of randomly sampled hypotheses (orientations); every plane is used.

    Returns
    -------
    MeanTemplate
    """
    total = None
    power = 0.0
    group_power = 0.0
    n = 0
    n_planes = 1
    for batch in batches:
        q = project(batch)
        n_planes = int(q.shape[0])
        s = q.sum(dim=(0, 1))
        total = s if total is None else total + s
        power += float(weighting.inner(q, q).sum())
        group = q.mean(dim=0)  # orientation groups over the planes
        group_power += float(weighting.inner(group, group).sum())
        n += int(q.shape[0] * q.shape[1])
    if total is None or n == 0:
        raise ValueError("No samples for the mean template.")
    orientations = n // n_planes
    if orientations < 2:
        raise ValueError("Need at least two sampled orientations.")
    mean = total / n
    mean_power = power / n
    m2 = float(weighting.inner(mean, mean))
    m2_unbiased = (orientations * m2 - group_power / orientations) / (orientations - 1)
    return MeanTemplate(
        mean=mean,
        mean_power=mean_power,
        sigma2=mean_power - m2,
        rho_bar=m2_unbiased / mean_power,
        n_samples=n,
    )


def estimate_kappa(
    project: ProjectFn,
    weighting: PowerWeighting,
    mean: torch.Tensor,
    pairs: Iterable[tuple[Any, Any]],
) -> tuple[float, float]:
    """Mean squared correlation of random pairs of search points at one pixel.

    kappa = E[rho(s, s')^2] over independent search points s, s' (all planes), with
    the mean projection removed. The per-pixel standard deviation of a search over
    many correlated points fluctuates with relative standard deviation sqrt(kappa / 2),
    and a null peak of height z raises its own pixel's standard deviation by a factor
    about sqrt(1 + kappa (z^2 - 1)).

    Parameters
    ----------
    project : ProjectFn
        Maps a batch of hypotheses to projections (planes, n, Ly, Lx // 2 + 1).
    weighting : PowerWeighting
        Lag-grid power weighting of the image.
    mean : torch.Tensor
        Mean projection (:func:`estimate_mean_template`).
    pairs : Iterable[tuple[Any, Any]]
        Pairs of independently sampled hypothesis batches.

    Returns
    -------
    tuple[float, float]
        kappa and its standard error over the pairs.
    """
    values = []
    for first, second in pairs:
        qa = (project(first) - mean).reshape(-1, *mean.shape)
        qb = (project(second) - mean).reshape(-1, *mean.shape)
        va = weighting.inner(qa, qa)
        vb = weighting.inner(qb, qb)
        cov = torch.stack([weighting.inner(qa[i : i + 1], qb) for i in range(len(qa))])
        corr = cov / torch.sqrt(va[:, None] * vb[None, :])
        values.append(float((corr**2).mean()))
    if len(values) < 2:
        raise ValueError("Need at least two pairs of batches for kappa.")
    arr = np.asarray(values)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


@dataclass
class AxisNeighbours:
    """Neighbours of a batch of grid points along one search axis.

    Attributes
    ----------
    name : str
        Axis name (e.g. "psi", "theta", "phi").
    steps : tuple[Any, Any, Any, Any]
        Hypothesis batches at +1, -1, +2 and -2 grid steps along the axis.
    valid_plus, valid_minus : np.ndarray | None
        Whether the +1 / -1 neighbour lies inside the searched range, shape (n,). None
        means always (a periodic or unbounded axis). Points with one valid neighbour
        are treated as one-sided along the axis; with none, the axis is skipped.
    """

    name: str
    steps: tuple[Any, Any, Any, Any]
    valid_plus: np.ndarray | None = None
    valid_minus: np.ndarray | None = None


NeighbourFn = Callable[[Any], Sequence[AxisNeighbours]]


@dataclass
class LatticeSamples:
    """Field correlations at one and two grid steps, at sampled grid points.

    Attributes
    ----------
    variance : np.ndarray
        Field variance ||P_s - M||^2 of each sampled point and plane, shape (k, planes).
    axes : dict[str, np.ndarray]
        Per axis, shape (k, planes, 3): correlation at one step, at two steps, and the
        number of in-range neighbours (2 = two-sided, 1 = one-sided, 0 = skip).
        Position axes "x" and "y" are always included.
    plane_correlation : np.ndarray | None
        Correlation between the planes of each sampled point, shape
        (k, planes, planes); None with a single plane.
    """

    variance: np.ndarray
    axes: dict[str, np.ndarray]
    plane_correlation: np.ndarray | None

    def median_correlations(self) -> dict[str, list[float]]:
        """Median one- and two-step correlation per axis (and adjacent planes)."""
        out = {
            name: [float(np.median(arr[..., 0])), float(np.median(arr[..., 1]))]
            for name, arr in self.axes.items()
        }
        if self.plane_correlation is not None:
            n = self.plane_correlation.shape[1]
            adjacent = [self.plane_correlation[:, j, j + 1] for j in range(n - 1)]
            out["defocus"] = [float(np.median(adjacent)), float("nan")]
        return out


def _correlation(
    weighting: PowerWeighting, q0: torch.Tensor, v0: torch.Tensor, q1: torch.Tensor
) -> np.ndarray:
    v1 = weighting.inner(q1, q1)
    corr = weighting.inner(q0, q1) / torch.sqrt(v0 * v1)
    return corr.T.cpu().double().numpy()  # (n, planes)


def sample_lattice(  # pylint: disable=too-many-locals
    project: ProjectFn,
    weighting: PowerWeighting,
    mean: torch.Tensor,
    batches: Iterable[Any],
    neighbours: NeighbourFn,
) -> LatticeSamples:
    """Correlations at one and two grid steps along every axis, at sampled points.

    For each sampled grid point and plane: the field variance, the correlations with
    its neighbours one and two pixels away in x and y, one and two grid steps along
    each axis that ``neighbours`` returns, and the correlation between its planes.
    The two-step correlation is taken as the mean of the +2 and -2 neighbours (equal
    to the correlation of the -1 and +1 neighbours for a locally stationary field).

    Parameters
    ----------
    project : ProjectFn
        Maps a batch of hypotheses to projections (planes, n, Ly, Lx // 2 + 1).
    weighting : PowerWeighting
        Lag-grid power weighting of the image.
    mean : torch.Tensor
        Mean projection (:func:`estimate_mean_template`).
    batches : Iterable[Any]
        Batches of sampled grid points.
    neighbours : NeighbourFn
        Returns the neighbours of a batch along each orientation axis.

    Returns
    -------
    LatticeSamples
    """
    variances: list[np.ndarray] = []
    axes: dict[str, list[np.ndarray]] = {"x": [], "y": []}
    planes: list[np.ndarray] = []
    for batch in batches:
        q0 = project(batch) - mean  # (planes, n, ...)
        v0 = weighting.inner(q0, q0)  # (planes, n)
        variances.append(v0.T.cpu().double().numpy())
        for name, (dy, dx) in (("x", (0, 1)), ("y", (1, 0))):
            r = [
                (weighting.shift_correlation(q0, m * dy, m * dx) / v0)
                .T.cpu()
                .double()
                .numpy()
                for m in (1, 2)
            ]
            axes[name].append(np.stack([r[0], r[1], np.full_like(r[0], 2.0)], -1))
        for axis in neighbours(batch):
            cs = [
                _correlation(weighting, q0, v0, project(h) - mean) for h in axis.steps
            ]
            n = cs[0].shape[0]
            vp = (np.ones(n, bool) if axis.valid_plus is None else axis.valid_plus)[
                :, None
            ]
            vm = (np.ones(n, bool) if axis.valid_minus is None else axis.valid_minus)[
                :, None
            ]
            r1 = np.where(vp & vm, 0.5 * (cs[0] + cs[1]), np.where(vp, cs[0], cs[1]))
            r2 = 0.5 * (cs[2] + cs[3])
            sided = np.broadcast_to(vp.astype(float) + vm.astype(float), r1.shape)
            axes.setdefault(axis.name, []).append(np.stack([r1, r2, sided], -1))
        n_planes = int(q0.shape[0])
        if n_planes > 1:
            norm = torch.sqrt(v0)
            corr = torch.stack(
                [weighting.inner(q0[j : j + 1], q0) for j in range(n_planes)]
            )  # (planes, planes, n)
            corr = corr / (norm[:, None] * norm[None, :])
            planes.append(corr.permute(2, 0, 1).cpu().double().numpy())
    if not variances:
        raise ValueError("No lattice samples.")
    return LatticeSamples(
        variance=np.concatenate(variances),
        axes={name: np.concatenate(arrs) for name, arrs in axes.items()},
        plane_correlation=np.concatenate(planes) if planes else None,
    )
