"""Null calibration of a match_template search: thresholds from null searches.

:class:`NullCalibrationManager` sets the detection threshold of a match_template
configuration for a target number of false positives per micrograph, two ways:

* **Phase-randomised null** (:meth:`NullCalibrationManager.run_phase_randomized_null`):
  search phase-randomised copies of the micrograph with exactly the same
  configuration and count their distinct peaks.
* **Analytic null** (:meth:`NullCalibrationManager.predict_analytic_null`): predict
  the null peak rate from the template, filters, grids and the micrograph's power
  spectrum, without a search.
* **Analytic refined null** (:meth:`NullCalibrationManager.
  predict_analytic_refined_null`): with a refine_template configuration and a
  candidate threshold, predict the null peak rate after the search's candidates are
  refined, for a threshold on refined z-scores.

See :mod:`leopard_em.analysis.empirical_null` and
:mod:`leopard_em.analysis.analytic_null`.
"""

import json
import os
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import torch
import yaml  # type: ignore
from pydantic import BaseModel, ConfigDict, Field, model_validator

from leopard_em.analysis.analytic_null import (
    AnalyticNullPrediction,
    RefinedNullPrediction,
    predict_match_template_null,
    predict_refine_template_null,
    refine_window,
)
from leopard_em.analysis.empirical_null import (
    DistinctPeaks,
    NullPeaks,
    distinct_peaks,
    valid_region,
)
from leopard_em.analysis.match_template_peaks import MatchTemplatePeaks
from leopard_em.analysis.zscore_metric import gaussian_noise_zscore_cutoff
from leopard_em.pydantic_models.custom_types import BaseModel2DTM
from leopard_em.pydantic_models.managers.match_template_manager import (
    MatchTemplateManager,
)
from leopard_em.pydantic_models.results.match_template_result import (
    _TENSOR_NAMES as RESULT_NAMES,
)
from leopard_em.utils.data_io import load_mrc_image
from leopard_em.utils.phase_randomization import phase_randomize_image


def peaks_at(result: Any, peaks: DistinctPeaks) -> MatchTemplatePeaks:
    """The match_template statistics of a result at the given distinct peaks.

    Same fields as :meth:`MatchTemplateResult.locate_peaks`, at positions found by
    :func:`~leopard_em.analysis.empirical_null.distinct_peaks` instead, which takes
    linear time; the pairwise peak filter of ``locate_peaks`` is too slow for the
    tens of thousands of peaks a low floor gives on a 4k null map.

    Parameters
    ----------
    result : MatchTemplateResult
        The search's result maps.
    peaks : DistinctPeaks
        Positions in those maps.

    Returns
    -------
    MatchTemplatePeaks
    """
    device = result.scaled_mip.device
    pos_y = torch.as_tensor(peaks.pos_y, dtype=torch.long, device=device)
    pos_x = torch.as_tensor(peaks.pos_x, dtype=torch.long, device=device)
    return MatchTemplatePeaks(
        pos_y=pos_y,
        pos_x=pos_x,
        mip=result.mip[pos_y, pos_x],
        scaled_mip=result.scaled_mip[pos_y, pos_x],
        psi=result.orientation_psi[pos_y, pos_x],
        theta=result.orientation_theta[pos_y, pos_x],
        phi=result.orientation_phi[pos_y, pos_x],
        relative_defocus=result.relative_defocus[pos_y, pos_x],
        correlation_mean=result.correlation_average[pos_y, pos_x],
        correlation_variance=result.correlation_variance[pos_y, pos_x],
        total_correlations=result.total_projections,
    )


class AnalyticRefineConfig(BaseModel):
    """Settings of the analytic refined null (used with a refine_template config).

    Attributes
    ----------
    n_candidates : int
        Grid points sampled as null candidates, spread over the defocus planes.
        Default 8 (about 35 s each on a GPU for a 77-offset refine).
    n_draws : int
        Gaussian draws per candidate. Default 200,000.
    shift_radius : int
        Positions refinement scores about each candidate, +-pixels. Default 1 (+-2
        gives the same gains on xenon_131, at ~8x the cost).
    condition_radius : int
        Positions at which the candidate is known to beat the search's grid points
        (it is a distinct peak), +-pixels. Default 2.
    defocus_step : float | None
        Defocus step of the modelled window, A, when coarser than the refine's own.
        Default 40 (40 and 80 A give the same threshold within 0.02 z for a 20 A
        refine). None: the refine's own step.
    """

    model_config = ConfigDict(extra="forbid")

    n_candidates: int = Field(default=8, ge=1)
    n_draws: int = Field(default=200_000, ge=1000)
    shift_radius: int = Field(default=1, ge=0)
    condition_radius: int = Field(default=2, ge=0)
    defocus_step: float | None = Field(default=40.0, gt=0.0)


class AnalyticNullConfig(BaseModel):
    """Settings of the analytic null prediction.

    Attributes
    ----------
    enabled : bool
        Run the prediction. Default True.
    n_mean_samples : int
        Orientations sampled for the mean template. Default 3000.
    n_kappa_pairs : int
        Pairs of orientation batches sampled for kappa. Default 256.
    n_lattice_samples : int
        Grid points sampled for the lattice correlations. Default 32.
    batch_size : int
        Orientations per batch. Default 4.
    seed : int
        Sampling seed. Default 0.
    device : str
        Device for the calculation, e.g. "cpu" or "cuda:0". Default "cpu".
    refine : AnalyticRefineConfig
        Settings of the refined null (with ``refine_template_config_path``).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    n_mean_samples: int = Field(default=3000, ge=2)
    n_kappa_pairs: int = Field(default=256, ge=2)
    n_lattice_samples: int = Field(default=32, ge=1)
    batch_size: int = Field(default=4, ge=1)
    seed: int = 0
    device: str = "cpu"
    refine: AnalyticRefineConfig = AnalyticRefineConfig()


class NullCalibrationManager(BaseModel2DTM):
    """Thresholds for a match_template configuration from null searches.

    Attributes
    ----------
    match_template_config_path : str
        The match_template YAML whose threshold is calibrated. Its micrograph,
        template and every search setting are used unchanged; only the result paths
        are redirected to ``output_directory``.
    output_directory : str
        Where the null results and the summary are written.
    seeds : list[int]
        One phase-randomised copy per seed. Default 1-5 (enough for +-0.05 z at one
        false positive per micrograph; use 10-20 for +-0.03). Use different seeds for
        different micrographs. An empty list skips the phase-randomised null.
    false_positives : list[float]
        Targets, in false positives per micrograph, to report thresholds for.
    peak_floor : float
        Null peaks are collected down to this z-score. It must lie below the
        thresholds sought. Default 6.0.
    mask_radius : float
        Peak suppression radius, pixels, as for the real micrograph. Default 10.
    border : int
        Exclude peaks within this many pixels of the z-score map's edge. Use the same
        value for the real micrograph. Default 0.
    exclude_mask_path : str | None
        Optional exclusion mask (MRC or .npy, nonzero = excluded) with the z-score
        map's shape, e.g. carbon or flat regions. Use the same mask for the real
        micrograph.
    write_null_particles : bool
        Also export each copy's result maps and a particle stack CSV of its distinct
        peaks at or above ``peak_floor`` (``null_seed<seed>_particles.csv``; the same
        peaks that are counted, so border and mask apply), e.g. to refine the null
        candidates the same way as the real ones. Default False.
    analytic : AnalyticNullConfig
        Settings of the analytic prediction.
    refine_template_config_path : str | None
        A refine_template YAML the search's candidates are refined with. With it (and
        ``refine_candidate_threshold``), the analytic prediction also gives the
        threshold on refined z-scores. Only its orientation, defocus and pixel size
        refinement settings are read.
    refine_candidate_threshold : float | None
        Match_template z-score at or above which candidates are refined. Required with
        ``refine_template_config_path``.
    """

    match_template_config_path: str
    output_directory: str
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    false_positives: list[float] = Field(default_factory=lambda: [1.0])
    peak_floor: float = 6.0
    mask_radius: float = Field(default=10.0, gt=0.0)
    border: int = Field(default=0, ge=0)
    exclude_mask_path: str | None = None
    write_null_particles: bool = False
    analytic: AnalyticNullConfig = AnalyticNullConfig()
    refine_template_config_path: str | None = None
    refine_candidate_threshold: float | None = None

    @model_validator(mode="after")  # type: ignore
    def _check_refine(self) -> "NullCalibrationManager":
        if (self.refine_template_config_path is None) != (
            self.refine_candidate_threshold is None
        ):
            raise ValueError(
                "refine_template_config_path and refine_candidate_threshold go "
                "together."
            )
        return self

    def _match_template_config(self) -> dict[str, Any]:
        with open(self.match_template_config_path, encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f)
        return config

    def make_match_template_manager(self, name: str) -> MatchTemplateManager:
        """The match_template manager, with its results redirected under ``name``.

        Parameters
        ----------
        name : str
            Prefix of the result files in ``output_directory``.

        Returns
        -------
        MatchTemplateManager
        """
        config = self._match_template_config()
        out = Path(self.output_directory)
        out.mkdir(parents=True, exist_ok=True)
        config["match_template_result"] = {
            "allow_file_overwrite": True,
            **{f"{n}_path": str(out / f"{name}_{n}.mrc") for n in RESULT_NAMES},
        }
        return MatchTemplateManager(**config)

    def _exclude_mask(self) -> np.ndarray | None:
        if self.exclude_mask_path is None:
            return None
        if str(self.exclude_mask_path).endswith(".npy"):
            return np.load(self.exclude_mask_path) != 0
        return load_mrc_image(self.exclude_mask_path).numpy() != 0

    def _map_shape(self, manager: MatchTemplateManager) -> tuple[int, int]:
        """Shape of the z-score map: the micrograph's valid correlation region."""
        with mrcfile.open(manager.micrograph_path, header_only=True) as mrc:
            height, width = int(mrc.header.ny), int(mrc.header.nx)
        if manager.micrograph is not None:
            height, width = (int(s) for s in manager.micrograph.shape[-2:])
        with mrcfile.open(manager.template_volume_path, header_only=True) as mrc:
            box_h, box_w = int(mrc.header.ny), int(mrc.header.nx)
        return height - box_h + 1, width - box_w + 1

    def valid_area(self, manager: MatchTemplateManager | None = None) -> int:
        """Number of valid z-score map pixels peaks are counted in."""
        manager = manager or self.make_match_template_manager("area")
        shape = self._map_shape(manager)
        return int(valid_region(shape, self.border, self._exclude_mask()).sum())

    def run_phase_randomized_null(self, orientation_batch_size: int = 16) -> NullPeaks:
        """Search a phase-randomised copy of the micrograph per seed; collect peaks.

        Each copy keeps the micrograph's Fourier amplitudes and gets random phases
        (:func:`~leopard_em.utils.phase_randomization.phase_randomize_image`), then
        is searched with the unchanged match_template configuration. Its distinct
        peaks above ``peak_floor`` are collected with the same suppression radius,
        border and mask as for the real micrograph.

        Parameters
        ----------
        orientation_batch_size : int
            Passed to ``run_match_template``. Default 16.

        Returns
        -------
        NullPeaks
        """
        mask = self._exclude_mask()
        peaks: list[DistinctPeaks] = []
        image = None
        for seed in self.seeds:
            name = f"null_seed{seed}"
            manager = self.make_match_template_manager(name)
            if image is None:
                image = load_mrc_image(manager.micrograph_path)
            manager.micrograph = phase_randomize_image(image, seed)
            manager.run_match_template(
                orientation_batch_size=orientation_batch_size,
                do_result_export=self.write_null_particles,
            )
            result = manager.match_template_result
            found = distinct_peaks(
                result.scaled_mip, self.peak_floor, self.mask_radius, self.border, mask
            )
            peaks.append(found)
            if self.write_null_particles:
                # the counted null peaks themselves, so refined nulls match the count
                result.match_template_peaks = peaks_at(result, found)
                df = manager.results_to_dataframe()
                df.to_csv(Path(self.output_directory) / f"{name}_particles.csv")
            del manager
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return NullPeaks.from_peaks(peaks, seeds=self.seeds)

    def predict_analytic_null(self) -> AnalyticNullPrediction:
        """Predict the null peak rate of the configuration, without a search.

        Returns
        -------
        AnalyticNullPrediction
        """
        manager = self.make_match_template_manager("analytic")
        area = self.valid_area(manager)
        kwargs = manager.make_backend_core_function_kwargs()
        settings = self.analytic
        return predict_match_template_null(
            kwargs,
            area,
            mask_radius=self.mask_radius,
            n_mean_samples=settings.n_mean_samples,
            n_kappa_pairs=settings.n_kappa_pairs,
            n_lattice_samples=settings.n_lattice_samples,
            batch_size=settings.batch_size,
            seed=settings.seed,
            device=settings.device,
        )

    def predict_analytic_refined_null(
        self, prediction: AnalyticNullPrediction | None = None
    ) -> RefinedNullPrediction:
        """Predict the null peak rate after refining the candidates, without a search.

        Parameters
        ----------
        prediction : AnalyticNullPrediction | None
            The search's prediction (:meth:`predict_analytic_null`); made if None.

        Returns
        -------
        RefinedNullPrediction
        """
        if (
            self.refine_template_config_path is None
            or self.refine_candidate_threshold is None
        ):
            raise ValueError(
                "Set refine_template_config_path and refine_candidate_threshold."
            )
        if prediction is None:
            prediction = self.predict_analytic_null()
        manager = self.make_match_template_manager("analytic")
        kwargs = manager.make_backend_core_function_kwargs()
        settings = self.analytic
        offsets, defocus = refine_window(
            self.refine_template_config_path, settings.refine.defocus_step
        )
        return predict_refine_template_null(
            kwargs,
            manager.optics_group,
            prediction,
            offsets,
            defocus,
            self.refine_candidate_threshold,
            shift_radius=settings.refine.shift_radius,
            condition_radius=settings.refine.condition_radius,
            n_candidates=settings.refine.n_candidates,
            n_draws=settings.refine.n_draws,
            n_mean_samples=settings.n_mean_samples,
            seed=settings.seed,
            device=settings.device,
        )

    def summarize(
        self,
        null_peaks: NullPeaks | None = None,
        prediction: AnalyticNullPrediction | None = None,
        refined: RefinedNullPrediction | None = None,
    ) -> dict[str, Any]:
        """Thresholds for each target and the null peak rates, as a JSON-able dict.

        Parameters
        ----------
        null_peaks : NullPeaks | None
            From :meth:`run_phase_randomized_null`.
        prediction : AnalyticNullPrediction | None
            From :meth:`predict_analytic_null`.
        refined : RefinedNullPrediction | None
            From :meth:`predict_analytic_refined_null`.

        Returns
        -------
        dict[str, Any]
            Includes, per target, the nominal (independent-trials) threshold, the
            phase-randomised null thresholds (``"mean"`` and ``"confidence"`` rules),
            the analytic threshold and, with a refinement, the analytic threshold on
            refined z-scores (``"analytic_refined"``).
        """
        manager = self.make_match_template_manager("summary")
        shape = self._map_shape(manager)
        # match_template searches orientations x defocus planes (one pixel size)
        positions = (
            manager.orientation_search_config.euler_angles.shape[0]
            * manager.defocus_search_config.defocus_values.numel()
        )
        out: dict[str, Any] = {
            "match_template_config_path": os.path.abspath(
                self.match_template_config_path
            ),
            "map_shape": list(shape),
            "valid_area": self.valid_area(manager),
            "search_points_per_pixel": int(positions),
            "peak_floor": self.peak_floor,
            "mask_radius": self.mask_radius,
            "border": self.border,
            "seeds": list(self.seeds),
            "thresholds": [],
        }
        for fp in self.false_positives:
            row: dict[str, Any] = {
                "false_positives": fp,
                "nominal": float(
                    gaussian_noise_zscore_cutoff(shape[0] * shape[1] * positions, fp)
                ),
            }
            if null_peaks is not None:
                row["phase_randomized"] = _safe_cutoff(null_peaks, fp, "mean")
                row["phase_randomized_confidence_90"] = _safe_cutoff(
                    null_peaks, fp, "confidence"
                )
            if prediction is not None:
                row["analytic"] = prediction.zscore_cutoff(fp)
            if refined is not None:
                row["analytic_refined"] = refined.zscore_cutoff(fp)
            out["thresholds"].append(row)
        if null_peaks is not None:
            grid = np.round(np.arange(self.peak_floor, self.peak_floor + 4.0, 0.05), 3)
            out["phase_randomized_rate"] = null_peaks.rate(grid).as_dict()
            out["null_peaks"] = null_peaks.as_dict()
        if prediction is not None:
            out["analytic"] = prediction.as_dict()
        if refined is not None:
            out["refine_template_config_path"] = os.path.abspath(
                str(self.refine_template_config_path)
            )
            out["analytic_refined"] = refined.as_dict()
        return out

    def run(self, orientation_batch_size: int = 16) -> dict[str, Any]:
        """Run the configured nulls and write ``null_summary.json``.

        Parameters
        ----------
        orientation_batch_size : int
            Passed to ``run_match_template``. Default 16.

        Returns
        -------
        dict[str, Any]
            The summary (:meth:`summarize`).
        """
        null_peaks = (
            self.run_phase_randomized_null(orientation_batch_size)
            if self.seeds
            else None
        )
        prediction = self.predict_analytic_null() if self.analytic.enabled else None
        refined = (
            self.predict_analytic_refined_null(prediction)
            if prediction is not None and self.refine_template_config_path is not None
            else None
        )
        summary = self.summarize(null_peaks, prediction, refined)
        path = Path(self.output_directory) / "null_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        return summary


def _safe_cutoff(null_peaks: NullPeaks, fp: float, rule: str) -> float | None:
    """The empirical threshold, or None if the peak floor is too high for it."""
    try:
        value = null_peaks.zscore_cutoff(fp, rule=rule)  # type: ignore[arg-type]
    except ValueError:
        return None
    return None if np.isnan(value) else value
