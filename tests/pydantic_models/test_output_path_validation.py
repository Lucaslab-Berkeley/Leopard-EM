"""Output paths must be rejected up front, not after an expensive run completes."""

import os

import pytest

from leopard_em.pydantic_models.managers import (
    ConstrainedSearchManager,
    RefineTemplateManager,
)
from leopard_em.pydantic_models.results import (
    MatchTemplateResultHDF5,
    MatchTemplateResultMRC,
)
from leopard_em.pydantic_models.results.correlation_table import CorrelationTable

MRC_PATH_FIELDS = (
    "mip_path",
    "scaled_mip_path",
    "correlation_average_path",
    "correlation_variance_path",
    "orientation_psi_path",
    "orientation_theta_path",
    "orientation_phi_path",
    "relative_defocus_path",
)


@pytest.fixture()
def existing_file(tmp_path):
    """An output path that is already occupied."""
    path = tmp_path / "already_here.h5"
    path.write_text("occupied")
    return str(path)


class TestCorrelationTablePathValidated:
    """``correlation_table_path`` is checked at construction, like every other path."""

    def test_mrc_backend_rejects_existing_path(self, tmp_path, existing_file):
        paths = {f: str(tmp_path / f"{f}.mrc") for f in MRC_PATH_FIELDS}
        with pytest.raises(ValueError, match="already exists"):
            MatchTemplateResultMRC(correlation_table_path=existing_file, **paths)

    def test_hdf5_backend_rejects_existing_path(self, tmp_path, existing_file):
        with pytest.raises(ValueError, match="already exists"):
            MatchTemplateResultHDF5(
                hdf5_path=str(tmp_path / "out.h5"),
                correlation_table_path=existing_file,
            )

    def test_accepted_when_overwrite_allowed(self, tmp_path, existing_file):
        result = MatchTemplateResultHDF5(
            hdf5_path=str(tmp_path / "out.h5"),
            correlation_table_path=existing_file,
            allow_file_overwrite=True,
        )
        assert result.correlation_table_path == existing_file

    def test_unset_path_is_allowed(self, tmp_path):
        result = MatchTemplateResultHDF5(hdf5_path=str(tmp_path / "out.h5"))
        assert result.correlation_table_path is None


def _minimal_correlation_table() -> CorrelationTable:
    return CorrelationTable(
        correlation_threshold=5.5,
        num_observations=0,
        defocus_offsets=[0.0],
        euler_angles=[(0.0, 0.0, 0.0)],
        search_index=[],
        x=[],
        y=[],
        correlation_value=[],
        correlation_mean=[],
        correlation_variance=[],
    )


class TestCorrelationTablePathReassignmentBypass:
    """Assigning correlation_table_path post-construction skips pydantic validation."""

    def test_export_rejects_path_assigned_after_construction(
        self, tmp_path, existing_file
    ):
        result = MatchTemplateResultHDF5(hdf5_path=str(tmp_path / "out.h5"))
        result.correlation_table_path = existing_file
        result.correlation_table = _minimal_correlation_table()

        with pytest.raises(ValueError, match="already exists"):
            result.export_correlation_table()

    def test_export_succeeds_when_overwrite_allowed(self, tmp_path, existing_file):
        result = MatchTemplateResultHDF5(
            hdf5_path=str(tmp_path / "out.h5"), allow_file_overwrite=True
        )
        result.correlation_table_path = existing_file
        result.correlation_table = _minimal_correlation_table()

        result.export_correlation_table()


class TestManagersCheckOutputPathBeforeCompute:
    """The run_* methods validate the output directory before any backend work.

    Existing outputs are not rejected: ``allow_file_overwrite`` is deprecated and
    particle stack outputs are always overwritten atomically.
    """

    class ComputeStarted(Exception):
        """Raised in place of the backend to mark that validation passed."""

    @classmethod
    def _run_method(cls, monkeypatch, manager_cls, method_name):
        def _started(*args, **kwargs):
            raise cls.ComputeStarted

        monkeypatch.setattr(manager_cls, "make_backend_core_function_kwargs", _started)
        return getattr(manager_cls.model_construct(), method_name)

    @pytest.fixture(
        params=[
            (RefineTemplateManager, "run_refine_template"),
            (ConstrainedSearchManager, "run_constrained_search"),
        ],
        ids=["refine", "constrained"],
    )
    def run(self, request, monkeypatch):
        return self._run_method(monkeypatch, *request.param)

    def test_existing_output_is_not_rejected(self, run, existing_file):
        with pytest.raises(self.ComputeStarted):
            run(output_dataframe_path=existing_file)

    def test_allow_file_overwrite_is_deprecated(self, run, existing_file):
        with pytest.warns(DeprecationWarning, match="allow_file_overwrite"):
            with pytest.raises(self.ComputeStarted):
                run(output_dataframe_path=existing_file, allow_file_overwrite=False)

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root ignores directory permissions",
    )
    def test_unwritable_directory_is_rejected(self, run, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            with pytest.raises(ValueError, match="does not permit writing"):
                run(output_dataframe_path=str(locked / "out.csv"))
        finally:
            locked.chmod(0o700)

    def test_constrained_search_sibling_tables_are_not_rejected(
        self, monkeypatch, tmp_path
    ):
        """The '_parameters' / '_above_threshold' siblings are overwritten too."""
        (tmp_path / "out_above_threshold.csv").write_text("occupied")
        run = self._run_method(
            monkeypatch, ConstrainedSearchManager, "run_constrained_search"
        )
        with pytest.raises(self.ComputeStarted):
            run(output_dataframe_path=str(tmp_path / "out.csv"))
