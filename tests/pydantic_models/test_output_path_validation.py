"""Output paths must be rejected up front, not after an expensive run completes."""

import pytest

from leopard_em.pydantic_models.managers import (
    ConstrainedSearchManager,
    RefineTemplateManager,
)
from leopard_em.pydantic_models.results import (
    MatchTemplateResultHDF5,
    MatchTemplateResultMRC,
)

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


class TestManagersCheckOutputPathBeforeCompute:
    """The run_* methods must raise before any backend work is started."""

    @staticmethod
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("compute started before the output path was validated")

    def test_refine_template(self, monkeypatch, existing_file):
        monkeypatch.setattr(
            RefineTemplateManager,
            "make_backend_core_function_kwargs",
            self._fail_if_called,
        )
        manager = RefineTemplateManager.model_construct()

        with pytest.raises(ValueError, match="already exists"):
            manager.run_refine_template(output_dataframe_path=existing_file)

    def test_constrained_search(self, monkeypatch, existing_file):
        monkeypatch.setattr(
            ConstrainedSearchManager,
            "make_backend_core_function_kwargs",
            self._fail_if_called,
        )
        manager = ConstrainedSearchManager.model_construct()

        with pytest.raises(ValueError, match="already exists"):
            manager.run_constrained_search(output_dataframe_path=existing_file)

    def test_constrained_search_checks_sibling_tables(self, monkeypatch, tmp_path):
        """The '_parameters' / '_above_threshold' siblings are checked too."""
        monkeypatch.setattr(
            ConstrainedSearchManager,
            "make_backend_core_function_kwargs",
            self._fail_if_called,
        )
        (tmp_path / "out_above_threshold.csv").write_text("occupied")
        manager = ConstrainedSearchManager.model_construct()

        with pytest.raises(ValueError, match="already exists"):
            manager.run_constrained_search(
                output_dataframe_path=str(tmp_path / "out.csv")
            )
