"""Tests for HDF5-backed particle stacks: file format, persistence and indexing."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pydantic import TypeAdapter, ValidationError

from leopard_em.pydantic_models.data_structures import (
    AnyParticleStack,
    ParticleStackCSV,
    ParticleStackHDF5,
    export_particle_stack,
)
from leopard_em.pydantic_models.data_structures import (
    _particle_stack_hdf5_io as hdf5_io,
)
from leopard_em.pydantic_models.managers import RefineTemplateManager

# pylint: disable=protected-access


def _export(workspace, name="stack.h5", **kwargs):
    return export_particle_stack(
        workspace.df,
        str(workspace.root / name),
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
        **kwargs,
    )


def _write_legacy_v1_file(path, df, box=(20, 20), template=(16, 16)):
    """Frozen copy of the Leopard-EM v1.3 ``ParticleStackHDF5.to_hdf5`` writer."""
    with h5py.File(path, "w") as f:
        f.attrs["leopard_em_version"] = "1.3.0"
        f.attrs["extracted_box_size"] = list(box)
        f.attrs["original_template_size"] = list(template)
        for flag in hdf5_io.PREPROCESSING_FLAG_ATTRS:
            f.attrs[flag] = False
        grp = f.create_group("particles")
        grp.attrs["columns"] = ["particle_id", *list(df.columns)]
        grp.create_dataset(
            "particle_id",
            data=np.array(df.index.tolist(), dtype=object),
            dtype=h5py.string_dtype(),
        )
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(
                series
            ):
                grp.create_dataset(col, data=series.to_numpy(dtype=np.float64))
            else:
                values = [
                    "" if v is None else (v if isinstance(v, str) else json.dumps(v))
                    for v in series
                ]
                grp.create_dataset(
                    col,
                    data=np.array(values, dtype=object),
                    dtype=h5py.string_dtype(),
                )
        f.attrs["image_stack_stored"] = False
        f.attrs["local_stats_stored"] = False


##########################
### Table round-trips  ###
##########################


def test_particle_table_roundtrip_preserves_types(workspace):
    """Numeric/bool dtypes survive; list/dict/None cells come back as CSV would."""
    df = workspace.df
    df["is_good"] = [True, False, True, True, False, True]
    df["even_zernikes"] = [{"Z40": 0.1}] * len(df)
    df["note"] = ["[1].mrc", None, "12", "", float("nan"), "plain"]
    stack = _export(workspace)

    loaded = ParticleStackHDF5(hdf5_path=stack.hdf5_path)
    out = loaded.get_dataframe_copy()

    assert isinstance(out.index, pd.RangeIndex)
    assert out.columns[0] == "particle_id"
    assert out["particle_index"].dtype == np.int64
    assert out["pos_x"].dtype == np.int64
    assert out["is_good"].dtype == bool
    assert out["is_good"].tolist() == df["is_good"].tolist()
    assert out["pos_x"].tolist() == df["pos_x"].tolist()
    # list/dict cells -> JSON text (what a CSV round trip gives), never re-parsed
    assert json.loads(out["mag_matrix"].iloc[0]) == [1.0, 0.0, 0.0, 1.0]
    assert json.loads(out["even_zernikes"].iloc[0]) == {"Z40": 0.1}
    notes = out["note"].tolist()
    assert notes[0] == "[1].mrc"  # JSON-looking strings stay strings
    assert notes[2] == "12"
    assert all(pd.isna(v) for v in (notes[1], notes[3], notes[4]))

    with h5py.File(stack.hdf5_path, "r") as f:
        assert f.attrs["format_version"] == hdf5_io.FORMAT_VERSION
        assert "writer_version" in f.attrs


def test_generated_particle_ids_are_unique_for_shared_stems(workspace):
    """Both micrographs are named 'micrograph.mrc' in different directories."""
    stack = _export(workspace)
    ids = stack["particle_id"].tolist()
    assert len(set(ids)) == len(ids)
    assert all(pid.startswith("micrograph-") for pid in ids)


def test_existing_particle_ids_are_kept_and_validated(workspace):
    workspace.df.insert(0, "particle_id", [f"p{i}" for i in range(len(workspace.df))])
    stack = _export(workspace)
    assert stack["particle_id"].tolist() == workspace.df["particle_id"].tolist()

    workspace.df["particle_id"] = ["dup"] * len(workspace.df)
    with pytest.raises(ValueError, match="duplicate"):
        _export(workspace, name="dups.h5")


def test_legacy_v1_file_loads(workspace):
    """Files written by v1.3 (float positions, particle_id index) still load."""
    df = workspace.df.drop(columns=["mag_matrix"]).assign(mag_matrix=None)
    df.index = pd.Index([f"old_{i:05d}" for i in range(len(df))], name="particle_id")
    path = workspace.root / "legacy.h5"
    _write_legacy_v1_file(path, df)

    loaded = ParticleStackHDF5(hdf5_path=str(path))

    assert loaded.extracted_box_size == (20, 20)
    assert loaded.leopard_em_version == "1.3.0"
    assert isinstance(loaded._df.index, pd.RangeIndex)
    assert loaded["particle_id"].tolist() == df.index.tolist()
    # float64-stored positions still index correctly
    images, indices = loaded.load_images_grouped_by_column("micrograph_path")
    stack = loaded.construct_image_stack(images, indices, extraction_size=(20, 20))
    assert stack.shape == (6, 20, 20)


def test_legacy_v1_duplicate_ids_are_regenerated(workspace):
    df = workspace.df.copy()
    df.index = pd.Index(["micrograph_00000"] * len(df), name="particle_id")
    path = workspace.root / "legacy_dups.h5"
    _write_legacy_v1_file(path, df)

    with pytest.warns(UserWarning, match="regenerating particle IDs"):
        loaded = ParticleStackHDF5(hdf5_path=str(path))
    assert loaded["particle_id"].is_unique


###################################
### Construction / persistence  ###
###################################


def test_yaml_config_loads_existing_hdf5_stack(workspace):
    """The documented YAML form (hdf5_path only) loads an existing file."""
    stack = _export(workspace)
    config = {
        "template_volume_path": str(workspace.template_path),
        "particle_stack": {"hdf5_path": stack.hdf5_path},
        "defocus_refinement_config": {"enabled": False},
        "orientation_refinement_config": {"enabled": False},
        "pixel_size_refinement_config": {"enabled": False},
        "preprocessing_filters": {},
        "computational_config": {"gpu_ids": "cpu"},
        "movie_config": {"enabled": False},
    }
    config_path = workspace.root / "refine.yaml"
    config_path.write_text(yaml.safe_dump(config))

    manager = RefineTemplateManager.from_yaml(str(config_path))

    ps = manager.particle_stack
    assert isinstance(ps, ParticleStackHDF5)
    assert ps.extracted_box_size == workspace.extracted_box_size
    assert ps.original_template_size == workspace.original_template_size
    assert ps.num_particles == len(workspace.df)
    # Deprecated / file-derived fields are not written back out
    assert "allow_file_overwrite" not in ps.model_dump()
    assert "image_stack_stored" not in ps.model_dump()


def test_particle_stack_union_picks_backend_from_keys(workspace):
    adapter = TypeAdapter(AnyParticleStack)
    stack = _export(workspace)
    assert isinstance(
        adapter.validate_python({"hdf5_path": stack.hdf5_path}), ParticleStackHDF5
    )

    csv_path = workspace.root / "stack.csv"
    workspace.df.to_csv(csv_path)
    csv_stack = adapter.validate_python(
        {
            "df_path": str(csv_path),
            "extracted_box_size": [20, 20],
            "original_template_size": [16, 16],
        }
    )
    assert isinstance(csv_stack, ParticleStackCSV)
    assert adapter.validate_python(csv_stack) is csv_stack

    with pytest.raises(ValidationError):
        adapter.validate_python({"extracted_box_size": [20, 20]})


def test_explicit_box_size_overrides_file_and_drops_incompatible_tensors(workspace):
    stack = _export(workspace)
    stack.local_stats.update(stack.get_local_stat_maps())
    stack.to_hdf5(include_local_stats=True)

    with pytest.warns(UserWarning, match="Ignoring stored"):
        loaded = ParticleStackHDF5(
            hdf5_path=stack.hdf5_path, extracted_box_size=(24, 24)
        )
    assert loaded.extracted_box_size == (24, 24)
    assert loaded.original_template_size == workspace.original_template_size
    assert loaded.local_stats == {}
    assert loaded.local_stats_stored  # the file still has them


def test_rewrite_without_tensors_updates_stored_flags(workspace):
    """Re-exporting without tensors must not leave stale *_stored attributes."""
    stack = _export(workspace)
    images, indices = stack.load_images_grouped_by_column("micrograph_path")
    stack.construct_image_stack(images, indices, stack.extracted_box_size)
    stack.local_stats.update(stack.get_local_stat_maps(columns=["mip_path"]))
    stack.to_hdf5(include_image_stack=True, include_local_stats=True)

    loaded = ParticleStackHDF5.from_hdf5(stack.hdf5_path)
    assert loaded.image_stack_stored and loaded.local_stats_stored
    torch.testing.assert_close(loaded.get_stored_image_stack(), stack.image_stack)

    loaded.to_hdf5()  # table only, same path
    reloaded = ParticleStackHDF5(hdf5_path=stack.hdf5_path)
    assert not reloaded.image_stack_stored
    assert not reloaded.local_stats_stored
    assert reloaded.get_stored_image_stack() is None


def test_use_stored_image_stack_false_skips_loading(workspace):
    stack = _export(workspace)
    images, indices = stack.load_images_grouped_by_column("micrograph_path")
    stack.construct_image_stack(images, indices, stack.extracted_box_size)
    stack.to_hdf5(include_image_stack=True)

    loaded = ParticleStackHDF5(hdf5_path=stack.hdf5_path, use_stored_image_stack=False)
    assert loaded.image_stack_stored
    assert loaded.image_stack is None
    assert loaded.get_stored_image_stack() is None


def test_set_column_invalidates_stored_tensors(workspace):
    stack = _export(workspace)
    images, indices = stack.load_images_grouped_by_column("micrograph_path")
    stack.construct_image_stack(images, indices, stack.extracted_box_size)
    stack.local_stats.update(
        stack.get_local_stat_maps(columns=["mip_path", "psi_path"])
    )
    stack.to_hdf5(include_image_stack=True, include_local_stats=True)
    assert stack.get_stored_image_stack() is not None

    stack.set_column("mip_path", str(workspace.result_paths[0]))
    assert set(stack.local_stats) == {"psi_path"}
    assert stack.get_stored_image_stack() is not None

    stack.set_column("pos_x", stack["pos_x"] + 1)
    assert stack.local_stats == {}
    assert stack.get_stored_image_stack() is None


def test_set_column_series_is_assigned_by_position(workspace):
    stack = _export(workspace)
    values = pd.Series(range(len(workspace.df)), index=stack["particle_id"])
    stack.set_column("particle_index", values)
    assert stack["particle_index"].tolist() == list(range(len(workspace.df)))


##########################
### Overwrite / export ###
##########################


def test_writes_overwrite_existing_files_atomically(workspace, monkeypatch):
    stack = _export(workspace)
    _export(workspace)  # same path again: no error, no flag needed
    original = (workspace.root / "stack.h5").read_bytes()

    def _fail(*_args, **_kwargs):
        raise RuntimeError("simulated failure mid-write")

    monkeypatch.setattr(hdf5_io, "_write_particle_table", _fail)
    with pytest.raises(RuntimeError, match="simulated"):
        stack.to_hdf5()
    assert (workspace.root / "stack.h5").read_bytes() == original
    assert not list(workspace.root.glob(".stack.h5.*.tmp"))


def test_allow_file_overwrite_is_deprecated_everywhere(workspace):
    stack = _export(workspace)
    csv_path = workspace.root / "stack.csv"

    with pytest.warns(DeprecationWarning, match="allow_file_overwrite"):
        ParticleStackHDF5(hdf5_path=stack.hdf5_path, allow_file_overwrite=False)
    with pytest.warns(DeprecationWarning, match="allow_file_overwrite"):
        ParticleStackHDF5.from_hdf5(stack.hdf5_path, allow_file_overwrite=True)
    with pytest.warns(DeprecationWarning, match="allow_file_overwrite"):
        csv_stack = _export(workspace, name="stack.csv", allow_file_overwrite=False)
    with pytest.warns(DeprecationWarning, match="allow_file_overwrite"):
        csv_stack.export_results(allow_file_overwrite=False)  # value ignored
    with pytest.warns(DeprecationWarning, match="allow_file_overwrite"):
        csv_stack.to_hdf5(str(workspace.root / "converted.h5"), False)
    assert csv_path.exists()


def test_export_particle_stack_accepts_v13_positional_call(workspace):
    """``export_particle_stack(df, path, source, format, overwrite)`` still binds."""
    source = _export(workspace)
    with pytest.warns(DeprecationWarning):
        out = export_particle_stack(
            workspace.df, str(workspace.root / "out.h5"), source, "hdf5", True
        )
    assert isinstance(out, ParticleStackHDF5)
    with h5py.File(out.hdf5_path, "r") as f:
        assert "particles" in f


def test_export_particle_stack_format_inference(workspace):
    source = _export(workspace)  # HDF5 source

    assert isinstance(
        export_particle_stack(workspace.df, str(workspace.root / "a.csv"), source),
        ParticleStackCSV,
    )
    assert isinstance(
        export_particle_stack(workspace.df, str(workspace.root / "a"), source),
        ParticleStackHDF5,
    )
    with pytest.raises(ValueError, match="conflicts with the file extension"):
        export_particle_stack(
            workspace.df, str(workspace.root / "b.csv"), source, output_format="hdf5"
        )


def test_export_particle_stack_without_source_roundtrips(workspace):
    stack = _export(workspace, name="no_source.h5")
    loaded = ParticleStackHDF5(hdf5_path=stack.hdf5_path)
    assert loaded.extracted_box_size == workspace.extracted_box_size
    assert not loaded.global_whitening_applied
    pd.testing.assert_frame_equal(
        loaded.get_dataframe_copy().drop(columns=["particle_id", "mag_matrix"]),
        workspace.df.drop(columns=["mag_matrix"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_csv_hdf5_csv_conversion_keeps_particle_ids(workspace):
    hdf5_stack = _export(workspace)
    csv_stack = export_particle_stack(
        hdf5_stack.get_dataframe_copy(), str(workspace.root / "ids.csv"), hdf5_stack
    )
    reloaded_csv = ParticleStackCSV(
        df_path=csv_stack.df_path,
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
    )
    assert reloaded_csv["particle_id"].tolist() == hdf5_stack["particle_id"].tolist()

    back = reloaded_csv.to_hdf5(str(workspace.root / "back.h5"))
    assert back["particle_id"].tolist() == hdf5_stack["particle_id"].tolist()


###########################
### Manager integration ###
###########################


def test_refined_results_keep_particle_ids(workspace):
    """particle_id survives refine_result_to_dataframe and the HDF5 re-export."""
    workspace.df["refined_relative_defocus"] = 0.0
    stack = _export(workspace)
    manager = RefineTemplateManager.model_construct(particle_stack=stack)
    num = stack.num_particles
    result = {
        "refined_pos_y": np.zeros(num, dtype=np.int64),
        "refined_pos_x": np.ones(num, dtype=np.int64),
        "refined_euler_angles": np.zeros((num, 3)),
        "refined_defocus_offset": np.zeros(num),
        "refined_pixel_size_offset": np.zeros(num),
        "refined_cross_correlation": np.ones(num),
        "refined_z_score": np.ones(num),
    }
    df_refined = manager.refine_result_to_dataframe(result, prefer_refined_angles=False)
    assert df_refined.columns[0] == "particle_id"
    assert df_refined["particle_id"].tolist() == stack["particle_id"].tolist()

    refined = export_particle_stack(
        df_refined, str(workspace.root / "refined.h5"), stack
    )
    assert refined["particle_id"].tolist() == stack["particle_id"].tolist()


def test_constrained_search_requires_single_correlation_maps(workspace):
    from leopard_em.pydantic_models.managers import ConstrainedSearchManager

    stack = _export(workspace)
    manager = ConstrainedSearchManager.model_construct(particle_stack_constrained=stack)
    with pytest.raises(ValueError, match="exactly one 'correlation_average_path'"):
        manager._constrained_correlation_map_paths()

    single = export_particle_stack(
        stack.get_dataframe_copy().iloc[[0, 2, 5]],  # micrograph 0 only
        str(workspace.root / "single.h5"),
        stack,
    )
    manager = ConstrainedSearchManager.model_construct(
        particle_stack_constrained=single
    )
    assert manager._constrained_correlation_map_paths() == {
        "correlation_average_path": str(workspace.result_paths[0]),
        "correlation_variance_path": str(workspace.result_paths[0]),
    }
