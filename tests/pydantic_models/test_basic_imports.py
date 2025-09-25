"""Ensure that all Pydantic models can be imported from their default locations."""


def test_manager_imports():
    """Test for manager imports."""
    try:
        from leopard_em.pydantic_models.managers import (
            MatchTemplateManager,
            OptimizeTemplateManager,
            RefineTemplateManager,
        )
    except ImportError as e:
        raise ImportError(
            "Failed to import one or more manager classes from "
            "leopard_em.pydantic_models.managers."
        ) from e


def test_config_imports():
    """Test for config imports."""
    try:
        from leopard_em.pydantic_models.config import (
            ArbitraryCurveFilterConfig,
            BandpassFilterConfig,
            ComputationalConfigMatch,
            ComputationalConfigRefine,
            DefocusSearchConfig,
            OrientationSearchConfig,
            PhaseRandomizationFilterConfig,
            PixelSizeSearchConfig,
            PreprocessingFilters,
            RefineOrientationConfig,
            WhiteningFilterConfig,
        )
    except ImportError as e:
        raise ImportError(
            "Failed to import one or more config classes from "
            "leopard_em.pydantic_models.config."
        ) from e


def test_data_structure_imports():
    """Test for data structure imports."""
    try:
        from leopard_em.pydantic_models.data_structures import (
            OpticsGroup,
            ParticleStack,
        )
    except ImportError as e:
        raise ImportError(
            "Failed to import ParticleStack from "
            "leopard_em.pydantic_models.data_structures."
        ) from e


def test_results_import():
    """Test for results imports."""
    try:
        from leopard_em.pydantic_models.results import MatchTemplateResult
    except ImportError as e:
        raise ImportError(
            "Failed to import one or more results classes from "
            "leopard_em.pydantic_models.results."
        ) from e
