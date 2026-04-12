"""Write spatially CTF-corrected MRCs using ``SpatialCtfMatchTemplateManager``."""

from __future__ import annotations

from leopard_em.pydantic_models.managers import SpatialCtfMatchTemplateManager

#######################################
### Editable parameters for program ###
#######################################

YAML_CONFIG_PATH = "/path/to/spatial_ctf_match_template.yaml"


def main() -> None:
    """Load YAML and run premultiply-only (requires YAML premultiply output paths)."""
    mgr = SpatialCtfMatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    mgr.run_premultiply_only()


if __name__ == "__main__":
    main()
