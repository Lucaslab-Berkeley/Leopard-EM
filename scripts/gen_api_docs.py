"""Generate mkdocstrings stub pages for all public leopard_em subpackages.

Run this before `zensical build`. Output goes to docs/api/ which is git-ignored.
Each stub file contains a single `:::module` directive that mkdocstrings expands
into full API documentation from the source docstrings.
"""

from pathlib import Path

SRC = Path("src/leopard_em")
OUT = Path("docs/api")

# One page per subpackage: maps page title -> module import path.
# Add entries here whenever a new public subpackage is introduced.
PAGES = {
    "index": ("leopard_em", "API Reference"),
    "analysis": ("leopard_em.analysis", "Analysis"),
    "backend": ("leopard_em.backend", "Backend"),
    "pydantic_models": ("leopard_em.pydantic_models", "Pydantic Models"),
    "pydantic_models/config": ("leopard_em.pydantic_models.config", "Config Models"),
    "pydantic_models/data_structures": (
        "leopard_em.pydantic_models.data_structures",
        "Data Structures",
    ),
    "pydantic_models/managers": (
        "leopard_em.pydantic_models.managers",
        "Managers",
    ),
    "pydantic_models/results": (
        "leopard_em.pydantic_models.results",
        "Results",
    ),
    "utils": ("leopard_em.utils", "Utilities"),
}


def main() -> None:
    """Generate the stub pages."""
    OUT.mkdir(parents=True, exist_ok=True)
    for page_path, (module, title) in PAGES.items():
        stub = OUT / f"{page_path}.md"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(f"# {title}\n\n:::{module}\n")
        print(f"  wrote {stub}")


if __name__ == "__main__":
    main()
