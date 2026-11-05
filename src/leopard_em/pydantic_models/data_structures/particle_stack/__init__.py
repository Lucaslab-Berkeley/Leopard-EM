"""Particle stack Pydantic models for dealing with extracted particle data.

Two public classes are provided for different storage back-ends:

* ``ParticleStackCSV`` - the original behavior, loading particle data from a
  CSV file and micrograph images from referenced paths on disk.
  ``ParticleStack`` is an alias for this class for backward compatibility.
* ``ParticleStackHDF5`` - stores the particle table, optional image stack, and
  optional per-particle local correlation statistics in a single HDF5 file.

Package layout (modules only import from modules listed below them)::

    export        AnyParticleStack field type, export_particle_stack and
                  particle_stack_to_hdf5 (conversion between back-ends)
    csv_stack     ParticleStackCSV (and the ParticleStack alias)
    hdf5_stack    ParticleStackHDF5, on top of the _hdf5_io file codec
    base          _ParticleStackBase: shared state and methods, delegating to
    _movie        per-particle extraction from (motion-corrected) movies
    _extraction   loading, cropping and scattering per group of particles sharing
                  an image
    _table        particle-table helpers (IDs, refined columns, row positions)
    geometry      ParticleBoxGeometry: crop origins and valid sizes
"""

from leopard_em.utils.cropping import (
    TORCH_TO_NUMPY_PADDING_MODE,
    get_cropped_image_regions,
)

from .base import _warn_allow_file_overwrite_deprecated
from .csv_stack import ParticleStack, ParticleStackCSV
from .export import AnyParticleStack, export_particle_stack
from .geometry import ParticleBoxGeometry
from .hdf5_stack import ParticleStackHDF5

__all__ = [
    "TORCH_TO_NUMPY_PADDING_MODE",
    "AnyParticleStack",
    "ParticleBoxGeometry",
    "ParticleStack",
    "ParticleStackCSV",
    "ParticleStackHDF5",
    "_warn_allow_file_overwrite_deprecated",
    "export_particle_stack",
    "get_cropped_image_regions",
]
