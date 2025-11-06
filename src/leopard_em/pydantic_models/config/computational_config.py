"""Computational configuration for 2DTM."""

from typing import Annotated, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field

# Type alias for non-negative integer
NonNegativeInt = Annotated[int, Field(ge=0)]


class BaseComputationalConfig(BaseModel):
    """Base class for computational configuration with shared GPU device handling."""

    @property
    def gpu_devices(self) -> list[torch.device]:
        """Convert requested GPU IDs to torch device objects.

        Returns
        -------
        list[torch.device]
        """
        # Handle special string cases first
        if self.gpu_ids == "all":
            if not torch.cuda.is_available():
                raise ValueError("No CUDA devices available.")
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        if self.gpu_ids == "cpu":
            return [torch.device("cpu")]

        # Normalize to list for uniform processing
        gpu_list = self.gpu_ids if isinstance(self.gpu_ids, list) else [self.gpu_ids]

        # Process each item in the normalized list
        devices = []
        for gpu_id in gpu_list:
            if isinstance(gpu_id, int):
                devices.append(torch.device(f"cuda:{gpu_id}"))
            elif isinstance(gpu_id, str):
                devices.append(torch.device(gpu_id))
            else:
                raise TypeError(
                    f"Invalid type for gpu_ids element: {type(gpu_id)}. "
                    "Expected int or str."
                )

        return devices


class ComputationalConfigMatch(BaseComputationalConfig):
    """Serialization of computational resources allocated for 2DTM.

    NOTE: The field `gpu_ids` is not validated at instantiation past being one of the
    valid types. For example, if "cuda:0" is specified but no CUDA device is available,
    the instantiation will succeed, and only upon translating `gpu_ids` to a list of
    `torch.device` objects will an error be raised. This is done to allow for
    configuration files to be loaded without requiring the actual hardware to be
    present at the time of loading.

    Attributes
    ----------
    gpu_ids : Optional[Union[int, list[int], str, list[str]]]
        Field which specifies which GPUs to use for computation. The following types
        of values are allowed:
        - A single integer, e.g. 0, which means to use GPU with ID 0.
        - A list of integers, e.g. [0, 2], which means to use GPUs with IDs 0 and 2.
        - A device specifier string, e.g. "cuda:0", which means to use GPU with ID 0.
        - A list of device specifier strings, e.g. ["cuda:0", "cuda:1"], which means to
          use GPUs with IDs 0 and 1.
        - The specific string "all" which means to use all available GPUs identified
          by torch.cuda.device_count().
        - The specific string "cpu" which means to use CPU.
    num_cpus : int
        Total number of CPUs to use, defaults to 1.
    backend : Optional[str]
        The backend to use for match template.
        Must be "streamed" or "batched".
        Defaults to "streamed".
    """

    # Type-hinting here is ensuring non-negative integers, and list of at least one
    gpu_ids: Optional[
        Union[
            str,
            NonNegativeInt,
            Annotated[list[NonNegativeInt], Field(min_length=1)],
            Annotated[list[str], Field(min_length=1)],
        ]
    ] = [0]
    num_cpus: Annotated[int, Field(ge=1)] = 1
    backend: Literal["streamed", "batched"] = "streamed"


class ComputationalConfigRefine(BaseComputationalConfig):
    """Serialization of computational resources allocated for 2DTM.

    NOTE: The field `gpu_ids` is not validated at instantiation past being one of the
    valid types. For example, if "cuda:0" is specified but no CUDA device is available,
    the instantiation will succeed, and only upon translating `gpu_ids` to a list of
    `torch.device` objects will an error be raised. This is done to allow for
    configuration files to be loaded without requiring the actual hardware to be
    present at the time of loading.

    Attributes
    ----------
    gpu_ids : Optional[Union[int, list[int], str, list[str]]]
        Field which specifies which GPUs to use for computation. The following types
        of values are allowed:
        - A single integer, e.g. 0, which means to use GPU with ID 0.
        - A list of integers, e.g. [0, 2], which means to use GPUs with IDs 0 and 2.
        - A device specifier string, e.g. "cuda:0", which means to use GPU with ID 0.
        - A list of device specifier strings, e.g. ["cuda:0", "cuda:1"], which means to
          use GPUs with IDs 0 and 1.
        - The specific string "all" which means to use all available GPUs identified
          by torch.cuda.device_count().
        - The specific string "cpu" which means to use CPU.
    num_cpus : int
        Total number of CPUs to use, defaults to 1.
    backend : Optional[str]
        The backend to use for match template.
        Must be "streamed" or "batched".
        Defaults to "streamed".
    """

    # Type-hinting here is ensuring non-negative integers, and list of at least one
    gpu_ids: Optional[
        Union[
            str,
            NonNegativeInt,
            Annotated[list[NonNegativeInt], Field(min_length=1)],
            Annotated[list[str], Field(min_length=1)],
        ]
    ] = [0]
    num_cpus: Annotated[int, Field(ge=1)] = 1
    backend: Literal["streamed", "batched"] = "batched"
