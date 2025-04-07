"""Particle stack Pydantic model for dealing with extracted particle data."""

from typing import Any, ClassVar, Literal

import pandas as pd
import torch
from pydantic import ConfigDict

from leopard_em.pydantic_models.correlation_filters import PreprocessingFilters
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.utils.data_io import load_mrc_image
from leopard_em.utils.particle_stack import get_cropped_image_regions

from .types import BaseModel2DTM, ExcludedTensor


class ParticleStack(BaseModel2DTM):
    """Pydantic model for dealing with particle stack data.

    Attributes
    ----------
    df_path : str
        Path to the DataFrame containing the particle data. The DataFrame must have
        the following columns (see the documentation for further information):
          - mip
          - scaled_mip
          - correlation_mean
          - correlation_variance
          - total_correlations
          - pos_x
          - pos_y
          - pos_x_img
          - pos_y_img
          - pos_x_img_angstrom
          - pos_y_img_angstrom
          - psi
          - theta
          - phi
          - relative_defocus
          - refined_relative_defocus
          - defocus_u
          - defocus_v
          - astigmatism_angle
          - pixel_size
          - refined_pixel_size
          - voltage
          - spherical_aberration
          - amplitude_contrast_ratio
          - phase_shift
          - ctf_B_factor
          - micrograph_path
          - template_path
          - mip_path
          - scaled_mip_path
          - psi_path
          - theta_path
          - phi_path
          - defocus_path
          - correlation_average_path
          - correlation_variance_path
    extracted_box_size : tuple[int, int]
        The size of the extracted particle boxes in pixels in units of pixels.
    original_template_size : tuple[int, int]
        The original size of the template used during the matching process. Should be
        smaller than the extracted box size.
    image_stack : ExcludedTensor
        The stack of images extracted from the micrographs. Is effectively a pytorch
        Tensor with shape (N, H, W) where N is the number of particles and (H, W) is
        the extracted box size.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized fields
    df_path: str
    extracted_box_size: tuple[int, int]
    original_template_size: tuple[int, int]

    # Imported tabular data (not serialized)
    _df: pd.DataFrame

    # Cropped out view of the particles from images
    image_stack: ExcludedTensor

    def __init__(self, skip_df_load: bool = False, **data: dict[str, Any]):
        """Initialize the ParticleStack object.

        Parameters
        ----------
        skip_df_load : bool, optional
            Whether to skip loading the DataFrame, by default False and the dataframe
            is loaded automatically.
        data : dict[str, Any]
            The data to initialize the object with.
        """
        super().__init__(**data)

        if not skip_df_load:
            self.load_df()

    def load_df(self) -> None:
        """Load the DataFrame from the specified path.

        Raises
        ------
        ValueError
            If the DataFrame is missing required columns.
        """
        tmp_df = pd.read_csv(self.df_path)

        # Validate the DataFrame columns
        missing_columns = [
            col for col in MATCH_TEMPLATE_DF_COLUMN_ORDER if col not in tmp_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing the following columns in DataFrame: {missing_columns}"
            )

        self._df = tmp_df

    def construct_image_stack(
        self,
        pos_reference: Literal["center", "top-left"] = "center",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """Construct stack of images from the DataFrame (updates image_stack in-place).

        This method preferentially selects refined position columns (refined_pos_x_img,
        refined_pos_y_img) if they are present in the DataFrame, falling back to
        unrefined positions (pos_x_img, pos_y_img) otherwise.

        Parameters
        ----------
        pos_reference : Literal["center", "top-left"], optional
            The reference point for the positions, by default "center". If "center", the
            boxes extracted will be image[y - box_size // 2 : y + box_size // 2, ...].
            If "top-left", the boxes will be image[y : y + box_size, ...].
        handle_bounds : Literal["pad", "clip", "error"], optional
            How to handle the bounds of the image, by default "pad". If "pad", the image
            will be padded with the padding value based on the padding mode. If "error",
            an error will be raised if any region exceeds the image bounds. NOTE:
            clipping is not supported since returned stack may have inhomogeneous sizes.
        padding_mode : Literal["constant", "reflect", "replicate"], optional
            The padding mode to use when padding the image, by default "constant".
            "constant" pads with the value `padding_value`, "reflect" pads with the
            reflection of the image at the edge, and "replicate" pads with the last
            pixel of the image. These match the modes available in
            `torch.nn.functional.pad`.
        padding_value : float, optional
            The value to use for padding when `padding_mode` is "constant", by default
            0.0.

        Returns
        -------
        torch.Tensor
            The stack of images, this is the internal 'image_stack' attribute.
        """
        # Create an empty tensor to store the image stack
        image_stack = torch.zeros((self.num_particles, *self.extracted_box_size))

        # Find the indexes in the DataFrame that correspond to each unique image
        image_index_groups = self._df.groupby("micrograph_path").groups

        # Loop over each unique image and extract the particles
        for img_path, indexes in image_index_groups.items():
            img = load_mrc_image(img_path)

            # with reference to center pixel
            # Determine which position columns to use (refined if available)
            y_col = (
                "refined_pos_y_img"
                if "refined_pos_y_img" in self._df.columns
                else "pos_y_img"
            )
            x_col = (
                "refined_pos_x_img"
                if "refined_pos_x_img" in self._df.columns
                else "pos_x_img"
            )

            pos_y = self._df.loc[indexes, y_col].to_numpy()
            pos_x = self._df.loc[indexes, x_col].to_numpy()

            pos_y = torch.tensor(pos_y)
            pos_x = torch.tensor(pos_x)

            cropped_images = get_cropped_image_regions(
                img,
                pos_y,
                pos_x,
                self.extracted_box_size,
                pos_reference=pos_reference,
                handle_bounds=handle_bounds,
                padding_mode=padding_mode,
                padding_value=padding_value,
            )
            image_stack[indexes] = cropped_images

        self.image_stack = image_stack

        return image_stack

    def construct_cropped_statistic_stack(
        self,
        stat: Literal[
            "mip",
            "scaled_mip",
            "correlation_average",
            "correlation_variance",
            "defocus",
            "psi",
            "theta",
            "phi",
        ],
    ) -> torch.Tensor:
        """Return a tensor of the specified statistic for each cropped image."""
        stat_col = f"{stat}_path"

        if stat_col not in self._df.columns:
            raise ValueError(f"Statistic '{stat}' not found in the DataFrame.")

        # Create an empty tensor to store the stat stack
        h, w = self.original_template_size
        H, W = self.extracted_box_size
        stat_stack = torch.zeros((self.num_particles, H - h + 1, W - w + 1))

        # Find the indexes in the DataFrame that correspond to each unique stat map
        stat_index_groups = self._df.groupby(stat_col).groups

        # Loop over each unique stat map and extract the particles
        for stat_path, indexes in stat_index_groups.items():
            stat_map = load_mrc_image(stat_path)

            # with reference to the exact pixel of the statistic (top-left)
            # need to account for relative extracted box size
            pos_y = self._df.loc[
                indexes,
                "refined_pos_y" if "refined_pos_y" in self._df.columns else "pos_y",
            ].to_numpy()
            pos_x = self._df.loc[
                indexes,
                "refined_pos_x" if "refined_pos_x" in self._df.columns else "pos_x",
            ].to_numpy()
            pos_y = torch.tensor(pos_y)
            pos_x = torch.tensor(pos_x)
            pos_y -= (H - h) // 2
            pos_x -= (W - w) // 2

            cropped_stat_maps = get_cropped_image_regions(
                stat_map,
                pos_y,
                pos_x,
                (H - h + 1, W - w + 1),
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="constant",
                padding_value=0.0,
            )
            stat_stack[indexes] = cropped_stat_maps

        return stat_stack

    def construct_filter_stack(
        self, preprocess_filters: PreprocessingFilters, output_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Get stack of Fourier filters from filter config and reference micrographs.

        Note that here the filters are assumed to be applied globally (i.e. no local
        whitening, etc. is being done). Whitening filters are calculated with reference
        to each original micrograph in the DataFrame.

        Parameters
        ----------
        preprocess_filters : PreprocessingFilters
            Configuration object of filters to apply.
        output_shape : tuple[int, int]
            What shape along the last two dimensions the filters should be.

        Returns
        -------
        torch.Tensor
            The stack of filters with shape (N, h, w) where N is the number of particles
            and (h, w) is the output shape.
        """
        # Create an empty tensor to store the filter stack
        filter_stack = torch.zeros((self.num_particles, *output_shape))

        # Find the indexes in the DataFrame that correspond to each unique image
        image_index_groups = self._df.groupby("micrograph_path").groups

        # Loop over each unique image and extract the particles
        for img_path, indexes in image_index_groups.items():
            img = load_mrc_image(img_path)

            image_dft = torch.fft.rfftn(img)
            image_dft[0, 0] = 0 + 0j
            cumulative_filter = preprocess_filters.get_combined_filter(
                ref_img_rfft=image_dft,
                output_shape=output_shape,
            )

            filter_stack[indexes] = cumulative_filter

        return filter_stack

    @property
    def num_particles(self) -> int:
        """Get the number of particles in the stack."""
        return len(self._df)

    @property
    def absolute_defocus_u(self) -> torch.Tensor:
        """Get the absolute defocus along the major axis."""
        return torch.tensor(
            self._df["defocus_u"] + self._df["refined_relative_defocus"]
        )

    @property
    def absolute_defocus_v(self) -> torch.Tensor:
        """Get the absolute defocus along the minor axis."""
        return torch.tensor(
            self._df["defocus_v"] + self._df["refined_relative_defocus"]
        )

    def __getitem__(self, key: str) -> Any:
        """Get an item from the DataFrame."""
        try:
            return self._df[key]
        except KeyError as err:
            raise KeyError(f"Key '{key}' not found in underlying DataFrame.") from err
