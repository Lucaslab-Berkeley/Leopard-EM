"""Particle stack Pydantic model for dealing with extracted particle data."""

import warnings
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict

from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.utils.data_io import load_mrc_image

TORCH_TO_NUMPY_PADDING_MODE = {
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
}


def _any_nan_or_inf(s: pd.Series) -> bool:
    """Helper function to check if any value in the Series is NaN or infinite.

    Parameters
    ----------
    s : pd.Series
        The Series to check.

    Returns
    -------
    bool
        True if any value in the Series is NaN or infinite, False otherwise.
    """
    return bool(s.isna().any() or s.isin([float("inf"), float("-inf")]).any())


def get_cropped_image_regions(
    image: torch.Tensor | np.ndarray,
    pos_y: torch.Tensor | np.ndarray,
    pos_x: torch.Tensor | np.ndarray,
    box_size: int | tuple[int, int],
    pos_reference: Literal["center", "top-left"] = "center",
    handle_bounds: Literal["pad", "error"] = "pad",
    padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Extracts regions from an image into a stack of cropped images.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        The input image from which to extract the regions.
    pos_y : torch.Tensor | np.ndarray
        The y positions of the regions to extract. Type must mach `image`
    pos_x : torch.Tensor | np.ndarray
        The x positions of the regions to extract. Type must mach `image`
    box_size : int | tuple[int, int]
        The size of the box to extract. If an integer is passed, the box will be square.
    pos_reference : Literal["center", "top-left"], optional
        The reference point for the positions, by default "center". If "center", the
        boxes extracted will be image[y - box_size // 2 : y + box_size // 2, ...]. If
        "top-left", the boxes will be image[y : y + box_size, ...].
    handle_bounds : Literal["pad", "clip", "error"], optional
        How to handle the bounds of the image, by default "pad". If "pad", the image
        will be padded with the padding value based on the padding mode. If "error", an
        error will be raised if any region exceeds the image bounds. Note clipping is
        not supported since returned stack may have inhomogeneous sizes.
    padding_mode : Literal["constant", "reflect", "replicate"], optional
        The padding mode to use when padding the image, by default "constant".
        "constant" pads with the value `padding_value`, "reflect" pads with the
        reflection of the image at the edge, and "replicate" pads with the last pixel
        of the image. These match the modes available in `torch.nn.functional.pad`.
    padding_value : float, optional
        The value to use for padding when `padding_mode` is "constant", by default 0.0.

    Returns
    -------
    torch.Tensor | np.ndarray
        The stack of cropped images extracted from the input image. Type will match the
        input image type.
    """
    if isinstance(box_size, int):
        box_size = (box_size, box_size)

    if pos_reference == "center":
        pos_y = pos_y - box_size[0] // 2
        pos_x = pos_x - box_size[1] // 2
    elif pos_reference == "top-left":
        pass
    else:
        raise ValueError(f"Unknown pos_reference: {pos_reference}")

    if isinstance(image, torch.Tensor):
        return _get_cropped_image_regions_torch(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

    if isinstance(image, np.ndarray):
        padding_mode_np = TORCH_TO_NUMPY_PADDING_MODE[padding_mode]
        return _get_cropped_image_regions_numpy(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode_np,
            padding_value=padding_value,
        )

    raise ValueError(f"Unknown image type: {type(image)}")


def _get_cropped_image_regions_numpy(
    image: np.ndarray,
    pos_y: np.ndarray,
    pos_x: np.ndarray,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: str,
    padding_value: float,
) -> np.ndarray:
    """Helper function for extracting regions from a numpy array.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] // 2
        bs0 = box_size[0] // 2
        image = np.pad(
            image,
            pad_width=((bs0, bs0), (bs1, bs1)),
            mode=padding_mode,
            constant_values=padding_value,
        )
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    cropped_images = np.stack(
        [image[y : y + box_size[0], x : x + box_size[1]] for y, x in zip(pos_y, pos_x)]
    )

    return cropped_images


def _get_cropped_image_regions_torch(
    image: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: Literal["constant", "reflect", "replicate"],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Helper function for extracting regions from a torch tensor.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] // 2
        bs0 = box_size[0] // 2
        image = torch.nn.functional.pad(
            image,
            pad=(bs1, bs1, bs0, bs0),
            mode=padding_mode,
            value=padding_value,
        )
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    # A fix for crops that go out of bounds. Pad with zeros.
    regions = []
    for y, x in zip(pos_y, pos_x):
        # Extract region
        region = image[y : y + box_size[0], x : x + box_size[1]]

        # Check if region is smaller than expected box_size
        h, w = region.shape
        if h != box_size[0] or w != box_size[1]:
            # Create a padded region with the correct dimensions
            padded_region = torch.full(
                box_size, padding_value, dtype=image.dtype, device=image.device
            )
            padded_region[:h, :w] = region
            regions.append(padded_region)
        else:
            regions.append(region)

    # Stack all regions
    cropped_images = torch.stack(regions)

    return cropped_images


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
        """Return a tensor of the specified statistic for each cropped image.

        NOTE: This function is very similar to `construct_image_stack` but returns the
        statistic in one of the result maps. Shape here is (N, H - h + 1, W - w + 1).

        Parameters
        ----------
        stat : Literal["mip", "scaled_mip", "correlation_average",
            "correlation_variance", "defocus", "psi", "theta", "phi"]
            The statistic to extract from the DataFrame.

        Returns
        -------
        torch.Tensor
            The stack of statistics with shape (N, H - h + 1, W - w + 1) where N is the
            number of particles and (H, W) is the extracted box size with (h, w) being
            the original template size.
        """
        stat_col = f"{stat}_path"

        if stat_col not in self._df.columns:
            raise ValueError(f"Statistic '{stat}' not found in the DataFrame.")

        # Create an empty tensor to store the stat stack
        h, w = self.original_template_size
        image_h, image_w = self.extracted_box_size
        stat_stack = torch.zeros((self.num_particles, image_h - h + 1, image_w - w + 1))

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
            pos_y -= (image_h - h) // 2
            pos_x -= (image_w - w) // 2

            cropped_stat_maps = get_cropped_image_regions(
                stat_map,
                pos_y,
                pos_x,
                (image_h - h + 1, image_w - w + 1),
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

            image_dft = torch.fft.rfftn(img)  # pylint: disable=not-callable
            image_dft[0, 0] = 0 + 0j
            cumulative_filter = preprocess_filters.get_combined_filter(
                ref_img_rfft=image_dft,
                output_shape=output_shape,
            )

            filter_stack[indexes] = cumulative_filter

        return filter_stack

    @property
    def df_columns(self) -> list[str]:
        """Get the columns of the DataFrame."""
        return list(self._df.columns.tolist())

    @property
    def num_particles(self) -> int:
        """Get the number of particles in the stack."""
        return len(self._df)

    def get_absolute_defocus(
        self, prefer_refined_defocus: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the absolute defocus values for each particle.

        NOTE: If the refined defocus values are requested but not present in the
        DataFrame (either no column or any NaN values), a user warning is raised and the
        original defocus values are returned instead.

        Parameters
        ----------
        prefer_refined_defocus : bool, optional
            Whether to use the refined defocus values (columns prefixed with 'refined_')
            or not, by default True.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple of two tensors containing the absolute defocus values along the
            major (defocus_u) and minor axes (defocus_v), respectively in units of
            Angstroms.

        Warnings
        --------
            Warns if NaN values or no column present for either 'refined_defocus_u' or
            'refined_defocus_v'. Falls back to the unrefined values.
        """
        rel_defocus_col = "relative_defocus"

        # Both refined columns must be present AND no values can be NaN or inf
        if prefer_refined_defocus:
            if "refined_relative_defocus" not in self._df.columns:
                warnings.warn(
                    "Refined defocus values not found in DataFrame, using original "
                    "defocus values...",
                    stacklevel=2,
                )
            elif _any_nan_or_inf(self._df["refined_relative_defocus"]):
                warnings.warn(
                    "Refined defocus values contain NaN or inf values, using original "
                    "defocus values...",
                    stacklevel=2,
                )
            else:
                rel_defocus_col = "refined_relative_defocus"

        # Get the defocus values from the DataFrame
        particle_defocus = torch.tensor(self._df[rel_defocus_col].to_numpy())
        defocus_u = torch.tensor(self._df["defocus_u"].to_numpy()) + particle_defocus
        defocus_v = torch.tensor(self._df["defocus_v"].to_numpy()) + particle_defocus

        return defocus_u, defocus_v

    def get_euler_angles(self, prefer_refined_angles: bool = True) -> torch.Tensor:
        """Return the Euler angles (phi, theta, psi) of all particles as a tensor.

        Parameters
        ----------
        prefer_refined_angles : bool, optional
            When true, the refined Euler angles are used (columns prefixed with
            'refined_'), otherwise the original angles are used, by default True.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 3) where N is the number of particles and the columns
            correspond to (phi, theta, psi) in ZYZ format.
        """
        # Ensure all three refined columns are present, warning if not
        phi_col = "phi"
        theta_col = "theta"
        psi_col = "psi"
        if prefer_refined_angles:
            if not all(
                x in self._df.columns
                for x in ["refined_phi", "refined_theta", "refined_psi"]
            ):
                warnings.warn(
                    "Refined angles not found in DataFrame, using original angles...",
                    stacklevel=2,
                )
            else:
                phi_col = "refined_phi"
                theta_col = "refined_theta"
                psi_col = "refined_psi"

        # Get the angles from the DataFrame
        phi = torch.tensor(self._df[phi_col].to_numpy())
        theta = torch.tensor(self._df[theta_col].to_numpy())
        psi = torch.tensor(self._df[psi_col].to_numpy())

        return torch.stack((phi, theta, psi), dim=-1)

    def __getitem__(self, key: str) -> Any:
        """Get an item from the DataFrame."""
        try:
            return self._df[key]
        except KeyError as err:
            raise KeyError(f"Key '{key}' not found in underlying DataFrame.") from err

    def set_column(self, column_name: str, value: Any) -> None:
        """Set a column in the underlying DataFrame.

        Parameters
        ----------
        column_name : str
            The name of the column to set
        value : Any
            The value to set the column to
        """
        self._df.loc[:, column_name] = value

    def get_dataframe_copy(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
        A copy of the underlying DataFrame
        """
        return self._df.copy()
