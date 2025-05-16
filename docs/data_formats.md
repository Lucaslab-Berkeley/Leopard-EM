# Description of Data Formats

To increase interoperability between external packages, we herein describe the different data formats used as input to and export from programs.
Orientations on a per-particle bases are currently stored as [Euler angles](https://en.wikipedia.org/wiki/Euler_angles#) in ZYZ format with angles ordered as \( \phi, \theta, \psi \).
Note that the package is still under development and the exact way the data is represented might change in the future.

## Data from the match template program

The match template program collates statistics from a large number of cross-correlograms taken over an orientation and defocus search space.
See the API on the `MatchTemplateResult` object for further information on how these data are stored, but here we provide an overview of what files get written to disk.

### Best statistic maps
Each of the "best" statistics is stored on a per-position basis in what we dub "statistics maps" saved as `.mrc` files.
We have the following tracked statistics for each valid (x, y) position:

- Maximum Intensity Projection (MIP): Maximum cross-correlation value over the entire search space.
- Scaled MIP (z-score): The MIP value normalized by the mean and variance of the cross-correlation over the entire search space.
- Correlation Mean: The mean of the cross-correlation values over the entire search space. Used to calculate the scaled MIP.
- Correlation Variance: The variance of the cross-correlation values over the entire search space. Used to calculate the scaled MIP.
- Phi: The \( \phi \) angle (in degrees) which produced the MIP value.
- Theta: The \( \theta \) angle (in degrees) which produced the MIP value.
- Psi: The \( \psi \) angle (in degrees) which produced the MIP value.
- Defocus: The relative defocus value (in Angstroms, relative to estimated defocus of micrograph) which produced the MIP value.

Each of these statistics maps are saved to disk in the [MRC format](https://www.ccpem.ac.uk/mrc-format/mrc2014/) based on paths provided in the `MatchTemplateResult` object.

#### A note on correlation modes and output shapes

Three general modes for convolution/correlation exist in digital signal processing: "full", "same", and "valid".
[This chapter of Digital Signals Theory](https://brianmcfee.net/dstbook-site/content/ch03-convolution/Modes.html) provides a good overview of these modes.

We use the the "valid" mode by default when saving these statistics maps, but they are initially stored in their "same" modes.
The `MatchTemplateResult.apply_valid_cropping` method does this "same" to "valid" cropping.
For an image with shape \( (H, W) \) and template \( (h, w) \), the modes will output statistics maps with the following shapes:

- same: \( (H, W) \)
- valid: \( (H - h + 1, W - w + 1) \)

Note that same mode pads the image with zeros along the edges and *does not* increase the number of particles detectable; values along the padded portions of the edge do not hold significance in the context of the particle detection.
In each case, the position in the map at \( (i, j) \) corresponds to the top-left corner of the template at that position, *not* the center of the template.

<!-- TODO: Figure out how to properly style the svg so it fits on the page -->
<!-- Below is an example comparing the two correlation modes:

<img src="../static/correlation_modes.svg" alt="Diagram showing correlation modes and output sizes" style="width: 50%;"> -->

### Match template **DataFrame**

Since not all positions \( (x, y) \) contain a particle from the match template search, using full statistics maps for downstream analysis can be inefficient in terms of speed, memory requirements, and code overhead.
The match template manager class has the method `MatchTemplateManager.results_to_dataframe()` which automatically picks peaks within the scaled MIP map and stores the peak locations, orientations, and defocus values in a pandas DataFrame.
We take a verbose approach to constructing this DataFrame where some columns store similar information about each particle.

Additional columns besides locations and orientations are included in the DataFrame to increase the utility of the data, namely the construction of `ParticleStack` objects.
The columns and corresponding descriptions are as follows:

| Column Name                   | Type  | Description |
|-------------------------------|-------|-------------|
| `mip`                         | float | Maximum cross-correlation value over all search orientations and relative defocus values.
| `scaled_mip`                  | float | Scaled MIP value (z-score) normalized by cross-correlation mean and variance.
| `correlation_mean`            | float | Mean of the cross-correlation values over the entire search space.
| `correlation_variance`        | float | Variance of the cross-correlation values over the entire search space.
| `total_correlations`          | int   | Total number of cross-correlations taken over the search space (defoci times orientations).
| `pos_x`                       | int   | Particle x position (units of pixels) in the statistics maps. Corresponds to the top-left corner of the template.
| `pos_y`                       | int   | Particle y position (units of pixels) in the statistics maps. Corresponds to the top-left corner of the template.
| `pos_x_img`                   | int   | Center of of the particle (x position, units of pixels) in the micrograph.
| `pos_y_img`                   | int   | Center of of the particle (y position, units of pixels) in the micrograph.
| `pos_x_img_angstrom`          | float | Center of the particle (x position, in Angstroms) in the micrograph.
| `pos_y_img_angstrom`          | float | Center of the particle (y position, in Angstroms) in the micrograph.
| `psi`                         | float | The \( \psi \) angle (in degrees) which produced the MIP value.
| `theta`                       | float | The \( \theta \) angle (in degrees) which produced the MIP value.
| `phi`                         | float | The \( \phi \) angle which (in degrees) produced the MIP value.
| `relative_defocus`            | float | The relative defocus value (in Angstroms) which produced the MIP value. Relative to `defocus_u` and `defocus_v`.
| `defocus_u`                   | float | Defocus value along the major axis for the **micrograph** (in Angstroms).
| `defocus_v`                   | float | Defocus value along the minor axis for the **micrograph** (in Angstroms).
| `astigmatism_angle`           | float | Angle of the astigmatism (in degrees) for defocus.
| `pixel_size`                  | float | Pixel size of the micrograph (in Angstroms).
| `voltage`                     | float | Voltage of the microscope (in kV).
| `spherical_aberration`        | float | Spherical aberration of the microscope (in mm).
| `amplitude_contrast_ratio`    | float | Amplitude contrast ratio of the microscope.
| `phase_shift`                 | float | Phase shift of the microscope (in degrees).
| `ctf_B_factor`                | float | B-factor of the CTF, in Angstroms^2.
| `micrograph_path`             | str   | Path to the original micrograph.
| `template_path`               | str   | Path to the template used for the search.
| `mip_path`                    | str   | Path to the saved MIP map.
| `scaled_mip_path`             | str   | Path to the saved scaled MIP map.
| `psi_path`                    | str   | Path to the saved psi map.
| `theta_path`                  | str   | Path to the saved theta map.
| `phi_path`                    | str   | Path to the saved phi map.
| `defocus_path`                | str   | Path to the saved defocus map.
| `correlation_average_path`    | str   | Path to the saved correlation mean map.
| `correlation_variance_path`   | str   | Path to the saved correlation variance map.

## Data from the refine template program

The refine template program takes in the DataFrame from the match template program and refines the orientation & defocus values of each particle.
Each of the refined parameters are stored in new columns prefixed with the `refined_` string.
Note that refined results can be re-refined, for example with a slightly different template, and the already refined parameters will be used.

### Refine template **DataFrame**
The program outputs another DataFrame with additional columns for the refined orientations, defocus values, and positions.
New columns with descriptions are listed below:

| Column Name                   | Type  | Description |
|-------------------------------|-------|-------------|
| `refined_mip`                 | float | New maximum cross-correlation over refinement search space.
| `refined_scaled_mip`          | float | New scaled MIP value (z-score) normalized by cross-correlation mean and variance.
| `refined_pos_x`               | int   | The refined x position of the particle, top-left corner of the template.
| `refined_pos_y`               | int   | The refined y position of the particle, top-left corner of the template.
| `refined_pos_x_img`           | int   | The refined x position of the particle, center of the particle in the micrograph.
| `refined_pos_y_img`           | int   | The refined y position of the particle, center of the particle in the micrograph.
| `refined_pos_x_img_angstrom`  | float | The refined x position of the particle, center of the particle in the micrograph (in Angstroms).
| `refined_pos_y_img_angstrom`  | float | The refined y position of the particle, center of the particle in the micrograph (in Angstroms).
| `refined_psi`                 | float | The refined \( \psi \) angle (in degrees).
| `refined_theta`               | float | The refined \( \theta \) angle (in degrees).
| `refined_phi`                 | float | The refined \( \phi \) angle (in degrees).
| `refined_relative_defocus`    | float | The refined relative defocus value (in Angstroms).