# Null calibration

Sets the detection threshold of a match_template configuration for a target number of
false positives per micrograph, from null searches instead of the independent-trials
formula:

* **phase-randomised null**: phase-randomised copies of the micrograph (same Fourier
  amplitudes, random phases) are searched with the unchanged configuration, and their
  distinct peaks set the threshold;
* **analytic null**: the null peak rate is predicted from the template, filters, grids
  and the micrograph's power spectrum, without a search.

Edit `null_calibration_example_config.yaml`, point `YAML_CONFIG_PATH` in
`run_null_calibration.py` at it and run the script. The thresholds are written to
`null_summary.json` in the output directory; pass one to
`results_to_dataframe(locate_peaks_kwargs={"z_score_cutoff": ...})` when picking the
real micrograph. See `docs/programs/null_calibration.md` for details and validation.
