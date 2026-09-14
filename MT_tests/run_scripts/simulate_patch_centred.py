"""Simulate the 4-patch template centred on the microtubule axis.

Same parameters as process_GMPCPP.ipynb; the only change is the input PDB, which has
been translated so the tube axis -- not the atom centroid -- sits at the origin.
"""

from ttsim3d.models import Simulator, SimulatorConfig

sim_conf = SimulatorConfig(
    voltage=300.0,
    apply_dose_weighting=True,
    dose_start=0.0,
    dose_end=50.0,
    dose_filter_modify_signal="rel_diff",
    upsampling=-1,
    mtf_reference="k3_300kV_FL2",
)

sim = Simulator(
    pdb_filepath="models/6dpu_4_patches_axis_centred.pdb",
    pixel_spacing=0.9194,
    volume_shape=(600, 600, 600),
    center_atoms=False,
    remove_hydrogens=True,
    b_factor_scaling=0.5,
    additional_b_factor=0,
    simulator_config=sim_conf,
)

if __name__ == "__main__":
    sim.run()
    sim.export_to_mrc("maps/GMPCPP_4patches_centred_0.9194_bscale0.5.mrc")
    print("wrote maps/GMPCPP_4patches_centred_0.9194_bscale0.5.mrc")
