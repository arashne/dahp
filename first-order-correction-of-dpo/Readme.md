# First-Order Correction of DPO and Synthetic Experiments (Section 7.2)

First, install the required packages in `requirements.txt`. 
Also create `data` and `figs` folders. This `data` folder one is the default path to store the trained policies and the `figs` folder is default location to store graphics. 

To train a policy, we follow `scripts/run_offline_simulator_trainer.py` 
You only need to specify the `solver_class` in this file to reproduce the results in the paper. Use:
* `solver_class = DPO` to run standard DPO,
* `solver_class = EstVarCorrectedDPO` to run first-order correction of DPO (Section 5),
* `solver_class = ShiraliEtAl` to use a consistent loss function (Section 6).

After training, use `scripts/post_process_offline_simulator_trainer.py` to visualize a single policy and 
`scripts/post_process_offline_simulator_trainer_compare.py` to compare multiple ones.
These scripts allow you to choose whether you want to save the figures (`save_figs=True`) or whether you want to plot NBC on the plots as well (`plot_nbc=True`).


