# Workflow

## Training data
- Generate training data for the desired process with PYTHIA. We have been using 2M events as a default.
- Extract the per-branching events. Harry uses `extract_p4_Nbranchs.py`, Yoni uses `pythia_snapshot_common.py`. These are both AI-generated and make different choices, we should double-check that they give the same result. Both segment the branching tree and output a separate PyTorch or Numpy file for each number of final-state particles.
This can get quite large so make sure there is enough storage available before you launch the process. 
- **Normalize the data** so that each event has **unit energy** and save as a `(NEvents, NParticles, 3)` PyTorch tensor of 3-vectors. All partons are massless by construction so we only need to save the 3-momentum.

## Diffusion models
- Precompute the forward diffusion process checkpoints using `precompute_forward.py` or the SLURM script `precompute.sh`. This saves time during training but takes up a lot of space (12 particles x 2M events x 100 checkpoints = 30 GB), so put these files in your scratch folder.
  Note that the arguments for this script involve the diffusion schedule ONLY: there is no neural network here yet. The default option is to save the starting data in q-space, which involves a random augmentation by the inverse RAMBO map.
- Train a score network using `diffusion.py` or `train.sh`. Right now, the diffusion schedule options need to be set by hand rather than pulling from the pre-computed events; this should be fixed ASAP so there is never any confusion.
  Setting up a Weights & Biases account will let you watch the progress of various metrics during training. With the current setup, the only one that really matters is the training loss, which should decrease quasi-monotonically during training: the validation loss is basically uninformative.
  Our default is to train each model for 12 hours using 4 GPUs on a single node of the Vector cluster, and take the last checkpoint as the trained model.
  - **To Do:** add Rikab's SEMD comparison as a validation metric during training
- Check that the trained model looks good by generating some events using `generate_samples.py` or `generate.sh`.
  Note that because of stupid GPU memory issues, the parameter `BATCH_SIZE` must be less than or equal to `65535/NParticles`. This doesn't affect the number of events that can be generated, only how they're split up into batches on the GPU. For 10 particles, this takes about 2 hours for 50k events.
  An example of the kinds of checks one can do is in `CheckGeneratedEvents.ipynb`.

## Forward-backward experiments
- Generate "round-trip" events with `forward_backward_sample.py` or `generate_fb_roundtrip.sh`. This script takes as input the q-space training data `Q0.pt`, applies the forward diffusion process for `T_FORWARD` steps to the first `N_SAMPLES` events for `N_TRAJ` independent trajectories, then uses the trained score model to denoise back to t = 0.
  The defaults we've been using are 256 sample points, each with 128 diffusion trajectories.
- Compute susceptibilities using the notebook `susceptibility_YK.ipynb`. To ensure there is no potential issue with permutation of the event list, this notebook compares the original un-noised q-space events in `Q0.pt` to the round-trip q-space events, converting both back to p-space with the RAMBO map `qspace.qs_to_ps`.

