import torch
import numpy as np
import sys
import omnilearned
sys.path.append("/home/yfkahn/projects/aip-yfkahn/shared/jet_interpretability_diffusion/GenerativeModelsOnPhaseSpace-main")
from omnilearn_lightning.model import PETLightning
from omnilearn_lightning import diffusion
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate forward-backward samples from trained diffusion model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--q-path", type=str, required=True, help="Path to q-space checkpoints")
    parser.add_argument("--t-forward", type=int, required=True, help="Forward step")
    parser.add_argument("--n-samples", type=int, required=True, help="Total number of samples to generate")
    parser.add_argument("--n-traj", type=int, required=True, help="Total number of reverse trajectories to generate")
    parser.add_argument("--save-head", type=str, required=True, help="Path heading to save reverse-process Qs")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model=PETLightning.load_from_checkpoint(args.model_path,map_location="cpu")
    model = model.to(device).eval()
    print(f"Model hparams:")
    for k, v in sorted(model.hparams.items()):
        print(f"  {k}: {v}")

    metadat = torch.load(args.q_path+'metadata.pt')
    Ncheckpoints=len(metadat['checkpoint_steps'])
    Ntot=metadat['N']
    Nparticles=metadat['n_particles']
    qspace_dat=np.memmap(args.q_path+'Q_checkpoints.npy',
                         dtype="float32",mode="r",shape=(Ncheckpoints,Ntot,Nparticles,3))

    #load the starting q-space data
    q0=np.array(qspace_dat[0,:args.n_samples])
    q0=torch.as_tensor(q0).to(device)
    if args.t_forward > metadat['T']:
        print('Forward time exceeds total diffusion time, setting this to Tmax')
        t_forward = metadat['T']
    else:
        t_forward = args.t_forward
    score_net=model.model
    Q_rev = torch.zeros((args.n_traj,args.n_samples,Nparticles,3))

    for i in range(args.n_traj):
        Q_forward = diffusion.forward_process(
            q0,metadat['gammas'][:t_forward].to(device),
            t_gaus=metadat['t_gaus'])
        
        Q_rev[i] = diffusion.sample_fromQ_at_t(
            score_net,
            Q_forward,
            t_forward,
            gammas=metadat['gammas'].to(device),
            t_gaus=metadat['t_gaus'],
            device=device)

    torch.save(Q_rev, args.save_head+'_qspace_roundtrip_T'+str(t_forward)+'.pt')

if __name__ == "__main__":
    main()
    
