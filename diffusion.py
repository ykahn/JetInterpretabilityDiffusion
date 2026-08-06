import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .qspace import qspace_score, sample_qspace


def build_gamma_schedule(
    t_steps: int,
    gamma_min: float,
    gamma_max: float,
    schedule_type: str = "linear",
    t_gaus: int = 0,
    gamma_gaus: float = 0.0001,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    if device is None:
        device = torch.device("cpu")

    if not (0 <= t_gaus < t_steps):
        raise ValueError(f"t_gaus={t_gaus} must satisfy 0 <= t_gaus < t_steps={t_steps}")

    T_score = t_steps - t_gaus
    linear = torch.linspace(gamma_min, gamma_max, T_score, device=device, dtype=dtype)
    target_total = linear.sum()

    if schedule_type == "linear":
        score_sched = linear
    else:
        t = torch.linspace(0, 1, T_score, device=device, dtype=dtype)

        if schedule_type == "geometric":
            sched = gamma_min * (gamma_max / gamma_min) ** t
        elif schedule_type == "cosine":
            s = (1 - torch.cos(t * math.pi)) / 2
            sched = gamma_min + (gamma_max - gamma_min) * s
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        score_sched = sched * (target_total / sched.sum())

    if t_gaus > 0:
        gaus_part = torch.full((t_gaus,), gamma_gaus, device=device, dtype=dtype)
        return torch.cat([gaus_part, score_sched])

    return score_sched


def forward_step(Q, gamma, noise=None, noscore=False):
    if noise is None:
        noise = torch.randn_like(Q)
    if noscore:
        Q_next = Q - gamma * Q + torch.sqrt(2 * gamma) * noise
    else:
        score_ref = qspace_score(Q)
        Q_next = Q + gamma * score_ref + torch.sqrt(2 * gamma) * noise
    return Q_next, noise


def forward_process(Q0, gammas, t_gaus=0):
    Q = Q0.clone()
    for t in range(len(gammas)):
        Q, _ = forward_step(Q, gammas[t], noscore=(t < t_gaus))
    return Q


def precompute_forward_samples(
    Q0_all: torch.Tensor,
    gammas: torch.Tensor,
    t_gaus: int = 0,
    verbose: bool = True,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, float]]:
    T = len(gammas)
    cached: Dict[int, Tuple[torch.Tensor, torch.Tensor, float]] = {}
    Q = Q0_all.clone()

    with torch.no_grad():
        for s in tqdm(range(T), desc="Pre-computing forward process", disable=not verbose):
            Q, noise = forward_step(Q, gammas[s], noscore=(s < t_gaus))
            cached[s + 1] = (Q.clone(), noise.clone(), (s + 1) / T)

    return cached


@torch.no_grad()
def reverse_step(Q, t_normalized, gamma, score_net, noscore=False, predict_eps=False):
    net_out = score_net(Q, t_normalized)
    if predict_eps:
        score_model = -net_out / math.sqrt(2.0 * gamma)
    else:
        score_model = net_out
    noise = torch.randn_like(Q)
    if noscore:
        return Q + gamma * Q + 2 * gamma * score_model + torch.sqrt(2 * gamma) * noise
    else:
        score_ref = qspace_score(Q)
        return Q - gamma * score_ref + 2 * gamma * score_model + torch.sqrt(2 * gamma) * noise


@torch.no_grad()
def sample(
    score_net,
    n_samples: int,
    n_particles: int,
    gammas: torch.Tensor,
    t_gaus: int = 0,
    device: torch.device = None,
    seed: int = -1,
    verbose: bool = True,
    predict_eps: bool = False,
) -> torch.Tensor:
    if seed >= 0:
        torch.manual_seed(seed)
    if device is None:
        device = gammas.device

    T = len(gammas)
    Q = sample_qspace(
        nevents=n_samples,
        nparticles=n_particles,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )

    gammas_reversed = gammas.flip(0)
    for s in tqdm(range(T), desc="Sampling", disable=not verbose):
        t_normalized = torch.full(
            (n_samples,), (T - s) / T, device=device, dtype=torch.float32
        )
        noscore = s >= T - t_gaus
        Q = reverse_step(
            Q, t_normalized, gammas_reversed[s], score_net,
            noscore=noscore, predict_eps=predict_eps,
        )

    return Q

def sample_fromQ_at_t(
    score_net,
    Q,
    forward_t: int,
    gammas: torch.Tensor,
    t_gaus: int = 0,
    device: torch.device = None,
    seed: int = -1,
    verbose: bool = True,
    predict_eps: bool = False,
) -> torch.Tensor: 
    if seed >= 0:
        torch.manual_seed(seed)
    if device is None:
        device = gammas.device

    T = len(gammas)
    n_samples = Q.shape[0]
    gammas_reversed = gammas.flip(0)
    for s in tqdm(range(T-forward_t,T), desc="Sampling", disable=not verbose): #start reverse process from forward_t
        t_normalized = torch.full( #everything else is identical to the normal "sample" function
        (n_samples,), (T - s) / T, device=device, dtype=torch.float32
        )
        noscore = s >= T - t_gaus
        Q = reverse_step(
        Q, t_normalized, gammas_reversed[s], score_net,
        noscore=noscore, predict_eps=predict_eps,
        )

    return Q
