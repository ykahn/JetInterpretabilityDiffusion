#!/usr/bin/env python3
"""
Extract intermediate PYTHIA QCD-shower states using color-line topology instead
of shower scales.

Main differences from the scale-ordered extractor:
  * QCD branchings are identified by PID + color-flow pattern, not by scale.
  * Resonance decays such as t -> W b and W -> q qbar are not counted as QCD
    shower branchings, although their colored daughters are used as shower seeds.
  * Branch ordering is by PYTHIA event-record creation order of the daughters,
    after filtering to color-compatible QCD branchings. This avoids using scale.
  * Recoiler copies are updated using PYTHIA copy links and color similarity,
    not by matching same-scale rows.

Inputs:
  1. Raw PYTHIA logs containing custom blocks headed by
       PYTHIA Event Listing  (complete event with scale)
     These blocks must include color1/color2 columns.

  2. parse_pythia_log.py .pt dict output. The dict must contain colors under one
     of: colors, colours, color. Shape should be (..., 2).

Internal p4 convention is [E, px, py, pz].
"""

from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Any
import math

import torch
from tqdm import tqdm


@dataclass
class Particle:
    no: int
    pid: int
    status: int
    p4: torch.Tensor          # [E, px, py, pz]
    mothers: tuple[int, int]
    daughters: tuple[int, int]
    colors: tuple[int, int]   # (color, anticolor)
    scale: float              # kept only for diagnostics / optional saving


# -----------------------------------------------------------------------------
# Input loading/parsing
# -----------------------------------------------------------------------------

def load_torch_object(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_color_tensor_from_pt_dict(data: dict[str, torch.Tensor]) -> torch.Tensor:
    for key in ["colors", "colours", "color", "colour"]:
        if key in data:
            colors = data[key]
            if colors.shape[-1] != 2:
                raise ValueError(f"data[{key!r}] should have last dimension 2, got {tuple(colors.shape)}")
            return colors
    raise KeyError(
        "This color-line extractor needs color tags in the .pt dict. "
        "Expected one of keys: colors, colours, color, colour. "
        "For raw logs, color tags are parsed from color1/color2 columns."
    )


def build_particles_for_event_from_pt(data: dict[str, torch.Tensor], i: int) -> dict[int, Particle]:
    colors_all = get_color_tensor_from_pt_dict(data)

    mask_i = data["mask"][i].bool()
    nos_i = data["no"][i]
    pid_i = data["pid"][i]
    status_i = data["status"][i]
    p4_i = data["p4"][i].float()
    mothers_i = data["mothers"][i]
    daughters_i = data["daughters"][i]
    scale_i = data["scale"][i].float()
    colors_i = colors_all[i]

    particles: dict[int, Particle] = {}

    for j in torch.where(mask_i)[0].tolist():
        no = int(nos_i[j].item())
        if no <= 0:
            continue

        particles[no] = Particle(
            no=no,
            pid=int(pid_i[j].item()),
            status=int(status_i[j].item()),
            p4=p4_i[j].clone(),
            mothers=(int(mothers_i[j, 0].item()), int(mothers_i[j, 1].item())),
            daughters=(int(daughters_i[j, 0].item()), int(daughters_i[j, 1].item())),
            colors=(int(colors_i[j, 0].item()), int(colors_i[j, 1].item())),
            scale=float(scale_i[j].item()),
        )

    return particles


def _parse_int_token(tok: str) -> int | None:
    try:
        return int(tok)
    except ValueError:
        return None


def parse_pythia_log_with_scale(path: Path, *, max_events: int | None = None) -> list[dict[int, Particle]]:
    """
    Parse custom PYTHIA blocks like:

      -------- PYTHIA Event Listing  (complete event with scale)  --------
       no id name status mother1 mother2 daughter1 daughter2 color1 color2 px py pz e m scale pTbeam
        0 90 system -11 ...
      -------- End PY

    Returns one dictionary no -> Particle per event.
    """
    start_marker = "PYTHIA Event Listing  (complete event with scale)"
    # Your custom code sometimes prints "-------- End PY" rather than the full marker.
    end_markers = [
        "End PYTHIA Event Listing  (complete event with scale)",
        "-------- End PY",
    ]

    events: list[dict[int, Particle]] = []
    current: dict[int, Particle] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()

            if start_marker in line and not any(m in line for m in end_markers):
                current = {}
                continue

            if current is None:
                continue

            if any(m in line for m in end_markers):
                if current:
                    events.append(current)
                    if max_events is not None and len(events) >= max_events:
                        break
                current = None
                continue

            if not line or line.startswith("no ") or line.startswith("no\t"):
                continue

            toks = line.split()
            if len(toks) < 17:
                continue

            no = _parse_int_token(toks[0])
            if no is None:
                continue

            try:
                pid = int(toks[1])
                status = int(toks[3])
                m1 = int(toks[4])
                m2 = int(toks[5])
                d1 = int(toks[6])
                d2 = int(toks[7])
                c1 = int(toks[8])
                c2 = int(toks[9])
                px = float(toks[10])
                py = float(toks[11])
                pz = float(toks[12])
                e = float(toks[13])
                scale = float(toks[15])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Could not parse PYTHIA row in {path}:\n{line}") from exc

            if no <= 0:
                continue

            current[no] = Particle(
                no=no,
                pid=pid,
                status=status,
                p4=torch.tensor([e, px, py, pz], dtype=torch.float32),
                mothers=(m1, m2),
                daughters=(d1, d2),
                colors=(c1, c2),
                scale=scale,
            )

    if not events:
        raise RuntimeError(
            f"No 'complete event with scale' blocks found in {path}. "
            "Make sure your PYTHIA log prints those custom blocks."
        )

    return events


def infer_input_format(path: Path, obj: Any | None) -> str:
    suffix = path.suffix.lower()
    if suffix in {".log", ".txt", ".out"}:
        return "log"
    if isinstance(obj, dict):
        return "pt-dict"
    raise ValueError("Could not infer input format. Use --input-format log or --input-format pt-dict.")


# -----------------------------------------------------------------------------
# Color-line QCD branch identification
# -----------------------------------------------------------------------------

QUARKS = {1, 2, 3, 4, 5}
RESONANCES_TO_SKIP_AS_QCD = {6, 23, 24, 25}  # t, Z, W, H


def is_quark(pid: int) -> bool:
    return abs(pid) in QUARKS


def is_gluon(pid: int) -> bool:
    return pid == 21


def is_colored_shower_parton(pid: int) -> bool:
    # Exclude top from shower seeds/branching parents: treat it as a resonance.
    return is_gluon(pid) or is_quark(pid)


def nz_colors(p: Particle) -> list[int]:
    return [c for c in p.colors if c != 0]


def color_counts(ps: list[Particle]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for p in ps:
        for c in nz_colors(p):
            counts[c] = counts.get(c, 0) + 1
    return counts


def can_have_two_daughters(p: Particle, particles: dict[int, Particle]) -> bool:
    d1, d2 = p.daughters
    return d1 > 0 and d2 > 0 and d1 != d2 and d1 in particles and d2 in particles


def qcd_pid_pattern(parent: Particle, child1: Particle, child2: Particle) -> bool:
    p = parent.pid
    a = child1.pid
    b = child2.pid

    # q -> q g or qbar -> qbar g. Preserve quark flavor/sign.
    if is_quark(p):
        return (a == p and is_gluon(b)) or (b == p and is_gluon(a))

    if is_gluon(p):
        # g -> g g
        if is_gluon(a) and is_gluon(b):
            return True
        # g -> q qbar of same flavor
        if is_quark(a) and is_quark(b) and a == -b:
            return True

    return False


def color_compatible_qcd_branch(parent: Particle, child1: Particle, child2: Particle) -> bool:
    """
    Conservative color-flow checks for PYTHIA-style color tags.
    Falls back to PID pattern when tags are missing/zero.
    """
    if not qcd_pid_pattern(parent, child1, child2):
        return False

    pc = nz_colors(parent)
    cc = color_counts([child1, child2])

    # If color tags are unavailable, do not veto a PID-compatible branch.
    if not pc:
        return True

    # q/qbar -> q/qbar + g: parent endpoint should remain present, and a new
    # color line should appear twice between daughter quark and gluon.
    if is_quark(parent.pid):
        parent_endpoint = parent.colors[0] if parent.pid > 0 else parent.colors[1]
        if parent_endpoint == 0:
            return True
        parent_seen = cc.get(parent_endpoint, 0) >= 1
        has_new_repeated = any(c != parent_endpoint and n >= 2 for c, n in cc.items())
        return parent_seen and has_new_repeated

    # g -> g g: both parent endpoints should be present, and a new internal
    # color line should appear twice.
    if is_gluon(parent.pid) and is_gluon(child1.pid) and is_gluon(child2.pid):
        c, a = parent.colors
        endpoints_seen = (c == 0 or cc.get(c, 0) >= 1) and (a == 0 or cc.get(a, 0) >= 1)
        has_new_repeated = any(col not in {c, a, 0} and n >= 2 for col, n in cc.items())
        return endpoints_seen and has_new_repeated

    # g -> q qbar: quark color and antiquark anticolor should match the two
    # gluon endpoints, up to ordering conventions.
    if is_gluon(parent.pid) and is_quark(child1.pid) and is_quark(child2.pid) and child1.pid == -child2.pid:
        q = child1 if child1.pid > 0 else child2
        qb = child2 if child1.pid > 0 else child1
        return q.colors[0] in parent.colors and qb.colors[1] in parent.colors

    return False


def is_qcd_branch(
    p: Particle,
    particles: dict[int, Particle],
    *,
    loose_color: bool = False,
) -> bool:
    if abs(p.pid) in RESONANCES_TO_SKIP_AS_QCD:
        return False
    if not is_colored_shower_parton(p.pid):
        return False
    if not can_have_two_daughters(p, particles):
        return False

    d1, d2 = p.daughters
    child1 = particles[d1]
    child2 = particles[d2]

    if not qcd_pid_pattern(p, child1, child2):
        return False

    if loose_color:
        return True

    return color_compatible_qcd_branch(p, child1, child2)


def build_qcd_edges(
    particles: dict[int, Particle],
    *,
    loose_color: bool = False,
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for p in particles.values():
        if is_qcd_branch(p, particles, loose_color=loose_color):
            d1, d2 = p.daughters
            edges.append((p.no, d1))
            edges.append((p.no, d2))
    return edges


def choose_frontier_roots(
    particles: dict[int, Particle],
    *,
    event_type: str = "ee2ttbar",
) -> list[int]:
    """
    Roots for the *event-record frontier*, not QCD shower roots.

    For ttbar we start from the hard top and antitop rows, so the first
    four-particle frontier in event 0 is 40,39,8,7 after t/tbar decays.
    For qqbar this usually starts from the hard q/qbar rows.
    """
    if event_type == "ee2ttbar":
        roots = [n for n in [6, 5] if n in particles]
        if roots:
            return roots

    if event_type == "ee2qqbar":
        roots = [n for n in [6, 5] if n in particles]
        if roots:
            return roots

    # Generic fallback: hard outgoing particles with |status| == 22 or 23.
    # Use descending event number so anti-particle side is usually first.
    roots = [
        p.no for p in particles.values()
        if abs(p.status) in {22, 23} and p.no > 0
    ]
    if roots:
        return sorted(set(roots), reverse=True)

    # Last fallback: colored/nontrivial particles without selected mothers.
    return sorted(
        [p.no for p in particles.values() if p.no > 0 and p.daughters != (0, 0)],
        reverse=True,
    )


def is_real_split(p: Particle, particles: dict[int, Particle]) -> bool:
    """A real frontier branching: two distinct daughters present in the record."""
    return can_have_two_daughters(p, particles)


def is_copy_link(p: Particle, particles: dict[int, Particle]) -> bool:
    d1, d2 = p.daughters
    return d1 > 0 and d1 == d2 and d1 in particles


# -----------------------------------------------------------------------------
# Recoiler update without using scale
# -----------------------------------------------------------------------------

def is_copy_update(old: Particle, cand: Particle) -> bool:
    """PYTHIA copies often have old.daughters=(new,new) and cand.mothers=(old,old)."""
    if cand.no <= old.no:
        return False
    if cand.no in {0, old.no}:
        return False
    if abs(cand.pid) != abs(old.pid):
        return False

    od1, od2 = old.daughters
    cm1, cm2 = cand.mothers

    if od1 == od2 == cand.no:
        return True
    if cm1 == cm2 == old.no:
        return True
    if old.no in {cm1, cm2} and cand.no in {od1, od2} and od1 == od2:
        return True
    return False


def shared_color_count(a: Particle, b: Particle) -> int:
    return len(set(nz_colors(a)) & set(nz_colors(b)))


def find_recoiler_update_by_color(
    *,
    particles: dict[int, Particle],
    old_state: list[Particle],
    parent: Particle,
    child1: Particle,
    child2: Particle,
    selected: set[int],
) -> tuple[int | None, Particle | None]:
    """
    Find updated recoiler copy without scale matching.

    Candidates must be PYTHIA copy/update rows of an old active particle. Among
    those, choose the one with strongest color overlap with the branch system and
    smallest event-record distance to the branch daughters.
    """
    excluded = {parent.no, child1.no, child2.no} | set(selected)
    old_active = [p for p in old_state if p.no != parent.no]
    branch_colors = set(nz_colors(parent) + nz_colors(child1) + nz_colors(child2))
    child_anchor = min(child1.no, child2.no)

    scored: list[tuple[float, int, Particle]] = []

    for old in old_active:
        for cand in particles.values():
            if cand.no in excluded:
                continue
            if not is_copy_update(old, cand):
                continue

            shared_old = shared_color_count(old, cand)
            shared_branch = len(set(nz_colors(cand)) & branch_colors)
            no_penalty = 1e-3 * abs(cand.no - child_anchor)

            score = 0.0
            score += 100.0  # passed copy-update test
            score += 20.0 if cand.pid == old.pid else 0.0
            score += 5.0 * shared_old
            score += 2.0 * shared_branch
            score -= no_penalty

            scored.append((score, old.no, cand))

    if not scored:
        return None, None

    scored.sort(key=lambda x: (-x[0], x[2].no))
    best_score, best_old_no, best_cand = scored[0]

    # If the top two are essentially tied but different old recoilers, avoid
    # making an arbitrary update.
    if len(scored) >= 2:
        second_score, second_old_no, second_cand = scored[1]
        if abs(best_score - second_score) < 1e-9 and best_old_no != second_old_no:
            return None, None

    return best_old_no, best_cand


# -----------------------------------------------------------------------------
# State evolution / ordering
# -----------------------------------------------------------------------------

def branch_record_key(p: Particle, particles: dict[int, Particle]) -> int:
    d1, d2 = p.daughters
    return min(d1, d2)


def branch_record_key(p: Particle, particles: dict[int, Particle]) -> int:
    d1, d2 = p.daughters
    return min(d1, d2) if d1 > 0 and d2 > 0 else 10**12


def pick_next_branch_index_frontier(
    state: list[Particle],
    particles: dict[int, Particle],
    *,
    loose_color: bool = False,
) -> int | None:
    """
    Pick the next active particle to split without using scales.

    This uses PYTHIA event-record creation order of the daughter rows. It does
    not restrict to QCD splittings, so top/W resonance decays are represented
    in the frontier. QCD color checks are still used only for recoiler updates.
    """
    candidates: list[tuple[int, Particle]] = []
    for idx, p in enumerate(state):
        if is_real_split(p, particles):
            candidates.append((idx, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1].no, branch_record_key(x[1], particles)))
    return candidates[0][0]


def apply_branching_with_color_recoiler_update(
    *,
    state: list[Particle],
    idx: int,
    particles: dict[int, Particle],
    selected: set[int],
    applied_edges: list[tuple[int, int]],
    loose_color: bool = False,
) -> list[Particle]:
    """
    Replace parent by daughters.

    For QCD branchings, also try to update a recoiler copy using color/copy
    links. For resonance decays, do not look for a recoiler.
    """
    old_state = list(state)

    parent = state[idx]
    d1, d2 = parent.daughters
    child1 = particles[d1]
    child2 = particles[d2]

    if not is_real_split(parent, particles):
        raise RuntimeError(f"Tried to apply non-split branch {parent.no}->{d1},{d2}")

    old_recoiler_no, new_recoiler = None, None
    if is_qcd_branch(parent, particles, loose_color=loose_color):
        old_recoiler_no, new_recoiler = find_recoiler_update_by_color(
            particles=particles,
            old_state=old_state,
            parent=parent,
            child1=child1,
            child2=child2,
            selected=selected,
        )

    new_state = state[:idx] + [child1, child2] + state[idx + 1:]

    selected.add(d1)
    selected.add(d2)
    applied_edges.append((parent.no, d1))
    applied_edges.append((parent.no, d2))

    if old_recoiler_no is not None and new_recoiler is not None:
        for j, p in enumerate(new_state):
            if p.no == old_recoiler_no:
                new_state[j] = new_recoiler
                selected.add(new_recoiler.no)
                applied_edges.append((old_recoiler_no, new_recoiler.no))
                break

    return new_state


def tree_relative_order_for_current_particles(
    particles: dict[int, Particle],
    applied_edges: list[tuple[int, int]],
    selected: set[int],
    current_state: list[Particle],
    *,
    roots_nos: list[int],
) -> list[Particle]:
    selected = {int(n) for n in selected}
    current_by_no = {int(p.no): p for p in current_state}
    current_nos = set(current_by_no)

    children_by_parent: dict[int, list[int]] = {}
    parents_by_child: dict[int, list[int]] = {n: [] for n in selected}

    for a, b in applied_edges:
        a = int(a)
        b = int(b)
        if a in selected and b in selected:
            children_by_parent.setdefault(a, []).append(b)
            parents_by_child.setdefault(b, []).append(a)

    for a in list(children_by_parent):
        children_by_parent[a] = sorted(set(children_by_parent[a]), reverse=True)

    roots = [n for n in roots_nos if n in selected]
    if not roots:
        roots = [
            n for n in selected
            if not any(parent in selected for parent in parents_by_child.get(n, []))
        ]
    if not roots:
        roots = list(selected)

    ordered_current_nos: list[int] = []
    visited: set[int] = set()

    def dfs(n: int) -> None:
        if n in visited or n not in selected:
            return
        visited.add(n)
        if n in current_nos:
            ordered_current_nos.append(n)
        for child in children_by_parent.get(n, []):
            dfs(child)

    for root in sorted(set(roots), reverse=True):
        dfs(root)

    missing = sorted(current_nos - set(ordered_current_nos), reverse=True)
    ordered_current_nos.extend(missing)
    rank = {no: i for i, no in enumerate(ordered_current_nos)}

    return sorted(current_state, key=lambda p: (rank.get(int(p.no), len(rank)), int(p.no)))


def order_state(
    state: list[Particle],
    particles: dict[int, Particle],
    selected: set[int],
    applied_edges: list[tuple[int, int]],
    *,
    sort_state: str,
    roots_nos: list[int],
) -> list[Particle]:
    state_for_save = list(state)

    if sort_state == "energy":
        # p4 convention is [E, px, py, pz]
        state_for_save.sort(key=lambda p: float(p.p4[0].item()), reverse=True)
    elif sort_state == "no":
        state_for_save.sort(key=lambda p: p.no)
    elif sort_state == "tree-relative":
        state_for_save = tree_relative_order_for_current_particles(
            particles=particles,
            applied_edges=applied_edges,
            selected=selected,
            current_state=state_for_save,
            roots_nos=roots_nos,
        )
    elif sort_state == "frontier":
        pass
    else:
        raise ValueError(f"Unknown sort_state: {sort_state}")

    return state_for_save


def intermediate_states_for_event_color(
    particles: dict[int, Particle],
    *,
    sort_state: str = "tree-relative",
    save_nos: bool = False,
    max_branch: int | float = math.inf,
    loose_color: bool = False,
    event_type: str = "ee2ttbar",
) -> dict[int, torch.Tensor] | dict[int, dict[str, torch.Tensor]]:
    """
    Return nbranch -> p4 tensor after n event-record frontier branchings.

    Unlike the earlier color-root version, this starts from hard-process roots
    such as t,tbar and keeps resonance decays in the frontier. It does not use
    shower scale to decide the order.
    """
    roots_nos = choose_frontier_roots(particles, event_type=event_type)
    roots = [particles[n] for n in roots_nos if n in particles]

    if len(roots) == 0:
        return {}

    state = list(roots)
    selected = {p.no for p in roots}
    applied_edges: list[tuple[int, int]] = []

    out: dict[int, Any] = {}
    nbranch = 0

    while nbranch < max_branch:
        idx = pick_next_branch_index_frontier(state, particles, loose_color=loose_color)
        if idx is None:
            break

        state = apply_branching_with_color_recoiler_update(
            state=state,
            idx=idx,
            particles=particles,
            selected=selected,
            applied_edges=applied_edges,
            loose_color=loose_color,
        )
        nbranch += 1

        state_for_save = order_state(
            state,
            particles,
            selected,
            applied_edges,
            sort_state=sort_state,
            roots_nos=roots_nos,
        )

        P = torch.stack([p.p4 for p in state_for_save], dim=0).float()
        if save_nos:
            nos = torch.tensor([p.no for p in state_for_save], dtype=torch.long)
            out[nbranch] = {"p4": P, "no": nos}
        else:
            out[nbranch] = P

    return out


# -----------------------------------------------------------------------------
# Filtering/saving
# -----------------------------------------------------------------------------

def normalize_total_energy(P: torch.Tensor) -> torch.Tensor:
    # P is [E, px, py, pz]
    Etot = P[:, 0].sum().clamp_min(1e-12)
    return P / Etot


def state_has_bad_momentum(P: torch.Tensor, *, zero_momentum_tol: float) -> bool:
    if not torch.isfinite(P).all().item():
        return True
    p3 = P[:, 1:4]
    p3_norm = torch.linalg.norm(p3, dim=-1)
    return torch.any(p3_norm <= zero_momentum_tol).item()


def centered_normalized_p3_from_p4(out: torch.Tensor) -> torch.Tensor:
    """out: (N_events, N_particles, 4), p4=[E,px,py,pz]."""
    p_centered = out[:, :, 1:4] - out[:, :, 1:4].mean(dim=1, keepdim=True)
    E = torch.linalg.norm(p_centered, dim=-1)
    E_tot = E.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return p_centered / E_tot[:, None, :]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create NBranch=k tensors of intermediate PYTHIA QCD-shower states "
            "using color-line topology instead of scale ordering. Resonance decays "
            "are kept in the event record but not counted as QCD shower branchings."
        )
    )

    parser.add_argument("input", type=Path, help="Input .pt dict or PYTHIA .log/.txt file")
    parser.add_argument("-o", "--outdir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--input-format", choices=["auto", "pt-dict", "log"], default="auto")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-branch", type=int, default=math.inf)

    parser.add_argument(
        "--sort-state",
        choices=["tree-relative", "frontier", "no", "energy"],
        default="tree-relative",
    )
    parser.add_argument(
        "--loose-color",
        action="store_true",
        help="Use PID patterns to identify QCD branchings even if color tags fail strict compatibility checks.",
    )
    parser.add_argument("--event-type", choices=["ee2ttbar", "ee2qqbar", "auto"], default="ee2ttbar")
    parser.add_argument("--normalize-total-energy", action="store_true")
    parser.add_argument("--zero-momentum-tol", type=float, default=0.0)
    parser.add_argument("--keep-zero-momentum", action="store_true")
    parser.add_argument("--save-nos", action="store_true")
    parser.add_argument("--no-save-p3-normalized", action="store_true")
    
    parser.add_argument(
        "--target-nparticles",
        type=int,
        default=None,
        help=(
            "If set, save one fixed-size tensor containing this many particles per event. "
            "For each event, use the first intermediate state with at least this many particles, "
            "then select this many particles according to --select-particles."
        ),
    )

    parser.add_argument(
        "--select-particles",
        choices=["first", "energy"],
        default="first",
        help=(
            "How to choose particles when a state has more than --target-nparticles. "
            "'first' keeps the first particles after --sort-state ordering. "
            "'energy' keeps the highest-energy particles."
        ),
    )

    args = parser.parse_args()

    obj: Any | None = None
    input_format = args.input_format
    if input_format == "auto":
        if args.input.suffix.lower() in {".log", ".txt", ".out"}:
            input_format = "log"
        else:
            obj = load_torch_object(args.input)
            input_format = infer_input_format(args.input, obj)
    elif input_format == "pt-dict":
        obj = load_torch_object(args.input)

    if input_format == "log":
        event_particles = parse_pythia_log_with_scale(args.input, max_events=args.max_events)
        num_events = len(event_particles)
        source_desc = "PYTHIA log"
    elif input_format == "pt-dict":
        if obj is None:
            obj = load_torch_object(args.input)
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a dict from {args.input}, got {type(obj)}")
        data = obj
        required_keys = ["p4", "no", "pid", "status", "mothers", "daughters", "scale", "mask"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing key {key!r}. This script expects parse_pythia_log.py output.")
        # Also checks color key exists.
        _ = get_color_tensor_from_pt_dict(data)
        num_total = int(data["p4"].shape[0])
        num_events = min(num_total, args.max_events) if args.max_events is not None else num_total
        event_particles = [build_particles_for_event_from_pt(data, i) for i in range(num_events)]
        source_desc = "parse_pythia_log.py .pt dict"
    else:
        raise ValueError(f"Unknown input format: {input_format}")

    prefix = args.prefix if args.prefix is not None else args.input.stem
    args.outdir.mkdir(parents=True, exist_ok=True)

    groups: dict[int, list[torch.Tensor]] = {}
    groups_nos: dict[int, list[torch.Tensor]] = {}
    total_branchings: list[int] = []
    root_counts: list[int] = []
    used_nbranches: dict[int, list[torch.Tensor]] = {}

    for particles in tqdm(event_particles, desc="events"):
        roots = choose_frontier_roots(particles, event_type=args.event_type)
        root_counts.append(len(roots))

        states = intermediate_states_for_event_color(
            particles,
            sort_state=args.sort_state,
            save_nos=args.save_nos,
            max_branch=args.max_branch,
            loose_color=args.loose_color,
            event_type=args.event_type,
        )

        total_branchings.append(max(states.keys()) if states else 0)

        if args.target_nparticles is not None:
            target = args.target_nparticles

            chosen = None
            chosen_nbranch = None
            chosen_nos = None

            for nbranch in sorted(states):
                state_obj = states[nbranch]

                if args.save_nos:
                    assert isinstance(state_obj, dict)
                    P = state_obj["p4"]
                    nos = state_obj["no"]
                else:
                    assert torch.is_tensor(state_obj)
                    P = state_obj
                    nos = None

                if P.shape[0] < target:
                    continue

                # If there are more than target particles, choose exactly target.
                if args.select_particles == "energy":
                    idx = torch.argsort(P[:, 0], descending=True)[:target]  # P is [E,px,py,pz]
                    P = P[idx]
                    if nos is not None:
                        nos = nos[idx]
                elif args.select_particles == "first":
                    P = P[:target]
                    if nos is not None:
                        nos = nos[:target]
                else:
                    raise ValueError(f"Unknown select_particles: {args.select_particles}")

                if args.normalize_total_energy:
                    P = normalize_total_energy(P)

                if not args.keep_zero_momentum:
                    if state_has_bad_momentum(P, zero_momentum_tol=args.zero_momentum_tol):
                        continue

                chosen = P
                chosen_nbranch = nbranch
                chosen_nos = nos
                break

            if chosen is not None:
                groups.setdefault(target, []).append(chosen)

                # Save the actual branch count used for each event.
                used_nbranches.setdefault(target, []).append(
                    torch.tensor(chosen_nbranch, dtype=torch.long)
                )

                if args.save_nos and chosen_nos is not None:
                    groups_nos.setdefault(target, []).append(chosen_nos)

            continue


        # Original behavior: group every state by fixed nbranch.
        for nbranch, state_obj in states.items():
            if args.save_nos:
                assert isinstance(state_obj, dict)
                P = state_obj["p4"]
                nos = state_obj["no"]
            else:
                assert torch.is_tensor(state_obj)
                P = state_obj
                nos = None

            if args.normalize_total_energy:
                P = normalize_total_energy(P)

            groups.setdefault(nbranch, []).append(P)
            if args.save_nos and nos is not None:
                groups_nos.setdefault(nbranch, []).append(nos)

    if not groups:
        raise RuntimeError("No frontier branchings found. Check root/daughter logic.")

    removed_by_nbranch: dict[int, int] = {}
    if not args.keep_zero_momentum:
        for nbranch in list(groups.keys()):
            kept: list[torch.Tensor] = []
            kept_nos: list[torch.Tensor] = []
            removed = 0

            for idx, P in enumerate(groups[nbranch]):
                if state_has_bad_momentum(P, zero_momentum_tol=args.zero_momentum_tol):
                    removed += 1
                    continue
                kept.append(P)
                if args.save_nos:
                    kept_nos.append(groups_nos[nbranch][idx])

            removed_by_nbranch[nbranch] = removed

            if kept:
                groups[nbranch] = kept
                if args.save_nos:
                    groups_nos[nbranch] = kept_nos
            else:
                del groups[nbranch]
                if args.save_nos and nbranch in groups_nos:
                    del groups_nos[nbranch]

    if not groups:
        raise RuntimeError("All states were removed by zero/nonfinite momentum filtering.")

    print(f"Input: {args.input}")
    print(f"Input format: {input_format} ({source_desc})")
    print(f"Events processed: {num_events}")
    print(
        "Frontier branchings/event found: "
        f"min={min(total_branchings)}, "
        f"mean={sum(total_branchings) / len(total_branchings):.2f}, "
        f"max={max(total_branchings)}"
    )
    print(
        "Hard frontier roots/event found: "
        f"min={min(root_counts)}, "
        f"mean={sum(root_counts) / len(root_counts):.2f}, "
        f"max={max(root_counts)}"
    )
    print(f"sort_state: {args.sort_state}")
    print(f"loose_color: {args.loose_color}")

    if removed_by_nbranch:
        print("Removed zero/nonfinite-momentum states:")
        any_removed = False
        for nbranch in sorted(removed_by_nbranch):
            n_removed = removed_by_nbranch[nbranch]
            if n_removed > 0:
                any_removed = True
                print(f"  NBranch={nbranch}: removed {n_removed}")
        if not any_removed:
            print("  none")

    print("Writing files:")
    
    if args.target_nparticles is not None:
        target = args.target_nparticles

        if target not in groups or not groups[target]:
            raise RuntimeError(
                f"No events found with at least {target} particles."
            )

        out = torch.stack(groups[target], dim=0).contiguous()

        out_path = args.outdir / (
            f"{prefix}_Nparticles={target}_variableNBranch.pt"
        )
        torch.save(out, out_path)

        # Save p3 normalized version too
        p_centered = out[:, :, 1:4] - out[:, :, 1:4].mean(dim=1, keepdim=True)
        E = torch.linalg.norm(p_centered, dim=-1)
        E_tot = E.sum(dim=1, keepdim=True).clamp_min(1e-12)
        p_normalized = p_centered / E_tot[:, None, :]

        p3_path = args.outdir / (
            f"{prefix}_Nparticles={target}_variableNBranch_p3_normalized.pt"
        )
        torch.save(p_normalized, p3_path)

        # nbranch_used = torch.stack(used_nbranches[target], dim=0).contiguous()
        # nbranch_path = args.outdir / (
        #     f"{prefix}_Nparticles={target}_variableNBranch_usedNBranch.pt"
        # )
        # torch.save(nbranch_used, nbranch_path)

        print(f"Saved selected fixed-particle tensor: {tuple(out.shape)} -> {out_path}")
        print(f"Saved normalized p3 tensor: {tuple(p_normalized.shape)} -> {p3_path}")
        # print(f"Saved used NBranch values: {tuple(nbranch_used.shape)} -> {nbranch_path}")

        if args.save_nos and target in groups_nos:
            out_nos = torch.stack(groups_nos[target], dim=0).contiguous()
            nos_path = args.outdir / (
                f"{prefix}_Nparticles={target}_variableNBranch_nos.pt"
            )
            torch.save(out_nos, nos_path)
            print(f"Saved selected PYTHIA nos: {tuple(out_nos.shape)} -> {nos_path}")

        return

if __name__ == "__main__":
    main()
