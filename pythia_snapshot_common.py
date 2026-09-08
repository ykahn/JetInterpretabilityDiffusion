"""Shared machinery for the parton-snapshot generation scripts.

This module factors out the flavor-agnostic pieces of
``generate_pythia_zqq_splittings.py`` so the several systematic-scan
generators can reuse them without copy-paste drift:

    * the pythia8 import guard,
    * the per-particle event-record schema (``RECORD_FIELDS``),
    * a base argument parser (``build_base_parser``) and seed resolution,
    * the generic final-state-radiation snapshot extractor
      (``extract_fsr_snapshots``), generalized to accept an arbitrary pair of
      seed partons (e.g. the qqbar from a Z, or the gg from a Higgs),
    * the event loop / stacking / per-snapshot 4-momentum conservation check /
      event-record dump / metadata writer (``run_generation``).

Each generator supplies its own ``configure_pythia`` (process + shower
settings) and an ``extract_snapshots(event) -> List[(k, ndarray)]`` callback,
where ``k`` is the snapshot index such that the array has shape ``(2 + k, 4)``
holding the (E, px, py, pz) of the ``2 + k`` active partons.  The output layout
(``splittings_k.npy``, ``event_ids_k.npy``, ``event_record.npz``,
``metadata.json``) is byte-compatible with the original script, so
``verify_pythia_zqq_splittings.py`` works unchanged on every dataset.
"""

from __future__ import annotations

import argparse
import json
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

# PYTHIA accepts seeds in [1, 900_000_000] when Random:setSeed = on.
PYTHIA_SEED_MAX = 900_000_000

try:
    import pythia8
except ImportError as exc:  # pragma: no cover - exercised only without pythia8
    raise SystemExit(
        "pythia8 Python bindings are not importable. On this machine they "
        "live in the `madgraph` conda env; run with "
        "/opt/anaconda3/envs/madgraph/bin/python (or `conda activate madgraph` first)."
    ) from exc


# Per-particle attributes pulled from the PYTHIA event record. Keep this list
# in one place so the buffer init, the per-event fill, and the savez payload
# all stay in sync. Identical to the original script's table.
RECORD_FIELDS: Tuple[Tuple[str, str, type], ...] = (
    # (column name,        Particle accessor,  numpy dtype)
    ("status",             "status",           np.int32),
    ("pdg_id",             "id",               np.int32),
    ("mother1",            "mother1",          np.int32),
    ("mother2",            "mother2",          np.int32),
    ("daughter1",          "daughter1",        np.int32),
    ("daughter2",          "daughter2",        np.int32),
    ("col",                "col",              np.int32),
    ("acol",               "acol",             np.int32),
    ("e",                  "e",                np.float32),
    ("px",                 "px",               np.float32),
    ("py",                 "py",               np.float32),
    ("pz",                 "pz",               np.float32),
    ("m",                  "m",                np.float32),
)


# Settings shared by every e+e- generator here: point-like leptons, ISR/MPI
# off, FSR on, QED showers off, parton level only, quiet banner.
COMMON_EE_SETTINGS: Tuple[str, ...] = (
    "PDF:lepton = off",
    "PartonLevel:ISR = off",
    "PartonLevel:MPI = off",
    "PartonLevel:FSR = on",
    "TimeShower:QEDshowerByQ = off",
    "TimeShower:QEDshowerByL = off",
    "TimeShower:QEDshowerByGamma = off",
    "Init:showProcesses = off",
    "Init:showMultipartonInteractions = off",
    "Init:showChangedSettings = off",
    "Init:showChangedParticleData = off",
    "Next:numberShowEvent = 0",
    "Next:numberShowInfo = 0",
    "Next:numberShowProcess = 0",
    "Next:numberCount = 0",
)


def build_base_parser(description: str, default_output_dir: str,
                      default_ecm: float = 91.188) -> argparse.ArgumentParser:
    """ArgumentParser carrying the options common to every generator.

    Individual scripts add their own knobs (``--alphas-mz``, ``--flavor``, ...).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-n", "--n-events", type=int, default=100_000,
                        help="Number of events to generate.")
    parser.add_argument("--ecm", type=float, default=default_ecm,
                        help="Center-of-mass energy in GeV.")
    parser.add_argument("--seed", type=int, default=42,
                        help="PYTHIA random seed (1 to 900_000_000). "
                             "Pass -1 to draw a fresh seed; the value used is "
                             "recorded in metadata.json for reproducibility.")
    parser.add_argument("--output-dir", type=Path, default=Path(default_output_dir))
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Per-component conservation tolerance (GeV).")
    parser.add_argument("--no-event-record", dest="save_event_record",
                        action="store_false", default=True,
                        help="Skip saving the full per-particle event record "
                             "(event_record.npz). Saves time and disk if you "
                             "only need the per-branching snapshots.")
    parser.add_argument("--all-events-only", dest="save_per_branching",
                        action="store_false", default=True,
                        help="Write only the combined final-state files "
                             "(splittings_all / event_ids_all / n_partons_all); "
                             "skip the per-branching splittings_k / event_ids_k "
                             "files (saves disk and memory).")
    return parser


def resolve_seed(seed: int) -> int:
    """Return a valid PYTHIA seed, drawing a random one when ``seed == -1``."""
    if seed == -1:
        seed = secrets.randbelow(PYTHIA_SEED_MAX) + 1
        print(f"Drew random seed: {seed}")
    elif not 1 <= seed <= PYTHIA_SEED_MAX:
        raise SystemExit(
            f"--seed must be -1 (random) or in [1, {PYTHIA_SEED_MAX}]; got {seed}"
        )
    return seed


def find_hard_outgoing(event, mother_pdg: int) -> List[int]:
    """Event-record indices of the outgoing partons emitted directly by a hard
    resonance of PDG id ``mother_pdg`` (status |code| == 23).

    Generalizes the original ``find_initial_zqq``: ``mother_pdg = 23`` returns
    the q qbar from a Z; ``mother_pdg = 25`` returns the g g (or t tbar) from a
    Higgs.  Returns the indices in record order.
    """
    out: List[int] = []
    for i in range(1, event.size()):
        p = event[i]
        if p.statusAbs() != 23:
            continue
        m1 = p.mother1()
        if m1 > 0 and event[m1].id() == mother_pdg:
            out.append(i)
    return out


def extract_fsr_snapshots(event, initial: Sequence[int], include_k0: bool = False
                          ) -> List[Tuple[int, np.ndarray]]:
    """Parton-level 4-momentum snapshots after each FSR branching.

    Generalization of the original ``extract_snapshots``: instead of hard-coding
    the Z -> q qbar finder, the two (or more) seed partons are passed in via
    ``initial`` (event-record indices of the colored partons emitted by the hard
    resonance).  The returned list contains ``(k, snap)`` pairs where ``snap``
    has shape ``(len(initial) + k, 4)`` and holds the (E, px, py, pz) of the
    active partons immediately after the k-th branching, k = 1, 2, ....

    ``include_k0=True`` prepends the k=0 snapshot of the bare seed partons
    (shape ``(len(initial), 4)``), i.e. the hard partons before any FSR.  This is
    meaningful only when the seed multiplicity makes the EFPs nontrivial: for the
    4-parton H -> bbbb final state the bare-parton EFPs carry real information,
    whereas for a 2-parton (back-to-back q qbar / g g) seed they are delta
    functions -- so it defaults to off and is enabled for the >=3-seed samples.

    Algorithm (unchanged from the original): walk the event record forward from
    just past the seed partons.  A "branching" is the run of consecutive entries
    whose ``mother1()`` is in the current active set, terminated by the first
    entry with ``|status| == 52`` (the recoiler) in PYTHIA's pT-ordered dipole
    shower.  Each FSR branching adds exactly one parton overall; if that
    invariant is ever violated the walk stops and the snapshots gathered so far
    are returned.
    """
    n_seed = len(initial)
    if n_seed < 2:
        return []

    active: Set[int] = set(initial)
    snapshots: List[Tuple[int, np.ndarray]] = []
    i = max(initial) + 1
    size = event.size()
    while i < size:
        if event[i].mother1() not in active:
            i += 1
            continue
        j = i
        new_idx: List[int] = []
        mothers_replaced: Set[int] = set()
        while j < size and event[j].mother1() in active:
            new_idx.append(j)
            mothers_replaced.add(event[j].mother1())
            status_added = event[j].status()
            j += 1
            if abs(status_added) == 52:
                break
        active -= mothers_replaced
        active |= set(new_idx)
        # Each FSR branching should add exactly one parton overall.
        if len(active) != n_seed + len(snapshots) + 1:
            break
        snap = np.array(
            [[event[k].e(), event[k].px(), event[k].py(), event[k].pz()]
             for k in sorted(active)],
            dtype=np.float64,
        )
        snapshots.append((len(snapshots) + 1, snap))
        i = j
    if include_k0:
        snap0 = np.array(
            [[event[k].e(), event[k].px(), event[k].py(), event[k].pz()]
             for k in sorted(initial)],
            dtype=np.float64,
        )
        snapshots.insert(0, (0, snap0))
    return snapshots


def extract_tree_snapshots(event, initial: Sequence[int]
                           ) -> List[Tuple[int, np.ndarray]]:
    """Snapshots after each branching of a full decay+shower tree.

    Generalizes ``extract_fsr_snapshots`` to event records that interleave
    resonance decays (t -> b W, W -> q qbar, H -> X) with FSR emissions, as in
    the H -> t tbar sample.  Each branching adds exactly one active object; a
    snapshot of the active partons' (E, px, py, pz) is recorded after every such
    branching, returned as ``(k, snap)`` with ``snap`` of shape
    ``(len(initial) + k, 4)``.

    A single branching is collected as the daughters of ONE mother (the decaying
    resonance or the FSR emitter).  For an FSR emission - identified by status
    |51| daughters - the trailing status |52| recoiler copy of the spectator is
    attached as well (it absorbs the recoil, so the post-branching snapshot
    conserves momentum).  Resonance decays (status |22|, |23|, |91|, ... two-body
    daughters, no recoiler) are kept to their single mother, which is what stops
    two adjacent decays (e.g. t and tbar) from being merged.  Pure 1->1 copies
    (delta == 0) are applied as relabelings without emitting a snapshot.

    The walk stops if a branching ever changes the object count by more than one
    (an unexpected record structure); the snapshots gathered so far are
    returned.  Energy-momentum conservation of every emitted snapshot is the
    downstream correctness check.
    """
    n_seed = len(initial)
    if n_seed < 2:
        return []

    active: Set[int] = set(initial)
    snapshots: List[Tuple[int, np.ndarray]] = []
    size = event.size()
    i = max(initial) + 1
    while i < size:
        m0 = event[i].mother1()
        if m0 not in active:
            i += 1
            continue
        st_first = event[i].status()
        new_idx: List[int] = []
        j = i
        while j < size and event[j].mother1() == m0:
            new_idx.append(j)
            j += 1
        mothers_replaced: Set[int] = {m0}
        # FSR emission carries a status-52 recoiler copy of the spectator.
        if (abs(st_first) == 51 and j < size
                and abs(event[j].status()) == 52
                and event[j].mother1() in active
                and event[j].mother1() != m0):
            new_idx.append(j)
            mothers_replaced.add(event[j].mother1())
            j += 1

        delta = len(new_idx) - len(mothers_replaced)
        active = (active - mothers_replaced) | set(new_idx)
        i = j
        if delta == 0:
            # pure relabeling / recoiler copy: no new object, no snapshot.
            continue
        if delta != 1 or len(active) != n_seed + len(snapshots) + 1:
            return snapshots
        snap = np.array(
            [[event[k].e(), event[k].px(), event[k].py(), event[k].pz()]
             for k in sorted(active)],
            dtype=np.float64,
        )
        snapshots.append((len(snapshots) + 1, snap))
    return snapshots


def run_generation(
    pythia,
    extract_snapshots: Callable[[object], List[Tuple[int, np.ndarray]]],
    *,
    n_events: int,
    ecm: float,
    seed: int,
    tolerance: float,
    output_dir: Path,
    save_event_record: bool,
    extra_metadata: Dict | None = None,
    n_seed: int = 2,
    save_all_events: bool = True,
    save_per_branching: bool = True,
) -> None:
    """Generate events, stack per-branching snapshots, and write the dataset.

    ``extract_snapshots`` maps a PYTHIA event to a list of ``(k, snap)`` pairs
    with ``snap`` of shape ``(n_seed + k, 4)``, where ``n_seed`` is the number of
    hard partons seeding the shower (2 for q qbar / g g, 4 for the H -> bbbb
    samples).  The system is produced at rest, so every snapshot's total
    4-momentum must equal ``(ecm, 0, 0, 0)``; this is checked per file and the
    process exits nonzero on any violation.

    In addition to the per-branching ``splittings_k.npy`` files, when
    ``save_all_events`` is set a single combined file ``splittings_all.npy`` is
    written holding EVERY event's FINAL (fully-showered) parton state, one row
    per event, zero-padded along the parton axis to the global maximum
    multiplicity ``n_seed + k_max``.  Shape ``(n_events, n_seed + k_max, 4)``;
    events with fewer partons have trailing all-zero rows.  Zero-momentum padding
    is inert for the repo's own guarded ``compute_special_torch`` / thrust
    (energy fraction z = 0); note it is NOT inert for a generic EFP
    implementation such as the EnergyFlow package, which derives per-particle
    angles and divides by zero on padding -- slice with ``n_partons_all`` first
    there.  ``event_ids_all.npy`` and the per-event real parton count
    ``n_partons_all.npy`` accompany it.

    ``save_per_branching=False`` writes ONLY the combined all-events files and
    skips the per-branching ``splittings_k.npy`` (buffers are not accumulated, so
    it also saves memory).  Conservation is then validated on the final-state
    array instead of per step.
    """
    if not (save_per_branching or save_all_events):
        raise SystemExit("nothing to save: enable per-branching or all-events output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # buffers[k] holds (event_id, snapshot_array) pairs for shower step k.
    buffers: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    n_generated = 0
    n_skipped = 0
    report_every = max(1, n_events // 20)

    record_cols: Dict[str, list] = (
        {name: [] for name, _, _ in RECORD_FIELDS} if save_event_record else {}
    )
    event_id_buf: List[int] = []
    event_offsets: List[int] = [0]
    # (event_id, final-state snapshot) for the combined all-events file.
    final_snaps: List[Tuple[int, np.ndarray]] = []

    for iev in range(n_events):
        if not pythia.next():
            n_skipped += 1
            continue
        n_generated += 1
        event = pythia.event
        if save_event_record:
            for idx in range(1, event.size()):
                p = event[idx]
                for name, accessor, _ in RECORD_FIELDS:
                    record_cols[name].append(getattr(p, accessor)())
                event_id_buf.append(iev)
            event_offsets.append(len(event_id_buf))
        snaps = extract_snapshots(event)
        if save_per_branching:
            for k, snap in snaps:
                buffers[k].append((iev, snap))
        if save_all_events and snaps:
            # the largest-k snapshot is this event's final showered state
            _, final_snap = max(snaps, key=lambda ks: ks[0])
            final_snaps.append((iev, final_snap))
        if (iev + 1) % report_every == 0:
            print(f"  ... event {iev + 1} / {n_events}")

    pythia.stat()
    if buffers:
        max_step = max(buffers)
    elif final_snaps:
        max_step = max(s.shape[0] for _, s in final_snaps) - n_seed
    else:
        max_step = 0
    print(f"Generated {n_generated} events ({n_skipped} skipped); "
          f"up to {max_step} branchings observed.")

    # The hard system is produced at rest, so the total 4-momentum is
    # (E_cm, 0, 0, 0) and each snapshot must conserve it.
    expected = np.array([ecm, 0.0, 0.0, 0.0], dtype=np.float64)

    metadata: Dict = {
        "n_events_requested": n_events,
        "n_events_generated": n_generated,
        "n_events_skipped": n_skipped,
        "ecm_GeV": ecm,
        "seed": seed,
        "tolerance_GeV": tolerance,
        "n_seed": n_seed,
        "expected_total_4momentum": expected.tolist(),
        "snapshot_files": {},
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    bad_steps: List[Tuple[int, float, int]] = []

    for step in sorted(buffers):
        ids = np.fromiter((eid for eid, _ in buffers[step]),
                          dtype=np.int64, count=len(buffers[step]))
        arr = np.stack([snap for _, snap in buffers[step]], axis=0)
        assert arr.shape[1] == n_seed + step, (
            f"internal error: expected {n_seed + step} partons at step {step}, "
            f"got {arr.shape[1]}"
        )

        total = arr.sum(axis=1)
        dev = np.abs(total - expected)
        max_dev = float(dev.max())
        n_bad = int(np.any(dev > tolerance, axis=1).sum())
        flag = "OK" if n_bad == 0 else "FAIL"
        print(f"  step {step:2d}: N = {arr.shape[0]:7d}  shape = {tuple(arr.shape)}  "
              f"max |sum p - p_tot| = {max_dev:.3e} GeV  "
              f"({n_bad} rows outside tol) [{flag}]")
        if n_bad:
            bad_steps.append((step, max_dev, n_bad))

        snap_file = output_dir / f"splittings_{step}.npy"
        ids_file = output_dir / f"event_ids_{step}.npy"
        np.save(snap_file, arr.astype(np.float32))
        np.save(ids_file, ids)
        metadata["snapshot_files"][step] = {
            "snapshot_file": snap_file.name,
            "event_ids_file": ids_file.name,
            "shape": list(arr.shape),
            "max_conservation_deviation_GeV": max_dev,
            "rows_outside_tolerance": n_bad,
        }

    n_bad_all = 0
    all_dev = 0.0
    if save_all_events and final_snaps:
        # one row per event = its final (fully-showered) parton state, zero-padded
        # along the parton axis to the global maximum multiplicity.
        max_p = max(s.shape[0] for _, s in final_snaps)
        n_ev = len(final_snaps)
        all_arr = np.zeros((n_ev, max_p, 4), dtype=np.float32)
        all_ids = np.empty(n_ev, dtype=np.int64)
        all_np = np.empty(n_ev, dtype=np.int32)
        for row, (eid, s) in enumerate(final_snaps):
            all_arr[row, :s.shape[0]] = s.astype(np.float32)
            all_ids[row] = eid
            all_np[row] = s.shape[0]
            # conservation on the float64 snapshot (matches the per-step check;
            # the saved float32 array rounds at ~1e-6 GeV, above tolerance).
            d = np.abs(s.sum(axis=0) - expected)
            all_dev = max(all_dev, float(d.max()))
            if bool(np.any(d > tolerance)):
                n_bad_all += 1
        flag = "OK" if n_bad_all == 0 else "FAIL"
        np.save(output_dir / "splittings_all.npy", all_arr)
        np.save(output_dir / "event_ids_all.npy", all_ids)
        np.save(output_dir / "n_partons_all.npy", all_np)
        print(f"  all-events: N = {n_ev:7d}  shape = {tuple(all_arr.shape)}  "
              f"(padded to {max_p} partons; real counts in n_partons_all.npy)  "
              f"max |sum p - p_tot| = {all_dev:.3e} GeV  "
              f"({n_bad_all} rows outside tol) [{flag}]")
        metadata["all_events_file"] = {
            "snapshot_file": "splittings_all.npy",
            "event_ids_file": "event_ids_all.npy",
            "n_partons_file": "n_partons_all.npy",
            "shape": list(all_arr.shape),
            "max_partons": int(max_p),
            "max_conservation_deviation_GeV": all_dev,
            "rows_outside_tolerance": n_bad_all,
            "note": ("final (fully-showered) parton state per event, zero-padded "
                     "to n_seed + k_max partons; trailing all-zero rows are "
                     "padding. n_partons_all[i] gives the real parton count for "
                     "event event_ids_all[i]."),
        }
    metadata["per_branching_saved"] = save_per_branching

    if save_event_record:
        event_id_arr = np.asarray(event_id_buf, dtype=np.int64)
        offsets_arr = np.asarray(event_offsets, dtype=np.int64)
        record_arrays = {
            name: np.asarray(record_cols[name], dtype=dtype)
            for name, _, dtype in RECORD_FIELDS
        }
        record_arrays["event_id"] = event_id_arr
        record_arrays["event_offsets"] = offsets_arr

        record_path = output_dir / "event_record.npz"
        np.savez(record_path, **record_arrays)
        n_particles = int(event_id_arr.shape[0])
        n_events_in_record = int(offsets_arr.shape[0] - 1)
        print(f"  event record: {n_events_in_record} events, "
              f"{n_particles} particle rows -> {record_path.name}")
        metadata["event_record"] = {
            "file": record_path.name,
            "n_events": n_events_in_record,
            "n_particles": n_particles,
            "columns": sorted(record_arrays.keys()),
            "note": ("event i occupies rows "
                     "event_offsets[i] : event_offsets[i+1] in every column "
                     "(particle indices within an event are implicit in row "
                     "order, starting at PYTHIA slot 1)."),
        }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    problems = []
    if bad_steps:
        problems.append(", ".join(
            f"step {s} ({n} rows, max dev {d:.3e} GeV)" for s, d, n in bad_steps))
    if n_bad_all:
        problems.append(f"all-events ({n_bad_all} rows, max dev {all_dev:.3e} GeV)")
    if problems:
        raise SystemExit("Energy-momentum conservation check FAILED: "
                         + "; ".join(problems))
    print("All snapshots conserve 4-momentum within tolerance.")
