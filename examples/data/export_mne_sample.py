#!/usr/bin/env python3
"""Export a small, real MNE sample recording to MATLAB integration fixtures.

Run explicitly from this directory's venv. The script never downloads on import.
Use --download only if the pinned source FIF is absent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import mne
import numpy as np
from scipy import __version__ as scipy_version
from scipy.io import savemat


HERE = Path(__file__).resolve().parent
SOURCE_NAME = "sample_audvis_trunc_raw.fif"
SOURCE_URL = (
    "https://raw.githubusercontent.com/mne-tools/mne-testing-data/"
    "33ef2a6f6345a1adf8764f8f63725f0ed9e4bb92/"
    "MEG/sample/sample_audvis_trunc_raw.fif"
)
EXPECTED_SHA256 = "d1ecdebf1ae23d9e7a5b56be63f45191778440cce6c8dfb0c270e75484b2aa50"
EVENT_LABELS = {
    1: "auditory/left",
    2: "auditory/right",
    3: "visual/left",
    4: "visual/right",
}
TMIN, TMAX = -0.1, 0.3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_source(download: bool) -> Path:
    path = HERE / SOURCE_NAME
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} absent; rerun with --download")
        with urlopen(SOURCE_URL, timeout=120) as response, path.open("wb") as out:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                out.write(chunk)
    digest = sha256(path)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Source SHA-256 mismatch: {digest}")
    return path


def positions_head_mm(info, picks: np.ndarray) -> np.ndarray:
    out = []
    for pick in picks:
        ch = info["chs"][int(pick)]
        loc = np.asarray(ch["loc"][:3], dtype=float)
        frame = int(ch["coord_frame"])
        if frame == int(mne.io.constants.FIFF.FIFFV_COORD_DEVICE):
            loc = mne.transforms.apply_trans(info["dev_head_t"], loc)
        elif frame != int(mne.io.constants.FIFF.FIFFV_COORD_HEAD):
            raise ValueError(f"Unexpected coordinate frame for {ch['ch_name']}: {frame}")
        out.append(loc * 1000)
    positions = np.asarray(out)
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError("Missing or invalid sensor position")
    return positions


def export() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--download", action="store_true")
    download = args.parse_args().download
    path = get_source(download)
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose="ERROR")
    selected = events[np.isin(events[:, 2], list(EVENT_LABELS))]
    observed = Counter(map(int, selected[:, 2]))
    if set(observed) != set(EVENT_LABELS) or min(observed.values()) < 2:
        raise ValueError(f"Insufficient repeated classes: {observed}")

    epochs = mne.Epochs(
        raw,
        selected,
        event_id={name: code for code, name in EVENT_LABELS.items()},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        detrend=None,
        proj=False,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        preload=True,
        verbose="ERROR",
    )
    if len(epochs) != len(selected):
        raise ValueError(f"Unexpected dropped epochs: {len(epochs)} / {len(selected)}")
    condition_ids = epochs.events[:, 2].astype(np.int32)
    event_samples = epochs.events[:, 0].astype(np.int32)
    seen: defaultdict[int, int] = defaultdict(int)
    fold_ids = np.empty(len(epochs), dtype=np.int32)
    for i, label in enumerate(condition_ids):
        # Alternate repeats within each original condition to put all classes
        # in two independent sets. This is for an integration test only.
        seen[int(label)] += 1
        fold_ids[i] = 1 if seen[int(label)] % 2 else 2
    for label in EVENT_LABELS:
        if set(fold_ids[condition_ids == label]) != {1, 2}:
            raise ValueError(f"Class {label} absent from a fold")

    manifest = {
        "source_url": SOURCE_URL,
        "source_repository": "https://github.com/mne-tools/mne-testing-data",
        "source_tag": "0.176",
        "source_commit": "33ef2a6f6345a1adf8764f8f63725f0ed9e4bb92",
        "source_file": SOURCE_NAME,
        "source_bytes": path.stat().st_size,
        "source_sha256": sha256(path),
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "event_channel": "STI 014",
        "event_labels": EVENT_LABELS,
        "event_counts": {str(k): v for k, v in sorted(observed.items())},
        "epoch_tmin_s": TMIN,
        "epoch_tmax_s": TMAX,
        "epoch_baseline": None,
        "epoch_detrend": None,
        "epoch_proj": False,
        "epoch_reject": None,
        "raw_highpass_hz": raw.info["highpass"],
        "raw_lowpass_hz": raw.info["lowpass"],
        "raw_sfreq_hz": raw.info["sfreq"],
        "raw_bads_excluded": raw.info["bads"],
        "outputs": {},
    }
    all_data = epochs.get_data(copy=False)
    for kind in ("eeg", "mag", "grad"):
        picks = mne.pick_types(
            epochs.info,
            meg=kind if kind != "eeg" else False,
            eeg=kind == "eeg",
            exclude="bads",
        )
        data = np.ascontiguousarray(all_data[:, picks, :], dtype=np.float64)
        labels = [epochs.info["chs"][int(p)]["ch_name"] for p in picks]
        types = [kind] * len(picks)
        positions = positions_head_mm(epochs.info, picks)
        if data.shape[0] != len(condition_ids) or not np.isfinite(data).all():
            raise ValueError(f"Bad output data for {kind}")
        output = HERE / f"sample_audvis_{kind}_epochs.mat"
        savemat(
            output,
            {
                "data": data,  # observation x feature x time; original SI units
                "condition_ids": condition_ids[:, None],
                "fold_ids": fold_ids[:, None],
                "sample_times_s": epochs.times[None, :],
                "feature_ids": np.asarray(labels, dtype=object)[:, None],
                "feature_types": np.asarray(types, dtype=object)[:, None],
                "positions_mm": positions,  # all in head coordinates
                "event_samples": event_samples[:, None],
                "event_labels": np.asarray(
                    [EVENT_LABELS[int(x)] for x in condition_ids], dtype=object
                )[:, None],
            },
            do_compression=True,
            oned_as="column",
        )
        manifest["outputs"][kind] = {
            "file": output.name,
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
            "shape_observation_feature_time": list(data.shape),
            "si_unit": {"eeg": "V", "mag": "T", "grad": "T/m"}[kind],
            "position_frame": "head",
            "position_unit": "mm",
            "fold_by_condition": {
                str(c): {str(f): int(np.sum((condition_ids == c) & (fold_ids == f))) for f in (1, 2)}
                for c in EVENT_LABELS
            },
        }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    export()
