# Optional real EEG and MEG example

The main synthetic example is self-contained. The optional example uses a short actual recording from [MNE testing-data tag 0.176](https://github.com/mne-tools/mne-testing-data/tree/0.176/MEG/sample), pinned to commit `33ef2a6f6345a1adf8764f8f63725f0ed9e4bb92`. Source SHA-256: `d1ecdebf1ae23d9e7a5b56be63f45191778440cce6c8dfb0c270e75484b2aa50`; size 12,036,840 bytes.

The public recording is **not bundled**. Download only when you explicitly run this command in a terminal:

```sh
python3 -m venv .venv-example
.venv-example/bin/python -m pip install mne==1.8.0 numpy==1.26.4 'scipy<1.14'
.venv-example/bin/python examples/data/export_mne_sample.py --download
```

Then in MATLAB:

```matlab
setup_rsa_meeg
summary = demo_public_sample(fullfile(pwd,'examples','data'));
```

The exporter checks the source SHA before reading and writes separate MAT inputs for EEG, magnetometers and gradiometers, plus a manifest with versions, time axis, processing settings and output checksums. The MNE [event tutorial](https://mne.tools/stable/auto_tutorials/intro/20_events_from_raw.html) identifies event codes 1/2 as left/right auditory and 3/4 as left/right visual. This excerpt yields 6/6/6/5 events, 23 epochs and 121 samples per epoch; bad channels are excluded, leaving 59 EEG, 102 mag and 203 grad channels. MEG sensor positions are transformed from device to head coordinates and all positions are converted to mm.

No additional filtering, baseline subtraction, detrending, projection, rejection or rereferencing is applied. Epoch bounds are nominally -.1 to .3 s; actual sample times are retained at approximately 300.3075 Hz. Alternating repetitions within each condition assign fold labels for software checks; this is not evidence that neighboring epochs are statistically independent in a scientific analysis. The MATLAB example uses correlation-distance searchlights and a modality RDM, with no population-level p-values for this single person.

The original [MNE sample dataset on OpenNeuro](https://openneuro.org/datasets/ds000248/versions/1.2.4) lists CC0; a separate licence declaration for this exact truncated testing-data copy was not found. We provide an explicit retrieval/export script and provenance rather than redistribute the recording. Consult the source dataset's terms before redistribution. This short excerpt validates I/O/execution only, not study-level power, source reconstruction, scientific superiority or historical paper replication.
