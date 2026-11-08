# roc-cmp
Fuzzy audio file comparison that tolerates resampling artifacts and leading/trailing silence or gaps.

CLI tool for CI of [roc-toolkit](https://github.com/roc-streaming/roc-toolkit): scores how much a
music file was damaged while being streamed, by comparing the reference WAV with the WAV dumped by
`roc-recv`. Only `numpy` and `scipy` are used.

```sh
uv run roc-cmp REF DUMP --events                    # also list every glitch
uv run roc-cmp REF DUMP --json                      # machine-readable
uv run roc-cmp REF DUMP --min-score 95              # exit status 1 if score < 95
```

## Score

`0..100`: percentage of the received audio that is clean. **100** = nothing wrong,
**0** = nothing resembling the reference was received (silent dump, noise, unrelated audio),
or everything received is distorted. Exit status is 2 on unreadable input.

Not penalised: silence before the stream starts, silence after it ends (partial send, receiver
stopped), different sample rate / bit depth / channel count (mono dump), and the few ms of clock
drift roc compensates for. `reference coverage` in the report says how much of the reference was
actually received, but does not affect the score.

Penalised: dropouts (digital-silence gaps >= 1.5 ms where the reference is audible, which is how
roc-recv renders unrecovered packet loss), skipped/repeated audio, noise, clipping, filtering,
dead or swapped channels, wrong speed, level changes. A glitch counts as ruining its own duration
plus 0.1 s on both sides (`--margin`), so the score reflects the number of audible disturbances,
not only the total length of the holes.

## How it works

1. Both files are converted to float and resampled to 16 kHz.
2. The dump is located in the reference: a few loud probe windows are cross-correlated against the
   whole reference for the global offset, then the offset is tracked per 0.5 s block (with a wide
   re-search if tracking is lost). This follows drift and exposes timeline jumps.
3. Each 32 ms dump frame is compared with the reference frame at the tracked position by
   log-spectral distance over 24 log-spaced bands, per channel (worst channel wins). Frames above
   `--threshold` are glitches.
4. Short digital-silence runs where the reference is audible are added as dropouts: a few ms of
   silence barely moves a 32 ms spectrum but is a clearly audible click.
