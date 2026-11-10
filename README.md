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

## UDP proxy

`udp-proxy` (package `udp_proxy`, independent of `roc_cmp`) is a localhost UDP relay with a per-stream
network simulator: packet loss, delay and jitter (without reordering).

```sh
uv run udp-proxy proxy.yaml
```

```yaml
bind: 127.0.0.1                  # optional, default 127.0.0.1
seed: 1                          # optional, makes loss/jitter reproducible (same drops every run)
shared_source_port: true         # optional, default true: all streams leave from one UDP port
streams:
  - ports: 10001:127.0.0.1:10002 # local_port:remote_ip:remote_port (ssh notation)
    loss: 5                      # percent of dropped packets, 0..100 (default 0)
    delay_ms: 50                 # constant delay (default 0)
    jitter_ms: 10                # delay varies uniformly by +-10 ms, never reorders (default 0)
  - ports: 10003:127.0.0.1:10004 # streams are independent, each has its own parameters
```

Packets received on `local_port` are sent to `remote_ip:remote_port`; packets coming back from the
remote (e.g. RTCP feedback) are returned to the last client that wrote to `local_port`. Both
directions are simulated independently with the stream parameters. Stop with Ctrl-C / SIGTERM.

## Python versions

Python **3.5 and newer**. The floor comes from what the code really needs: `async`/`await`
(`udp-proxy`), the `typing` module, the `@` matrix operator and `math.gcd` (`roc-cmp`). Newer
syntax and APIs (f-strings, dataclasses, `X | None`, `asyncio.run`, ...) are deliberately not used.
Python 3.4 (Ubuntu 14.04) is not supported.

`udp-proxy` needs only the standard library and PyYAML (>= 3.13); `roc-cmp` also needs numpy (>= 1.15)
and scipy (>= 1.1). The whole test suite is run on CPython 3.5.5 (numpy 1.15.2, scipy 1.1.0),
3.6.15 (1.19.5, 1.5.3), 3.7.12 and 3.8 to 3.14 (newest installable numpy/scipy for each).

Default `python3` in the roc-toolkit CI images (`rocstreaming/env-*`):

| image | python3 | |
|---|---|---|
| ubuntu 14.04 | 3.4 | not supported |
| ubuntu 16.04 | 3.5 | ok, the minimum |
| ubuntu 18.04 | 3.6 | ok |
| ubuntu 20.04 / 22.04 / 24.04 / 26.04 | 3.8 / 3.10 / 3.12 / 3.14 | ok |
| debian, alpine, fedora, arch | 3.13, 3.12, 3.14, 3.14 | ok |
| opensuse leap | 3.6 (15.x), 3.13 (16.x) | ok |

These images install only the distro's `python` (Python 2 on Ubuntu <= 20.04), so `python3`, pip
and the dependencies have to be added to the image for the tools to run there.

Installing on an old interpreter: `pip install .` from the source tree needs Python >= 3.8 because of
the build backend. On older ones install the wheel built elsewhere (`uv build --wheel`, it is pure
Python) or run from the source tree:

```sh
PYTHONPATH=src python3 -m udp_proxy proxy.yaml
PYTHONPATH=src python3 -m roc_cmp REF DUMP
```

roc-recv dumps are 24-bit WAV, which scipy reads only since 1.6 (Python 3.7); `roc-cmp` has its own
reader for them, so older scipy works too.

## Test data

`data/sonata_no_14_moonlight.mp3`: Beethoven, Piano Sonata No. 14 in C♯ minor "Moonlight",
Op. 27 No. 2. Downloaded on 10/10/2026 from
[Musopen](https://musopen.org/music/2547-piano-sonata-no-14-in-c-sharp-minor-moonlight-sonata-op-27-no-2/).
