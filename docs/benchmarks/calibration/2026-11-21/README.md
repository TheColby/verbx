# Hosted CI reference calibration, November 21, 2026

The short algorithmic reference is now **0.9319392919999956 seconds**, the median
of three measured, warmed hosted CI renders below. The former 0.30-second value
came from baseline version `0.7.1`, described as alpha-era and lacking original
runner metadata. It is not a current hosted-run reference. Only this reference
changes. Its absolute maximum remains **2.50 seconds**; other references, maximum
budgets, the 3× comparison multiplier, and measurement policy are unchanged.

| Run | Branch | Short first / warmed seconds | Long warmed seconds | Matrix warmed seconds | 192k warmed seconds |
|---|---|---:|---:|---:|---:|
| [44923643042](https://github.com/TheColby/verbx/actions/runs/44923643042) | main | 4.720089 / 0.573312 | 1.818808 | 0.006198 | 0.382747 |
| [44924437373](https://github.com/TheColby/verbx/actions/runs/44924437373) | codex/0.9.11-development | 6.143326 / 0.931939 | 2.964511 | 0.008280 | 0.645156 |
| [44924526507](https://github.com/TheColby/verbx/actions/runs/44924526507) | main | 6.187366 / 0.949787 | 3.000790 | 0.008328 | 0.661719 |

These jobs use Python 3.11 on `ubuntu-latest`; setup logs identify Ubuntu 26.04.1,
image `20261116.202.1`. The archived JSON files are exact downloaded
`benchmark-report` artifacts, retaining first-pass results, output shapes,
comparison metadata, and original failed/passed conclusions. The adjacent
[provenance.json](provenance.json) records run links, exact source SHAs, artifact
paths, and SHA-256 checksums. The reports are evidence, not regenerated results
against the new baseline.

The first run passed the old reference gate; the two later runs failed only the
short scenario's 0.90-second relative gate. All outputs have identical sample and
channel counts. A source comparison between the three SHAs found no differences
in `src/verbx/core`, `scripts/benchmark_render_baseline.py`, or `pyproject.toml`.
Short, long-tail, and resampling times increased together by roughly 1.6–1.7×
between the first and later runs; the two later runs agree within about 3% for
these paths. This supports hosted execution variability rather than a newly
introduced Python rendering change as the immediate explanation. The tiny matrix
scenario is noisier in relative terms.

The sample is only three independent CI jobs with one measurement per scenario.
CPU models, host contention, and resolved dependency versions were not captured
in these reports. A common image is not proof of identical hardware. This is a
provisional current-host calibration, not a controlled before/after performance
study and not proof that no historical algorithmic regression exists. No claim
is made that the original alpha reference was wrong on its original host.

The calibrated relative limit is about 2.796 seconds, so the unchanged 2.50-second
absolute limit is the effective short-room gate. This intentionally replaces the
old 0.90-second effective gate; it does not disguise that change as a JIT fix.
All three archived measured results pass the new reference and existing absolute
budgets. A future slowdown above an absolute budget or a remaining relative
threshold still fails. Collect additional hosted observations, with hardware and
dependency provenance, before further recalibration or attributing historical
performance changes to code.
