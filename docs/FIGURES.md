# Illustrated Guide

This chapter collects the visual reference material used throughout the guide: signal-flow diagrams, graphs of key DSP tradeoffs, analysis dashboards, and topology sketches. Every quantitative axis is labeled with a unit; conceptual scales are explicitly marked as normalized or categorical. The figures are generated with `python3 scripts/generate_userguide_figures.py`, and this chapter is generated with `python3 scripts/generate_figure_guide.py`, so the PDF can be rebuilt reproducibly.

## System Flow

The figure below introduces **End-to-end render signal flow**. The complete render path from source file through validation, DSP, post-processing, and report output. Boxes, arrows, and spatial placement communicate topology and sequence; their dimensions are schematic and not measurements. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 1: End-to-end render signal flow.](assets/userguide_figures/01_signal_flow.png)

**Figure 1: End-to-end render signal flow.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Realtime latency budget by block size**. The duration contributed by one audio block at 48 kHz, before driver and algorithmic buffering are added. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Audio block size (frames at 48 kHz)** and the vertical axis is **One-block duration (ms)**.

![Figure 2: Realtime latency budget by block size.](assets/userguide_figures/02_realtime_latency.png)

**Figure 2: Realtime latency budget by block size.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **CLI command map**. The primary command families arranged around the verbx executable. Boxes, arrows, and spatial placement communicate topology and sequence; their dimensions are schematic and not measurements. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 18: CLI command map.](assets/userguide_figures/18_cli_command_map.png)

**Figure 18: CLI command map.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Analysis JSON structure**. The top-level input, metrics, render, and warning groups in a machine-readable sidecar. Boxes, arrows, and spatial placement communicate topology and sequence; their dimensions are schematic and not measurements. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 22: Analysis JSON structure.](assets/userguide_figures/22_json_tree.png)

**Figure 22: Analysis JSON structure.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

## Reverb Physics and Analysis

The figure below introduces **$T_{60}$ decay families**. Idealized decay slopes for several nominal reverberation times, each reaching a 60 dB loss at its labeled $T_{60}$. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Time after excitation (s)** and the vertical axis is **Relative decay level (dB)**.

![Figure 3: $T_{60}$ decay families.](assets/userguide_figures/03_rt60_decay_families.png)

**Figure 3: $T_{60}$ decay families.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Energy decay curve fitting windows**. The EDT, $T_{20}$, and $T_{30}$ regression regions used to estimate decay from different portions of an energy decay curve. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Normalized decay time (0-1)** and the vertical axis is **Energy decay level (dB)**.

![Figure 4: Energy decay curve fitting windows.](assets/userguide_figures/04_edc_fit_windows.png)

**Figure 4: Energy decay curve fitting windows.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Frequency-dependent decay bands**. Low-, mid-, and high-frequency tails with different nominal decay constants. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Time after excitation (s)** and the vertical axis is **Relative band level (linear, 0-1)**.

![Figure 9: Frequency-dependent decay bands.](assets/userguide_figures/09_multiband_decay.png)

**Figure 9: Frequency-dependent decay bands.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Room size inference curves**. Sabine-style volume estimates across $T_{60}$ for three assumed mean absorption coefficients. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Measured $T_{60}$ (s)** and the vertical axis is **Estimated room volume (m³)**.

![Figure 16: Room size inference curves.](assets/userguide_figures/16_room_size_inference.png)

**Figure 16: Room size inference curves.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Analysis metrics dashboard**. A compact view of representative $T_{60}$, DRR, $C_{80}$, peak, LUFS, and EDT fields emitted to JSON. Each card reports a separate metric with its own printed unit, so card size and position do not encode magnitude. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 17: Analysis metrics dashboard.](assets/userguide_figures/17_analysis_dashboard.png)

**Figure 17: Analysis metrics dashboard.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

## Algorithms and Processing

The figure below introduces **Feedback matrix texture heatmap**. A 16 by 16 orthogonal feedback matrix illustrating weak and strong coupling among FDN delay lines. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Destination delay line (index)** and the vertical axis is **Source delay line (index)**. The color or radial scale reports **Absolute coupling coefficient (0-1)**.

![Figure 5: Feedback matrix texture heatmap.](assets/userguide_figures/05_fdn_matrix_heatmap.png)

**Figure 5: Feedback matrix texture heatmap.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Partitioned convolution layout**. An impulse response divided into progressively larger FFT partitions to balance latency and throughput. Boxes, arrows, and spatial placement communicate topology and sequence; their dimensions are schematic and not measurements. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 11: Partitioned convolution layout.](assets/userguide_figures/11_partitioned_convolution.png)

**Figure 11: Partitioned convolution layout.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **IR morphing blend space**. A conceptual interpolation triangle connecting room, plate, and cathedral impulse-response families. Points and paths occupy a two-dimensional design space; proximity indicates similar states, not physical distance. The horizontal axis is **Blend coordinate $A$ (normalized, 0-1)** and the vertical axis is **Blend coordinate $B$ (normalized, 0-1)**.

![Figure 12: IR morphing blend space.](assets/userguide_figures/12_ir_morph_space.png)

**Figure 12: IR morphing blend space.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Shimmer feedback path**. The pitch-shift, diffusion, and feedback loop that turns late energy into a sustained harmonic layer. Boxes, arrows, and spatial placement communicate topology and sequence; their dimensions are schematic and not measurements. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 15: Shimmer feedback path.](assets/userguide_figures/15_shimmer_feedback.png)

**Figure 15: Shimmer feedback path.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Infinite-style reverb tail behavior**. Ordinary, extreme, and freeze-like tail energy compared over normalized time. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Normalized elapsed time (0-1)** and the vertical axis is **Relative tail energy (linear, 0-1)**.

![Figure 24: Infinite-style reverb tail behavior.](assets/userguide_figures/24_infinite_reverb.png)

**Figure 24: Infinite-style reverb tail behavior.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

## Controls and Tradeoffs

The figure below introduces **Analysis window function shapes**. Hann, Blackman, Kaiser, and Tukey tapers plotted over a common normalized frame. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Normalized sample position (0-1)** and the vertical axis is **Window amplitude (linear, 0-1)**.

![Figure 6: Analysis window function shapes.](assets/userguide_figures/06_window_functions.png)

**Figure 6: Analysis window function shapes.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Limiter transfer curves**. Hard, soft-knee, and transparent limiting laws compared around the ceiling region. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Input level (dBFS)** and the vertical axis is **Output level (dBFS)**.

![Figure 7: Limiter transfer curves.](assets/userguide_figures/07_limiter_transfer.png)

**Figure 7: Limiter transfer curves.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Reverb ducking envelope**. Dry-source activity and the resulting attenuation of the wet return over an eight-second example. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Time (s)** and the vertical axis is **Relative signal level (linear, 0-1)**.

![Figure 8: Reverb ducking envelope.](assets/userguide_figures/08_ducking_envelope.png)

**Figure 8: Reverb ducking envelope.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Dereverb strength versus artifact tradeoff**. The competing trends of clarity, naturalness, and aggregate usefulness as reduction strength increases. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Dereverb amount (%)** and the vertical axis is **Perceptual score (normalized, 0-1)**.

![Figure 10: Dereverb strength versus artifact tradeoff.](assets/userguide_figures/10_dereverb_tradeoff.png)

**Figure 10: Dereverb strength versus artifact tradeoff.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Block size CPU and latency tradeoff**. Normalized scheduling pressure and block latency plotted against common audio buffer sizes. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Audio block size (frames)** and the vertical axis is **Normalized cost or latency (0-1)**.

![Figure 21: Block size CPU and latency tradeoff.](assets/userguide_figures/21_cpu_block_tradeoff.png)

**Figure 21: Block size CPU and latency tradeoff.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Preset design radar**. A five-axis profile for time, tone, width, motion, and safety. Each spoke is a named category and distance from the center is the normalized radial score. The color or radial scale reports **Normalized parameter amount (0-1)**.

![Figure 23: Preset design radar.](assets/userguide_figures/23_preset_radar.png)

**Figure 23: Preset design radar.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

## Spatial and Library Views

The figure below introduces **Spatial layout families**. Listener-centered stereo, 5.1, and 7.1.4 speaker arrangements. Boxes, arrows, and spatial placement communicate topology and sequence; their dimensions are schematic and not measurements. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 13: Spatial layout families.](assets/userguide_figures/13_spatial_layouts.png)

**Figure 13: Spatial layout families.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Ambisonics order channel growth**. The quadratic channel-count growth produced by the three-dimensional Ambisonics relation $(N + 1)^{2}$. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Ambisonics order $N$ (integer)** and the vertical axis is **Channel count (channels)**.

![Figure 14: Ambisonics order channel growth.](assets/userguide_figures/14_ambisonics_order.png)

**Figure 14: Ambisonics order channel growth.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **IR library coverage grid**. Duration families crossed with synthesis methods, with four impulse responses represented per cell. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Synthesis family (category)** and the vertical axis is **Duration family (category)**. The color or radial scale reports **Impulse responses (count/cell)**.

![Figure 20: IR library coverage grid.](assets/userguide_figures/20_ir_library_grid.png)

**Figure 20: IR library coverage grid.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Reference corpus shape**. The relative sizes of the curated implementation bibliography and extended discovery index. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Reference collection (category)** and the vertical axis is **Bibliography entries (count)**.

![Figure 19: Reference corpus shape.](assets/userguide_figures/19_reference_corpus.png)

**Figure 19: Reference corpus shape.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

## Additional Diagnostics and Design Graphs

The figure below introduces **Early Reflection Timing**. Tap spacing sketches perceived room geometry before the late field blooms. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Time after direct sound (ms)** and the vertical axis is **Reflection amplitude (dBFS)**.

![Figure 25: Early Reflection Timing.](assets/userguide_figures/25_early_reflections.png)

**Figure 25: Early Reflection Timing.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Pre-Delay Perception**. A few milliseconds can separate source presence from room size. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Pre-delay (ms)** and the vertical axis is **Perceived source-room separation (normalized, 0-1)**.

![Figure 26: Pre-Delay Perception.](assets/userguide_figures/26_predelay_perception.png)

**Figure 26: Pre-Delay Perception.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Diffusion Build-Up**. Dense tails emerge as echo density crosses the fusion threshold. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Time after excitation (ms)** and the vertical axis is **Echo density (reflections/s)**.

![Figure 27: Diffusion Build-Up.](assets/userguide_figures/27_diffusion_buildup.png)

**Figure 27: Diffusion Build-Up.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Damping EQ Targets**. Low, mid, and high shelves shape the perceived material of a room. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Relative damping gain (dB)**.

![Figure 28: Damping EQ Targets.](assets/userguide_figures/28_damping_eq_targets.png)

**Figure 28: Damping EQ Targets.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Modulation Depth Safety**. Depth and rate interact: motion is useful until pitch smear takes over. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Modulation rate (Hz)** and the vertical axis is **Modulation depth (ms)**. The color or radial scale reports **Artifact risk (normalized, 0-1)**.

![Figure 29: Modulation Depth Safety.](assets/userguide_figures/29_modulation_depth_safety.png)

**Figure 29: Modulation Depth Safety.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Stereo Width Correlation**. Width controls should preserve mono safety while expanding ambience. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Stereo width (%)** and the vertical axis is **Inter-channel correlation (unitless, –1 to +1)**.

![Figure 30: Stereo Width Correlation.](assets/userguide_figures/30_stereo_width_correlation.png)

**Figure 30: Stereo Width Correlation.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Haas Zone**. Tiny left/right delays can widen image before becoming discrete echoes. Colored regions divide the horizontal quantity into operational or perceptual regimes rather than implying a continuous measured response. The horizontal axis is **Inter-channel delay (ms)** and the vertical axis is **Perceptual region (category)**.

![Figure 31: Haas Zone.](assets/userguide_figures/31_haas_zone.png)

**Figure 31: Haas Zone.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Gate Tail Shapes**. Classic gated reverb depends on hold, release, and threshold timing. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Time after transient (ms)** and the vertical axis is **Wet level (dBFS)**.

![Figure 32: Gate Tail Shapes.](assets/userguide_figures/32_gate_tail_shapes.png)

**Figure 32: Gate Tail Shapes.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Reverse Reverb Envelope**. Reverse tails rise into the transient instead of decaying away from it. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Time before transient (ms)** and the vertical axis is **Wet envelope level (linear, 0-1)**.

![Figure 33: Reverse Reverb Envelope.](assets/userguide_figures/33_reverse_reverb_envelope.png)

**Figure 33: Reverse Reverb Envelope.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Spectral Tilt Analyzer**. Broadband tilt gives a fast visual clue for dark versus bright renders. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Spectral magnitude (dBFS)**.

![Figure 34: Spectral Tilt Analyzer.](assets/userguide_figures/34_spectral_tilt_analyzer.png)

**Figure 34: Spectral Tilt Analyzer.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Loudness Normalization Path**. Peak, RMS, and LUFS views catch different gain staging problems. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Normalization method (category)** and the vertical axis is **Target or measured level (dB)**.

![Figure 35: Loudness Normalization Path.](assets/userguide_figures/35_loudness_normalization_path.png)

**Figure 35: Loudness Normalization Path.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Sample Rate Cost**. Higher sample rates improve bandwidth at a predictable CPU cost. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Sample rate (kHz)** and the vertical axis is **Relative CPU cost (%)**.

![Figure 36: Sample Rate Cost.](assets/userguide_figures/36_sample_rate_cost.png)

**Figure 36: Sample Rate Cost.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Oversampling Alias Guard**. Limiter oversampling moves foldback artifacts away from the audible band. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Frequency (kHz)** and the vertical axis is **Alias energy (dBFS)**.

![Figure 37: Oversampling Alias Guard.](assets/userguide_figures/37_oversampling_alias_guard.png)

**Figure 37: Oversampling Alias Guard.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Lookahead Limiter Timing**. Lookahead catches peaks before gain reduction becomes audible pumping. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Time relative to peak (ms)** and the vertical axis is **Gain reduction (dB)**.

![Figure 38: Lookahead Limiter Timing.](assets/userguide_figures/38_lookahead_limiter_timing.png)

**Figure 38: Lookahead Limiter Timing.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Dry/Wet Crossfade Laws**. Linear, equal-power, and DJ-style blends feel different near the center. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Dry/wet control (%)** and the vertical axis is **Channel gain (linear, 0-1)**.

![Figure 39: Dry/Wet Crossfade Laws.](assets/userguide_figures/39_dry_wet_crossfade_laws.png)

**Figure 39: Dry/Wet Crossfade Laws.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **IR Trim Finder**. Trim logic balances silence removal against preserving natural onset. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Time in impulse response (ms)** and the vertical axis is **IR amplitude (dBFS)**.

![Figure 40: IR Trim Finder.](assets/userguide_figures/40_ir_trim_finder.png)

**Figure 40: IR Trim Finder.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Silence Detector Thresholds**. Noise floors and thresholds decide where batch processing can skip work. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Detector threshold (dBFS)** and the vertical axis is **Detected activity (%)**.

![Figure 41: Silence Detector Thresholds.](assets/userguide_figures/41_silence_detector_thresholds.png)

**Figure 41: Silence Detector Thresholds.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Batch Render Throughput**. Parallel workers help until IO and memory pressure dominate. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Parallel workers (count)** and the vertical axis is **Render throughput (files/min)**.

![Figure 42: Batch Render Throughput.](assets/userguide_figures/42_batch_render_throughput.png)

**Figure 42: Batch Render Throughput.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Cache Hit Savings**. Reusable analysis and IR material turn repeated renders into fast paths. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Cache condition (category)** and the vertical axis is **Elapsed processing time (%)**.

![Figure 43: Cache Hit Savings.](assets/userguide_figures/43_cache_hit_savings.png)

**Figure 43: Cache Hit Savings.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Native Parity Slice**. A narrow deterministic slice keeps Python and native engines aligned. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 44: Native Parity Slice.](assets/userguide_figures/44_native_parity_slice.png)

**Figure 44: Native Parity Slice.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Test Matrix Coverage**. Golden audio, CLI smoke, realtime, and docs checks cover different risks. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Test family (category)** and the vertical axis is **Platform or engine (category)**. The color or radial scale reports **Coverage (%)**.

![Figure 45: Test Matrix Coverage.](assets/userguide_figures/45_test_matrix_coverage.png)

**Figure 45: Test Matrix Coverage.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Preset Morph Trajectory**. Morph paths should move smoothly through perceptual control space. Points and paths occupy a two-dimensional design space; proximity indicates similar states, not physical distance. The horizontal axis is **Timbral brightness (normalized, 0-1)** and the vertical axis is **Spatial width (normalized, 0-1)**.

![Figure 46: Preset Morph Trajectory.](assets/userguide_figures/46_preset_morph_trajectory.png)

**Figure 46: Preset Morph Trajectory.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Realtime Dropout Risk**. CPU load, block size, and driver buffers define the danger zone. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Audio block size (frames)** and the vertical axis is **Callback CPU load (%)**. The color or radial scale reports **Dropout risk (normalized, 0-1)**.

![Figure 47: Realtime Dropout Risk.](assets/userguide_figures/47_realtime_dropout_risk.png)

**Figure 47: Realtime Dropout Risk.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Release Readiness Dashboard**. The release gate is healthiest when docs, tests, render, and realtime agree. Each spoke is a named category and distance from the center is the normalized radial score. The color or radial scale reports **Completion score (normalized, 0-1)**.

![Figure 48: Release Readiness Dashboard.](assets/userguide_figures/48_release_readiness_dashboard.png)

**Figure 48: Release Readiness Dashboard.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

## Extended Figure Atlas

The figure below introduces **Comb Filter Notches**. Short delays carve predictable notches that can make tails metallic. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Magnitude response (dB)**.

![Figure 49: Comb Filter Notches.](assets/userguide_figures/49_comb_filter_notches.png)

**Figure 49: Comb Filter Notches.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Allpass Diffuser Response**. Allpass stages preserve energy while scrambling phase and timing. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Group delay (ms)**.

![Figure 50: Allpass Diffuser Response.](assets/userguide_figures/50_allpass_diffuser_response.png)

**Figure 50: Allpass Diffuser Response.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **FDN Delay Distribution**. Prime-ish delay spacing avoids obvious repeating echo patterns. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **FDN delay line (index)** and the vertical axis is **Delay length (samples)**.

![Figure 51: FDN Delay Distribution.](assets/userguide_figures/51_fdn_delay_distribution.png)

**Figure 51: FDN Delay Distribution.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Modal Density Growth**. Large rooms pack more resonances into each octave. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Modes per octave (count)**.

![Figure 52: Modal Density Growth.](assets/userguide_figures/52_modal_density_growth.png)

**Figure 52: Modal Density Growth.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Schroeder Frequency Estimate**. Below the transition band, individual modes matter more. Colored regions divide the horizontal quantity into operational or perceptual regimes rather than implying a continuous measured response. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Acoustic behavior (category)**.

![Figure 53: Schroeder Frequency Estimate.](assets/userguide_figures/53_schroeder_frequency_estimate.png)

**Figure 53: Schroeder Frequency Estimate.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Air Absorption Roll-Off**. Long bright tails need damping to avoid synthetic glare. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Air attenuation (dB/m)**.

![Figure 54: Air Absorption Roll-Off.](assets/userguide_figures/54_air_absorption_rolloff.png)

**Figure 54: Air Absorption Roll-Off.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Material Absorption Map**. Wall, carpet, curtain, and glass assumptions shape decay by band. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Frequency band (Hz)** and the vertical axis is **Surface material (category)**. The color or radial scale reports **Absorption coefficient (0-1)**.

![Figure 55: Material Absorption Map.](assets/userguide_figures/55_material_absorption_map.png)

**Figure 55: Material Absorption Map.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Early Late Balance**. Presence comes from early energy; envelopment comes from late energy. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Energy component (category)** and the vertical axis is **Relative energy (%)**.

![Figure 56: Early Late Balance.](assets/userguide_figures/56_early_late_balance.png)

**Figure 56: Early Late Balance.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Source Distance Cue**. Distance changes direct-to-reverberant balance before it changes tone. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Source distance (m)** and the vertical axis is **Direct-to-reverberant ratio (dB)**.

![Figure 57: Source Distance Cue.](assets/userguide_figures/57_source_distance_cue.png)

**Figure 57: Source Distance Cue.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Mic Pattern Pickup**. Cardioid, omni, and figure-eight captures feed the room differently. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Arrival angle (degrees)** and the vertical axis is **Relative microphone sensitivity (dB)**.

![Figure 58: Mic Pattern Pickup.](assets/userguide_figures/58_mic_pattern_pickup.png)

**Figure 58: Mic Pattern Pickup.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Sidechain Detector Modes**. Peak and RMS detectors react on different musical timescales. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Time after onset (ms)** and the vertical axis is **Detector envelope (linear, 0-1)**.

![Figure 59: Sidechain Detector Modes.](assets/userguide_figures/59_sidechain_detector_modes.png)

**Figure 59: Sidechain Detector Modes.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Ducking Release Families**. Release curvature decides whether ambience breathes or pumps. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Time after sidechain release (ms)** and the vertical axis is **Wet gain (dB)**.

![Figure 60: Ducking Release Families.](assets/userguide_figures/60_ducking_release_families.png)

**Figure 60: Ducking Release Families.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Limiter Knee Families**. Knee width trades transparent onset against strict peak containment. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Input level relative to threshold (dB)** and the vertical axis is **Output level (dBFS)**.

![Figure 61: Limiter Knee Families.](assets/userguide_figures/61_limiter_knee_families.png)

**Figure 61: Limiter Knee Families.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **True Peak Margin**. Inter-sample peaks make safety margin useful even when samples look safe. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Time around sample peak (µs)** and the vertical axis is **Signal level (dBFS)**.

![Figure 62: True Peak Margin.](assets/userguide_figures/62_true_peak_margin.png)

**Figure 62: True Peak Margin.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **LUFS Integration Windows**. Momentary, short-term, and integrated loudness answer different questions. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Program time (s)** and the vertical axis is **Integrated loudness (LUFS)**.

![Figure 63: LUFS Integration Windows.](assets/userguide_figures/63_lufs_integration_windows.png)

**Figure 63: LUFS Integration Windows.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Crest Factor Map**. Transient-heavy inputs need different limiter behavior from pads. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **RMS level (dBFS)** and the vertical axis is **Peak level (dBFS)**. The color or radial scale reports **Crest factor (dB)**.

![Figure 64: Crest Factor Map.](assets/userguide_figures/64_crest_factor_map.png)

**Figure 64: Crest Factor Map.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Transient Preservation**. Dereverb should reduce tail energy without flattening attack detail. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Time around transient (ms)** and the vertical axis is **Signal amplitude (linear, –1 to +1)**.

![Figure 65: Transient Preservation.](assets/userguide_figures/65_transient_preservation.png)

**Figure 65: Transient Preservation.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Dereverb Mask Strength**. Mask aggressiveness governs the speech-cleanup versus artifact tradeoff. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Dereverb mask strength (%)** and the vertical axis is **Artifact or suppression score (normalized, 0-1)**.

![Figure 66: Dereverb Mask Strength.](assets/userguide_figures/66_dereverb_mask_strength.png)

**Figure 66: Dereverb Mask Strength.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Spectral Gate Residuals**. Residual maps reveal where dereverb is leaving flutter or musical noise. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Time (s)** and the vertical axis is **Frequency (Hz)**. The color or radial scale reports **Residual magnitude (dBFS)**.

![Figure 67: Spectral Gate Residuals.](assets/userguide_figures/67_spectral_gate_residuals.png)

**Figure 67: Spectral Gate Residuals.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Noise Floor Tracking**. Slow floor estimates avoid chasing quiet reverb as if it were noise. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Program time (s)** and the vertical axis is **Estimated noise floor (dBFS)**.

![Figure 68: Noise Floor Tracking.](assets/userguide_figures/68_noise_floor_tracking.png)

**Figure 68: Noise Floor Tracking.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Multichannel Routing Matrix**. Channel maps keep immersive and stereo renders auditable. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Input channel (index)** and the vertical axis is **Output channel (index)**. The color or radial scale reports **Routing gain (dB)**.

![Figure 69: Multichannel Routing Matrix.](assets/userguide_figures/69_multichannel_routing_matrix.png)

**Figure 69: Multichannel Routing Matrix.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Ambisonic Decode Spread**. Decode spread converts abstract soundfield order into speaker energy. Points and paths occupy a two-dimensional design space; proximity indicates similar states, not physical distance. The horizontal axis is **Speaker azimuth (degrees)** and the vertical axis is **Speaker elevation (degrees)**. The color or radial scale reports **Relative decode energy (0-1)**.

![Figure 70: Ambisonic Decode Spread.](assets/userguide_figures/70_ambisonic_decode_spread.png)

**Figure 70: Ambisonic Decode Spread.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Binaural HRTF Blend**. HRTF blending needs smooth interpolation across azimuth and elevation. Points and paths occupy a two-dimensional design space; proximity indicates similar states, not physical distance. The horizontal axis is **Source azimuth (degrees)** and the vertical axis is **Source elevation (degrees)**. The color or radial scale reports **HRTF blend weight (0-1)**.

![Figure 71: Binaural HRTF Blend.](assets/userguide_figures/71_binaural_hrtf_blend.png)

**Figure 71: Binaural HRTF Blend.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Loudspeaker Layouts: Plan and Elevation**. Nominal channel bearings for stereo, 5.1, and 7.1.4, with the immersive height layer shown separately. Listener-centered plan views encode nominal azimuth, while the separate side elevation distinguishes bed and height layers. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 72: Loudspeaker Layouts: Plan and Elevation.](assets/userguide_figures/72_speaker_layout_coverage.png)

**Figure 72: Loudspeaker Layouts: Plan and Elevation.**

Read each plan with front at the top and the listener at the center. Blue marks identify front and center bed channels, teal marks identify side and rear bed channels, and gold marks identify overhead channels. Radial lines indicate nominal bearing only; they are not cables or signal-flow paths. The elevation inset shows why the four height channels cannot be understood from azimuth alone, while the separate LFE key emphasizes that the subwoofer channel has no prescribed bearing. These angles are explanatory nominal placements; use the applicable monitoring standard and room-calibration procedure for installation.

The figure below introduces **IR Capture Checklist**. Capture quality depends on sweep level, silence, trim, and calibration. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 73: IR Capture Checklist.](assets/userguide_figures/73_ir_capture_checklist.png)

**Figure 73: IR Capture Checklist.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Sweep Deconvolution Path**. Measured IRs move through sweep, inverse filter, trim, and normalize steps. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 74: Sweep Deconvolution Path.](assets/userguide_figures/74_sweep_deconvolution_path.png)

**Figure 74: Sweep Deconvolution Path.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **IR Tail Trim Decision**. Trim should stop after useful decay, not after the first low-energy valley. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Time in impulse response (s)** and the vertical axis is **IR decay level (dBFS)**.

![Figure 75: IR Tail Trim Decision.](assets/userguide_figures/75_ir_tail_trim_decision.png)

**Figure 75: IR Tail Trim Decision.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **IR Normalization Modes**. Peak, energy, and loudness normalization each preserve a different invariant. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Normalization mode (category)** and the vertical axis is **Resulting reference level (dB)**.

![Figure 76: IR Normalization Modes.](assets/userguide_figures/76_ir_normalization_modes.png)

**Figure 76: IR Normalization Modes.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Convolution Partition Plan**. Small early partitions and larger late partitions balance CPU and latency. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **IR time offset (samples)** and the vertical axis is **Partition size (samples)**.

![Figure 77: Convolution Partition Plan.](assets/userguide_figures/77_convolution_partition_plan.png)

**Figure 77: Convolution Partition Plan.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **FFT Size Efficiency**. FFT cost rises in steps, so partition sizes should sit on friendly powers. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **FFT size (samples)** and the vertical axis is **Relative processing cost (%)**.

![Figure 78: FFT Size Efficiency.](assets/userguide_figures/78_fft_size_efficiency.png)

**Figure 78: FFT Size Efficiency.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **SIMD Batch Shape**. Native kernels are fastest when channel and frame batches align cleanly. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Frame batch size (frames)** and the vertical axis is **Channel batch size (channels)**. The color or radial scale reports **SIMD utilization (%)**.

![Figure 79: SIMD Batch Shape.](assets/userguide_figures/79_simd_batch_shape.png)

**Figure 79: SIMD Batch Shape.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Memory Bandwidth Pressure**. Long tails can become bandwidth-bound before they become math-bound. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Impulse-response duration (s)** and the vertical axis is **Memory bandwidth (GB/s)**.

![Figure 80: Memory Bandwidth Pressure.](assets/userguide_figures/80_memory_bandwidth_pressure.png)

**Figure 80: Memory Bandwidth Pressure.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Thread Pool Scaling**. More workers help until contention and IO erase the gain. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Worker threads (count)** and the vertical axis is **Speedup (× realtime)**.

![Figure 81: Thread Pool Scaling.](assets/userguide_figures/81_thread_pool_scaling.png)

**Figure 81: Thread Pool Scaling.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Realtime Callback Budget**. Audio callbacks need headroom below the hard deadline. Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible. The horizontal axis is **Callback time (ms)** and the vertical axis is **Budget use (%)**.

![Figure 82: Realtime Callback Budget.](assets/userguide_figures/82_realtime_callback_budget.png)

**Figure 82: Realtime Callback Budget.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **XRuns By Block Size**. Dropout risk falls with buffer size but latency rises in exchange. The trace shows how the vertical response changes as the horizontal control or measurement advances. The horizontal axis is **Audio block size (frames)** and the vertical axis is **XRuns (count/hour)**.

![Figure 83: XRuns By Block Size.](assets/userguide_figures/83_xruns_by_block_size.png)

**Figure 83: XRuns By Block Size.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Device Buffer Stack**. Round-trip latency is the sum of API, driver, hardware, and DSP buffers. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 84: Device Buffer Stack.](assets/userguide_figures/84_device_buffer_stack.png)

**Figure 84: Device Buffer Stack.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **CLI Option Families**. Render, analysis, IR, limiter, realtime, and batch flags cluster by job. Each spoke is a named category and distance from the center is the normalized radial score. The color or radial scale reports **Option coverage (normalized, 0-1)**.

![Figure 85: CLI Option Families.](assets/userguide_figures/85_cli_option_families.png)

**Figure 85: CLI Option Families.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Preset Taxonomy**. Presets should be searchable by space, tone, motion, and safety intent. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Preset attribute (category)** and the vertical axis is **Preset family (category)**. The color or radial scale reports **Library density (presets/cell)**.

![Figure 86: Preset Taxonomy.](assets/userguide_figures/86_preset_taxonomy.png)

**Figure 86: Preset Taxonomy.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **JSON Schema Coverage**. Schema fields make render results machine-checkable across releases. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Schema field group (category)** and the vertical axis is **Command (category)**. The color or radial scale reports **Fields implemented (%)**.

![Figure 87: JSON Schema Coverage.](assets/userguide_figures/87_json_schema_coverage.png)

**Figure 87: JSON Schema Coverage.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Analysis Regression Bands**. Metric tolerances should be tight where outputs are deterministic. Colored regions divide the horizontal quantity into operational or perceptual regimes rather than implying a continuous measured response. The horizontal axis is **Regression metric (category)** and the vertical axis is **Allowed deviation (%)**.

![Figure 88: Analysis Regression Bands.](assets/userguide_figures/88_analysis_regression_bands.png)

**Figure 88: Analysis Regression Bands.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Golden Audio Drift**. Golden fixtures reveal unexpected changes in gain, tail, or spectrum. The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly. The horizontal axis is **Frequency (Hz)** and the vertical axis is **Golden-output drift (dB)**.

![Figure 89: Golden Audio Drift.](assets/userguide_figures/89_golden_audio_drift.png)

**Figure 89: Golden Audio Drift.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Documentation Build Pipeline**. Markdown, figures, Pandoc, LaTeX, and PDF checks form one reproducible chain. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 90: Documentation Build Pipeline.](assets/userguide_figures/90_documentation_build_pipeline.png)

**Figure 90: Documentation Build Pipeline.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Table Wrap Stress Test**. Long CLI options and URLs need wrapping before they hit the page edge. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Table column width (characters)** and the vertical axis is **Content length (characters)**. The color or radial scale reports **Overflow risk (normalized, 0-1)**.

![Figure 91: Table Wrap Stress Test.](assets/userguide_figures/91_table_wrap_stress_test.png)

**Figure 91: Table Wrap Stress Test.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Reference Annotation Flow**. Curated notes, extended entries, and cross-links serve different readers. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 92: Reference Annotation Flow.](assets/userguide_figures/92_reference_annotation_flow.png)

**Figure 92: Reference Annotation Flow.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Citation Corpus Growth**. Reference expansion should grow breadth without diluting implementation guidance. Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition. The horizontal axis is **Documentation release (version index)** and the vertical axis is **References (count)**.

![Figure 93: Citation Corpus Growth.](assets/userguide_figures/93_citation_corpus_growth.png)

**Figure 93: Citation Corpus Growth.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Release Branch Flow**. Feature, docs, CI, tag, and package steps should stay visible. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 94: Release Branch Flow.](assets/userguide_figures/94_release_branch_flow.png)

**Figure 94: Release Branch Flow.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Homebrew Formula Refresh**. Version, checksum, bottle, and audit checks keep installs boring. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 95: Homebrew Formula Refresh.](assets/userguide_figures/95_homebrew_formula_refresh.png)

**Figure 95: Homebrew Formula Refresh.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Platform Support Grid**. macOS, Linux, and Windows support differs across audio backends. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Feature or backend (category)** and the vertical axis is **Operating system (category)**. The color or radial scale reports **Support state (category)**.

![Figure 96: Platform Support Grid.](assets/userguide_figures/96_platform_support_grid.png)

**Figure 96: Platform Support Grid.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Error Message Quality**. Good CLI errors name the bad input, likely cause, and next command. Each spoke is a named category and distance from the center is the normalized radial score. The color or radial scale reports **Quality score (normalized, 0-1)**.

![Figure 97: Error Message Quality.](assets/userguide_figures/97_error_message_quality.png)

**Figure 97: Error Message Quality.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **User Workflow Map**. Explore, render, analyze, compare, and automate form the core loop. Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale. It has no numeric axes because it is a structural diagram rather than a measurement plot.

![Figure 98: User Workflow Map.](assets/userguide_figures/98_user_workflow_map.png)

**Figure 98: User Workflow Map.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **Feature Maturity Radar**. Stable, beta, experimental, and roadmap features need distinct labeling. Each spoke is a named category and distance from the center is the normalized radial score. The color or radial scale reports **Maturity score (normalized, 0-1)**.

![Figure 99: Feature Maturity Radar.](assets/userguide_figures/99_feature_maturity_radar.png)

**Figure 99: Feature Maturity Radar.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.

The figure below introduces **End-to-End Confidence Map**. The full system is healthiest when audio, docs, tests, and packaging align. Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label. The horizontal axis is **Subsystem (category)** and the vertical axis is **Verification layer (category)**. The color or radial scale reports **Confidence score (normalized, 0-1)**.

![Figure 100: End-to-End Confidence Map.](assets/userguide_figures/100_end_to_end_confidence_map.png)

**Figure 100: End-to-End Confidence Map.**

Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test.
