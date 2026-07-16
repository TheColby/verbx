#!/usr/bin/env python3
"""Generate the illustrated-guide Markdown with extended accessible descriptions."""

from __future__ import annotations

from pathlib import Path

from generate_userguide_figures import ATLAS_AXES, EXTRA_FIGURES, MORE_FIGURES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "FIGURES.md"

CORE = {
    1: ("End-to-end render signal flow", "The complete render path from source file through validation, DSP, post-processing, and report output.", "schematic", "", "", ""),
    2: ("Realtime latency budget by block size", "The duration contributed by one audio block at 48 kHz, before driver and algorithmic buffering are added.", "bars", "Audio block size (frames at 48 kHz)", "One-block duration (ms)", ""),
    3: ("$T_{60}$ decay families", "Idealized decay slopes for several nominal reverberation times, each reaching a 60 dB loss at its labeled $T_{60}$.", "multi", "Time after excitation (s)", "Relative decay level (dB)", ""),
    4: ("Energy decay curve fitting windows", "The EDT, $T_{20}$, and $T_{30}$ regression regions used to estimate decay from different portions of an energy decay curve.", "multi", "Normalized decay time (0-1)", "Energy decay level (dB)", ""),
    5: ("Feedback matrix texture heatmap", "A 16 by 16 orthogonal feedback matrix illustrating weak and strong coupling among FDN delay lines.", "heat", "Destination delay line (index)", "Source delay line (index)", "Absolute coupling coefficient (0-1)"),
    6: ("Analysis window function shapes", "Hann, Blackman, Kaiser, and Tukey tapers plotted over a common normalized frame.", "multi", "Normalized sample position (0-1)", "Window amplitude (linear, 0-1)", ""),
    7: ("Limiter transfer curves", "Hard, soft-knee, and transparent limiting laws compared around the ceiling region.", "multi", "Input level (dBFS)", "Output level (dBFS)", ""),
    8: ("Reverb ducking envelope", "Dry-source activity and the resulting attenuation of the wet return over an eight-second example.", "multi", "Time (s)", "Relative signal level (linear, 0-1)", ""),
    9: ("Frequency-dependent decay bands", "Low-, mid-, and high-frequency tails with different nominal decay constants.", "multi", "Time after excitation (s)", "Relative band level (linear, 0-1)", ""),
    10: ("Dereverb strength versus artifact tradeoff", "The competing trends of clarity, naturalness, and aggregate usefulness as reduction strength increases.", "multi", "Dereverb amount (%)", "Perceptual score (normalized, 0-1)", ""),
    11: ("Partitioned convolution layout", "An impulse response divided into progressively larger FFT partitions to balance latency and throughput.", "schematic", "", "", ""),
    12: ("IR morphing blend space", "A conceptual interpolation triangle connecting room, plate, and cathedral impulse-response families.", "space", "Blend coordinate $A$ (normalized, 0-1)", "Blend coordinate $B$ (normalized, 0-1)", ""),
    13: ("Spatial layout families", "Listener-centered stereo, 5.1, and 7.1.4 speaker arrangements.", "schematic", "", "", ""),
    14: ("Ambisonics order channel growth", "The quadratic channel-count growth produced by the three-dimensional Ambisonics relation $(N + 1)^{2}$.", "bars", "Ambisonics order $N$ (integer)", "Channel count (channels)", ""),
    15: ("Shimmer feedback path", "The pitch-shift, diffusion, and feedback loop that turns late energy into a sustained harmonic layer.", "schematic", "", "", ""),
    16: ("Room size inference curves", "Sabine-style volume estimates across $T_{60}$ for three assumed mean absorption coefficients.", "multi", "Measured $T_{60}$ (s)", "Estimated room volume (m³)", ""),
    17: ("Analysis metrics dashboard", "A compact view of representative $T_{60}$, DRR, $C_{80}$, peak, LUFS, and EDT fields emitted to JSON.", "dashboard", "", "", ""),
    18: ("CLI command map", "The primary command families arranged around the verbx executable.", "schematic", "", "", ""),
    19: ("Reference corpus shape", "The relative sizes of the curated implementation bibliography and extended discovery index.", "bars", "Reference collection (category)", "Bibliography entries (count)", ""),
    20: ("IR library coverage grid", "Duration families crossed with synthesis methods, with four impulse responses represented per cell.", "heat", "Synthesis family (category)", "Duration family (category)", "Impulse responses (count/cell)"),
    21: ("Block size CPU and latency tradeoff", "Normalized scheduling pressure and block latency plotted against common audio buffer sizes.", "multi", "Audio block size (frames)", "Normalized cost or latency (0-1)", ""),
    22: ("Analysis JSON structure", "The top-level input, metrics, render, and warning groups in a machine-readable sidecar.", "schematic", "", "", ""),
    23: ("Preset design radar", "A five-axis profile for time, tone, width, motion, and safety.", "radar", "Preset dimension (category)", "", "Normalized parameter amount (0-1)"),
    24: ("Infinite-style reverb tail behavior", "Ordinary, extreme, and freeze-like tail energy compared over normalized time.", "multi", "Normalized elapsed time (0-1)", "Relative tail energy (linear, 0-1)", ""),
}

SECTIONS = (
    ("System Flow", (1, 2, 18, 22)),
    ("Reverb Physics and Analysis", (3, 4, 9, 16, 17)),
    ("Algorithms and Processing", (5, 11, 12, 15, 24)),
    ("Controls and Tradeoffs", (6, 7, 8, 10, 21, 23)),
    ("Spatial and Library Views", (13, 14, 20, 19)),
    ("Additional Diagnostics and Design Graphs", tuple(range(25, 49))),
    ("Extended Figure Atlas", tuple(range(49, 101))),
)

VISUAL_LANGUAGE = {
    "bars": "Bar height encodes the quantity on the vertical axis, while each horizontal category identifies a tested or illustrative condition.",
    "curve": "The trace shows how the vertical response changes as the horizontal control or measurement advances.",
    "multi": "The colored traces share one coordinate system so their slopes, crossings, and endpoints can be compared directly.",
    "timeline": "Vertical events and the envelope are positioned against a common time base, making onset, hold, and decay relationships visible.",
    "heat": "Each cell combines one horizontal and one vertical condition; color encodes the third quantity named in the scale label.",
    "bands": "Colored regions divide the horizontal quantity into operational or perceptual regimes rather than implying a continuous measured response.",
    "space": "Points and paths occupy a two-dimensional design space; proximity indicates similar states, not physical distance.",
    "layout": (
        "Listener-centered plan views encode nominal azimuth, while the separate side "
        "elevation distinguishes bed and height layers."
    ),
    "radar": "Each spoke is a named category and distance from the center is the normalized radial score.",
    "stack": "Boxes and arrows show order and dependency. Their position and size are schematic and carry no numeric scale.",
    "schematic": "Boxes, arrows, and spatial placement communicate topology and sequence; their dimensions are schematic and not measurements.",
    "dashboard": "Each card reports a separate metric with its own printed unit, so card size and position do not encode magnitude.",
}


def metadata() -> dict[int, tuple[str, str, str, str, str, str, str]]:
    items = {number: (*values, f"{number:02d}_placeholder.png") for number, values in CORE.items()}
    filenames = {
        1: "01_signal_flow.png", 2: "02_realtime_latency.png", 3: "03_rt60_decay_families.png",
        4: "04_edc_fit_windows.png", 5: "05_fdn_matrix_heatmap.png", 6: "06_window_functions.png",
        7: "07_limiter_transfer.png", 8: "08_ducking_envelope.png", 9: "09_multiband_decay.png",
        10: "10_dereverb_tradeoff.png", 11: "11_partitioned_convolution.png", 12: "12_ir_morph_space.png",
        13: "13_spatial_layouts.png", 14: "14_ambisonics_order.png", 15: "15_shimmer_feedback.png",
        16: "16_room_size_inference.png", 17: "17_analysis_dashboard.png", 18: "18_cli_command_map.png",
        19: "19_reference_corpus.png", 20: "20_ir_library_grid.png", 21: "21_cpu_block_tradeoff.png",
        22: "22_json_tree.png", 23: "23_preset_radar.png", 24: "24_infinite_reverb.png",
    }
    for number, filename in filenames.items():
        items[number] = (*CORE[number], filename)
    for title, subtitle, filename, kind, number in (*EXTRA_FIGURES, *MORE_FIGURES):
        xlab, ylab, scale = ATLAS_AXES.get(number, ("", "", ""))
        items[number] = (title, subtitle, kind, xlab, ylab, scale, filename)
    return items


def describe(item: tuple[str, str, str, str, str, str, str]) -> tuple[str, str]:
    title, summary, kind, xlab, ylab, scale, _ = item
    axis_sentence = ""
    if xlab and ylab:
        axis_sentence = f" The horizontal axis is **{xlab}** and the vertical axis is **{ylab}**."
    if scale:
        axis_sentence += f" The color or radial scale reports **{scale}**."
    if not xlab and not ylab and not scale:
        axis_sentence = " It has no numeric axes because it is a structural diagram rather than a measurement plot."

    caveat = (
        "Unless the figure explicitly prints measured values, the geometry is an explanatory model rather than a benchmark from a specific audio file. "
        "Use `verbx analyze` and its JSON report when exact values are needed for a render, device, room, or regression test."
    )
    lead = (
        f"The figure below introduces **{title}**. {summary} "
        f"{VISUAL_LANGUAGE[kind]}{axis_sentence}"
    )
    if kind == "layout":
        return lead, (
            "Read each plan with front at the top and the listener at the center. Blue "
            "marks identify front and center bed channels, teal marks identify side and "
            "rear bed channels, and gold marks identify overhead channels. Radial lines "
            "indicate nominal bearing only; they are not cables or signal-flow paths. The "
            "elevation inset shows why the four height channels cannot be understood from "
            "azimuth alone, while the separate LFE key emphasizes that the subwoofer "
            "channel has no prescribed bearing. These angles are explanatory nominal "
            "placements; use the applicable monitoring standard and room-calibration "
            "procedure for installation."
        )
    follow = (
        "Read the figure from the labeled input or independent dimension toward the reported response, then compare color, slope, area, or stage order as appropriate. "
        "Its practical purpose is to make the relevant verbx control or engineering tradeoff easier to predict before listening: abrupt changes suggest sensitive settings, broad regions suggest forgiving settings, and converging traces suggest conditions that should sound or measure similarly. "
        f"{caveat}"
    )
    return lead, follow


def main() -> int:
    items = metadata()
    lines = [
        "# Illustrated Guide",
        "",
        "This chapter collects the visual reference material used throughout the guide: signal-flow diagrams, graphs of key DSP tradeoffs, analysis dashboards, and topology sketches. Every quantitative axis is labeled with a unit; conceptual scales are explicitly marked as normalized or categorical. The figures are generated with `python3 scripts/generate_userguide_figures.py`, and this chapter is generated with `python3 scripts/generate_figure_guide.py`, so the PDF can be rebuilt reproducibly.",
        "",
    ]
    for heading, numbers in SECTIONS:
        lines.extend((f"## {heading}", ""))
        for number in numbers:
            item = items[number]
            title, *_, filename = item
            lead, follow = describe(item)
            lines.extend((
                lead, "",
                f"![Figure {number}: {title}.](assets/userguide_figures/{filename})", "",
                f"**Figure {number}: {title}.**", "",
                follow, "",
            ))
    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(items)} extended figure descriptions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
