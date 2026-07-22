#!/usr/bin/env python3
"""Build the consolidated USERGUIDE Markdown and PDF outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
DEFAULT_MD = ROOT / "docs" / "USERGUIDE.md"
DEFAULT_PDF = ROOT / "USERGUIDE.pdf"
PDF_PREAMBLE = ROOT / "docs" / "assets" / "pandoc_pdf_preamble.tex"
CARD_ILLUSTRATIONS = ROOT / "docs" / "assets" / "verbx_card_illustrations.tex"
TABLE_CAPTION_FILTER = ROOT / "docs" / "assets" / "caption_tables.lua"
DIRECTIONAL_QUOTES_FILTER = ROOT / "docs" / "assets" / "directional_quotes.lua"
INDEX_STYLE = ROOT / "docs" / "assets" / "verbx_index.ist"
REFERENCE_METADATA = ROOT / "docs" / "reference_metadata.json"
PLUGIN_GUIDE_GENERATOR = ROOT / "scripts_generate_plugin_guide.py"
BOOK_SUPPLEMENT_GENERATOR = ROOT / "scripts_generate_book_supplements.py"
GLOSSARY_GENERATOR = ROOT / "scripts_generate_glossary.py"
LITERATURE_SORTER = ROOT / "scripts_sort_literature.py"
COMPOSITION_YEAR_NORMALIZER = ROOT / "scripts_normalize_composition_years.py"
TYPOGRAPHY_CHECKER = ROOT / "scripts_check_typography.py"
REVERB_PRIMER_ASSET_GENERATOR = ROOT / "scripts" / "generate_reverb_primer_assets.py"
TERMINAL_ASSET_GENERATOR = ROOT / "scripts" / "generate_terminal_screenshots.py"
IMMERSIVE_AUDIO_ASSET_GENERATOR = ROOT / "scripts" / "generate_immersive_audio_figures.py"
AI_AUGMENTATION_ASSET_GENERATOR = ROOT / "scripts" / "generate_ai_augmentation_figures.py"
DEFAULT_AUTHOR = "Colby Leider, PhD"
BOOK_EDITION_DATE = "July 26, 2026"
RESEARCH_REFERENCE_PATTERN = re.compile(
    r"(?m)^\*\*\[(?P<key>[^]]+)\]\*\*\s+(?P<authors>.+?)\s+"
    r"\((?P<year>(?:18|19|20)\d{2}[a-z]?|n\.d\.)\)\.\s+"
    r"(?P<title>.+?)\.\s+\*(?P<venue>[^*]+)\*\.\s+"
    r"(?:DOI:\s+\[(?P<doi>[^]]+)\]\(https://doi\.org/(?P=doi)\)"
    r"|Source:\s+\[(?P<source_label>[^]]+)\]\((?P<source_url>[^)]+)\)"
    r"|(?P<note>\(Also listed as \[[^]]+\]\)))"
)
TEX_GYRE_FONT_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/tex-gyre/"
TEX_GYRE_MATH_FONT_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/tex-gyre-math/"
HOMEBREW_SOURCE = ROOT / "docs" / "HOMEBREW.md"
ARTIFICIAL_REVERB_HISTORY_SOURCE = (
    ROOT / "docs" / "HISTORY_OF_ARTIFICIAL_REVERBERATION.md"
)
BENCHMARK_SOURCE = ROOT / "docs" / "benchmarks" / "README.md"
OPEN_SOURCE_PORTFOLIO_SOURCE = ROOT / "docs" / "OPEN_SOURCE_IMAGE_PORTFOLIO.md"

USERGUIDE_SOURCES: tuple[Path, ...] = (
    ROOT / "README.md",
    ROOT / "docs" / "IMMERSIVE_AUDIO.md",
    ROOT / "docs" / "INTRODUCTORY_BLOCK_DIAGRAMS.md",
    ROOT / "docs" / "CLI_REFERENCE.md",
    ROOT / "docs" / "EXTREME_COOKBOOK.md",
    ROOT / "docs" / "PLUGIN_GUIDE.md",
    ROOT / "docs" / "IR_SYNTHESIS.md",
    ROOT / "docs" / "AI_AUGMENTATION.md",
    ROOT / "docs" / "SCHEMA_REFERENCE.md",
    ROOT / "docs" / "IR_MORPH_QA.md",
    ROOT / "docs" / "SOFA_FEASIBILITY.md",
    ROOT / "docs" / "FIGURES.md",
    ROOT / "docs" / "MUSICAL_PIECES_APPENDIX.md",
    ROOT / "docs" / "MUSICAL_PIECES_EXPANSION.md",
    ROOT / "docs" / "HOMEWORK_ASSIGNMENTS.md",
    ROOT / "docs" / "FINITE_ELEMENT_MODELING.md",
    ROOT / "docs" / "MICROTONAL_SCALA_WORKFLOWS.md",
    ROOT / "docs" / "REFERENCES.md",
    ROOT / "docs" / "FAQ.md",
    ROOT / "docs" / "PUBLIC_ALPHA_NOTES.md",
    ROOT / "docs" / "GLOSSARY.md",
)
CHAPTER_ONE_SUPPLEMENTS: tuple[Path, ...] = (HOMEBREW_SOURCE,)
CHAPTER_TWO_SUPPLEMENTS: tuple[Path, ...] = (
    ARTIFICIAL_REVERB_HISTORY_SOURCE,
    OPEN_SOURCE_PORTFOLIO_SOURCE,
)
REFERENCE_CHAPTER_SUPPLEMENTS: tuple[Path, ...] = (BENCHMARK_SOURCE,)
USERGUIDE_INCLUDED_SOURCES: tuple[Path, ...] = (
    USERGUIDE_SOURCES[0],
    *CHAPTER_ONE_SUPPLEMENTS,
    *CHAPTER_TWO_SUPPLEMENTS,
    *REFERENCE_CHAPTER_SUPPLEMENTS,
    *USERGUIDE_SOURCES[1:],
)

CHAPTER_EPIGRAPHS: tuple[tuple[str, str, str], ...] = (
    (
        "Let echo, too, perform her part, prolonging every note with art.",
        "Joseph Addison, *Ode for St. Cecilia's Day*",
        "1699",
    ),
    (
        "Because of the reverberation, there's always more to the sound than just the sound.",
        "Pauline Oliveros",
        "Date unknown",
    ),
    (
        "Multitudinous echoes awoke and died in the distance.",
        "Henry Wadsworth Longfellow, *Evangeline*",
        "1847",
    ),
    (
        "Sweetest Echo, sweetest nymph, that liv'st unseen within thy airy shell.",
        "John Milton, *Comus*",
        "1637",
    ),
    (
        "How sweet the answer Echo makes to music at night.",
        "Thomas Moore, *Echo*",
        "1821",
    ),
    (
        "And more than echoes talk along the walls.",
        "Alexander Pope, *Eloisa to Abelard*",
        "1717",
    ),
    (
        "The reverb was so long that a trainwreck would sound good in there.",
        "Frank Speller (1938–2017), American organist and composer, on his recital in Westminster Abbey",
        "1992",
    ),
    (
        "Repeating your ultimate word.",
        "John Godfrey Saxe, *The Story of Echo*",
        "1865",
    ),
    (
        "As if a double hunt were heard at once.",
        "William Shakespeare, *Titus Andronicus*",
        "1594",
    ),
    (
        "Lost Echo sits amid the voiceless mountains, and feeds her grief.",
        "Percy Bysshe Shelley, *Adonais*",
        "1821",
    ),
    (
        "I used the bathtub for reverberation.",
        "Pauline Oliveros",
        "January 2003",
    ),
    (
        "The only technical things I know are treble, volume and reverb, that's all.",
        "Johnny Thunders",
        "Date unknown",
    ),
)

# Exercises belong to the principal teaching chapters. Appendices, the command
# schema, bibliography, FAQ, alpha notes, and glossary are reference matter.
CHAPTER_EXERCISES: dict[str, tuple[str, ...]] = {
    "verbx": (
        "Render one dry excerpt through a short room, a plate, and a long hall; loudness-match the results and describe the change in distance, width, and spectral decay.",
        "Create a fully wet return for one source and a parallel wet/dry insert for the same source. Compare the two routing models in a short mix.",
        "Run `verbx analyze` on a dry file and one rendered file, then identify which reported quantities support, complicate, or contradict your listening notes.",
        "Make three deterministic variations of one preset by changing only RT60, pre-delay, and damping. Preserve the command lines and JSON reports.",
        "Build a one-minute listening test that alternates dry, early-field, late-field, and complete versions without revealing their order to a listener.",
    ),
    "What Is Reverb? (and why verbx sounds different)": (
        "Measure the direct arrival and first five reflections in a supplied or self-recorded impulse response; sketch the likely boundary geometry without claiming more precision than the evidence permits.",
        "Compare a feedback comb, an allpass diffuser, and an FDN using the same impulse. Relate audible density to the corresponding signal-flow diagram.",
        "Plot an energy-decay curve for two rooms or two presets, fit EDT and T30 where valid, and explain why the estimates disagree or agree.",
        "Make a short percussion study in which early reflections establish one apparent room while the late field establishes another. State the intended perceptual contradiction.",
        "Choose one pole-zero plot from the chapter and predict its magnitude response before viewing the accompanying response graph.",
    ),
    "verbx Reference": (
        "Use a labeled impulse to verify every route in one multichannel or matrix-convolution configuration; document channel order, arrival time, polarity, and level.",
        "Capture three candidate impulse responses from one space using unchanged geometry, then compare their noise floors and decay estimates before choosing a production version.",
        "Create one convolution render and one algorithmic approximation of the same room role. Identify what the approximation preserves and what it changes.",
        "Design a safe long-tail render with explicit duration, container, limiter, and report settings. Explain each guardrail.",
        "Create a minimal repeatable render manifest containing input hashes, command line, version, output format, and analysis sidecar location.",
    ),
    "Immersive Reverb, Surround Sound, and Dolby Atmos": (
        "Build a source-bound early field, an environment-bound late field, and one gesture-bound effect from the same dry source. Assign a distinct spatial role to each.",
        "Verify a 7.1.2 or 7.1.4 handoff with spoken labels or impulses before auditioning music; record the exact channel map used by the receiving DAW.",
        "Render an FOA intermediate, rotate it, decode it, and compare that result with a fixed stereo or surround print at matched loudness.",
        "Prepare separate dry, early, and late stems for a WFS renderer. State which spatial decision belongs to verbx and which belongs to the calibrated array system.",
        "Audit one immersive mix on its target layout, binaural render, stereo fold-down, and one reduced layout. List the hierarchy that survives each translation.",
    ),
    "System Orientation Through Block Diagrams": (
        "Trace one CLI render from input file to output file and annotate every stage at which gain, channel count, or time alignment can change.",
        "Redraw one block diagram for a mono source and one for a multichannel source, naming the data contract at each boundary.",
        "Choose one failure mode from the diagrams and design a minimal diagnostic signal that isolates it.",
        "Compare offline render and realtime paths for one preset. Identify the state that must be shared and the state that must remain host or device specific.",
        "Explain one diagram to a collaborator using only source, early field, late field, safety, and output as your vocabulary.",
    ),
    "CLI Reference": (
        "Write a command that creates a 100-percent-wet stereo return with an analysis JSON, then explain why each output option is explicit.",
        "Use `--dry-run` or the equivalent validation path on one expensive render and record which estimated resources or warnings change after one parameter edit.",
        "Create a preset, render it twice with a fixed seed, and verify reproducibility using hashes and analysis reports.",
        "Deliberately submit one invalid layout or incompatible option combination, then rewrite the command so it fails fast for the right reason.",
        "Make a concise shell script that renders three controlled variants and preserves their reports beside the audio.",
    ),
    "verbx Extreme Workflow Cookbook (with 100 Recipes)": (
        "Select five recipes that target the same source class and arrange them from most transparent to most transformed; explain the progression.",
        "Take one recipe and create a conservative, moderate, and extreme version while changing only the parameters essential to its stated effect.",
        "Convert one recipe into a reusable preset and add a machine-readable analysis report to the output directory.",
        "Test one recipe on speech, percussion, and sustained harmony. Identify which source property determines whether it succeeds.",
        "Invent a recipe title and provide a runnable command, a listening goal, a safety check, and one likely failure mode.",
    ),
    "VERBX AUv3/VST3 Plug-in Handbook": (
        "Recreate one CLI reverb design in the plug-in, then print the return and analyze it with the CLI. Document intentional differences.",
        "Automate RT60 coarse and fine controls separately over a phrase. Describe the musical reason to use each control's range.",
        "Use the Expert page to make a stable wide late field without changing the dry source's localization; compare stereo and mono fold-downs.",
        "Use the spectrum analyzer to identify a spectral buildup, then correct it with one reverb parameter before reaching for unrelated channel EQ.",
        "Create a DAW preset with a reverse or freeze gesture and record the host automation so another session can reproduce the form.",
    ),
    "IR Synthesis – A Dual-Layer Reference": (
        "Synthesize three IRs with equal nominal RT60 but different early-reflection structures. Compare direct-to-reverberant ratio and perceived distance.",
        "Morph two IRs while retaining a fixed source and output level; identify the moment at which the result stops reading as one coherent room.",
        "Create a stereo or multichannel IR matrix and verify every route with isolated impulses before processing musical audio.",
        "Analyze a synthesized IR for decay, spectrum, and channel correlation, then revise only the parameter most directly related to the weakest metric.",
        "Design an IR library naming convention that preserves geometry, source/receiver roles, normalization, sample rate, and licensing evidence.",
    ),
    "AI Research and Data Augmentation": (
        "Create a small train, validation, and test augmentation plan that prevents room-identity leakage across splits.",
        "Render a controlled augmentation set in which RT60 changes while source, seed, level, and channel layout remain fixed. Produce a CSV or JSON comparison.",
        "Choose one task, such as ASR or source separation, and state which acoustic variations should be invariant versus predictive for the model.",
        "Run quality checks on a small batch and reject at least one intentionally implausible output using an explicit metric rule.",
        "Write a compact dataset card that records provenance, render parameters, split policy, and known limitations of the augmented audio.",
    ),
    "Illustrated Guide": (
        "Choose five figures from different chapters and write one prediction before reading each caption, then compare prediction with the diagram's actual claim.",
        "Redraw one signal-flow diagram with different parameter values while preserving its causal structure.",
        "Use one sonogram pair to identify a time-scale and frequency-scale invariant, then explain why equal axes matter for the comparison.",
        "Pair one pole-zero plot with its magnitude response and identify a feature that cannot be inferred safely from one plot alone.",
        "Make one original labeled figure from a verbx render, including units, a caption, source data, and a one-paragraph interpretation.",
    ),
    "Finite-Element Modeling for Reverb and Resonant Systems": (
        "Model a simple rectangular cavity with stated boundary conditions and compare its lowest predicted modes with a measured or synthesized response.",
        "Change one material or boundary parameter at a time and relate the simulated change to absorption, damping, or modal spacing.",
        "Estimate the computational cost of refining a mesh in time and space, then explain why a faster method may be preferable for late reverberation.",
        "Use a finite-element result only for the early or low-frequency part of a hybrid design, then choose a statistical late-field model for the remainder.",
        "Write a validation plan that distinguishes agreement with one microphone position from agreement across a meaningful listening region.",
    ),
    "Microtonal Workflows, Scala Import, and Scale-Tuned Reverberation": (
        "Import one Scala scale and create two scale-tuned reverb designs with different root frequencies. Compare their interaction with the same sustained chord.",
        "Use an impulse, a chromatic sweep, a harmonic sound, and a musical excerpt to test whether a tuned resonance design generalizes across sources.",
        "Compare a five-degree, twelve-degree, and thirty-one-degree scale with matched target budgets. Describe changes in spectral density and beating.",
        "Compose a short phrase in which reverb tuning changes at a harmonic boundary without changing the dry instrument's tuning.",
        "Document the Scala file, root mapping, frequency range, target count, and any transposition used so another listener can reproduce the scale-tuned field.",
    ),
}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _build_markdown(author: str) -> str:
    lines: list[str] = [
        "# verbx User Guide",
        "",
        f"_{author}, {BOOK_EDITION_DATE}_",
        "",
        "_Generated by `python3 scripts_generate_docs_pdf.py`._",
        "",
        "This manual consolidates the README plus the user-facing guides,",
        "reference material, and practical tips shipped in `docs/`.",
        "",
        "## Included Sources",
        "",
    ]
    lines.extend(f"- `{_rel(path)}`" for path in USERGUIDE_INCLUDED_SOURCES)
    lines.extend(["", "---", ""])

    for index, source in enumerate(USERGUIDE_SOURCES):
        if index > 0:
            lines.extend(["", "\\newpage", "", f"<!-- {_rel(source)} -->", ""])
        lines.append(_markdown_for_userguide(source).rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _markdown_for_userguide(source: Path) -> str:
    """Return source Markdown adjusted for its new home in docs/USERGUIDE.md."""

    markdown = source.read_text(encoding="utf-8")
    if source == ROOT / "docs" / "CLI_REFERENCE.md":
        # Rich/Typer pads captured help to terminal width; that spacing has no
        # semantic value inside fenced blocks and makes the generated book dirty.
        markdown = "\n".join(line.rstrip() for line in markdown.splitlines()) + "\n"
    if source == ROOT / "docs" / "GLOSSARY.md":
        # Keep the glossary appendix numbered, but present its A-Z dividers as
        # unnumbered headings that never create table-of-contents entries.
        def unnumbered_glossary_letter(match: re.Match[str]) -> str:
            letter = match.group(1)
            target = letter.lower()
            return (
                "```{=latex}\n"
                rf"\hypertarget{{{target}}}{{}}"
                "\n"
                rf"\section*{{{letter}}}"
                "\n```"
            )

        markdown = re.sub(
            r"(?m)^## ([A-Z])$",
            unnumbered_glossary_letter,
            markdown,
        )
    if source == ROOT / "README.md":
        markdown = markdown.replace('src="docs/assets/', 'src="assets/')
        markdown = markdown.replace("](docs/assets/", "](assets/")
        anchor = "Homebrew maintainer details: [`docs/HOMEBREW.md`](docs/HOMEBREW.md)"
        replacement = (
            "Homebrew maintainer and release details are consolidated below.\n\n"
            f"{_homebrew_chapter_one_supplement()}"
        )
        if anchor not in markdown:
            raise ValueError("README Homebrew insertion anchor is missing")
        markdown = markdown.replace(anchor, replacement, 1)
        history_marker = "### Why verbx Sounds Different"
        if history_marker not in markdown:
            raise ValueError("README artificial-reverberation history marker is missing")
        markdown = markdown.replace(
            history_marker,
            f"{_chapter_two_history_supplement()}\n\n{history_marker}",
            1,
        )
        portfolio_marker = "### Musical Examples"
        if portfolio_marker not in markdown:
            raise ValueError("README open-source portfolio insertion marker is missing")
        markdown = markdown.replace(
            portfolio_marker,
            f"{_chapter_two_portfolio_supplement()}\n\n{portfolio_marker}",
            1,
        )
        benchmark_marker = "## DSP Architecture"
        if benchmark_marker not in markdown:
            raise ValueError("README benchmark insertion marker is missing")
        markdown = markdown.replace(
            benchmark_marker,
            f"{_reference_benchmark_supplement()}\n\n{benchmark_marker}",
            1,
        )
    return markdown


def _homebrew_chapter_one_supplement() -> str:
    """Embed the nonduplicated Homebrew material under Chapter 1."""

    markdown = HOMEBREW_SOURCE.read_text(encoding="utf-8")
    section_start = markdown.find("## Maintainer Workflow")
    if section_start < 0:
        raise ValueError("Homebrew maintainer section is missing")

    maintenance = _demote_markdown_headings(markdown[section_start:].rstrip())
    return (
        "## Homebrew Distribution and Release Maintenance\n\n"
        "The installation commands above cover the user path. The tap's release, "
        "automation, and compatibility details follow for maintainers and packagers. "
        "The published tap formula lives at `Formula/verbx.rb` in "
        "`TheColby/homebrew-verbx`.\n\n"
        f"{maintenance}"
    )


def _chapter_two_history_supplement() -> str:
    """Embed the illustrated artificial-reverberation history in Chapter 2."""

    markdown = ARTIFICIAL_REVERB_HISTORY_SOURCE.read_text(encoding="utf-8").strip()
    expected_heading = (
        "### A History of Artificial Reverberation: Architectural, Mechanical, "
        "Electrical, Electromechanical, and Digital"
    )
    if not markdown.startswith(expected_heading):
        raise ValueError("Artificial-reverberation history heading is missing")
    return markdown


def _chapter_two_portfolio_supplement() -> str:
    """Fold the rights-cleared portfolio into the early reverb chapter."""

    markdown = OPEN_SOURCE_PORTFOLIO_SOURCE.read_text(encoding="utf-8").strip()
    markdown = _demote_markdown_headings(_demote_markdown_headings(markdown))
    markdown = re.sub(r"\bFigure 17-\d+\b", "the following figure", markdown)
    markdown = markdown.replace("\nthe following figure", "\nThe following figure")
    return markdown


def _reference_benchmark_supplement() -> str:
    """Fold the former benchmark chapter into the main reference chapter."""

    markdown = BENCHMARK_SOURCE.read_text(encoding="utf-8").strip()
    return _demote_markdown_headings(_demote_markdown_headings(markdown))


def _demote_markdown_headings(markdown: str) -> str:
    """Demote prose headings by one level without touching fenced code."""

    output: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
        elif fence is None and re.match(r"^#{1,5}\s", line):
            line = f"#{line}"
        output.append(line)
    return "\n".join(output)


def _write_markdown(path: Path, author: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_build_markdown(author), encoding="utf-8")


def _pandoc_base_command(markdown_path: Path, author: str) -> list[str]:
    return [
        "pandoc",
        str(markdown_path),
        "--from=gfm+tex_math_dollars+raw_attribute+smart",
        "--toc",
        "--list-of-figures",
        "--list-of-tables",
        "--number-sections",
        "--standalone",
        "--top-level-division=chapter",
        "--resource-path=.:docs:examples",
        "--metadata=title:verbx User Guide",
        "--metadata=subtitle:Reverb, Spatial Audio, Dereverberation, Plug-in Design, "
        "and Educational Exercises",
        f"--metadata=author:{author}",
        f"--metadata=date:{BOOK_EDITION_DATE}",
        "--include-in-header",
        str(PDF_PREAMBLE),
        "--include-in-header",
        str(CARD_ILLUSTRATIONS),
        "--lua-filter",
        str(TABLE_CAPTION_FILTER),
        "--lua-filter",
        str(DIRECTIONAL_QUOTES_FILTER),
        "-V",
        "documentclass=book",
        "-V",
        "classoption=openany",
        "-V",
        "papersize=letter",
        "-V",
        "toc-depth=2",
        "-V",
        "geometry:margin=1in",
        "-V",
        "colorlinks=true",
        "-V",
        "urlcolor=blue",
        "-V",
        "mainfont=texgyreschola-regular.otf",
        "-V",
        f"mainfontoptions:Path={TEX_GYRE_FONT_DIR}",
        "-V",
        "mainfontoptions:BoldFont=texgyreschola-bold.otf",
        "-V",
        "mainfontoptions:ItalicFont=texgyreschola-italic.otf",
        "-V",
        "mainfontoptions:BoldItalicFont=texgyreschola-bolditalic.otf",
        "-V",
        "mainfontoptions:Ligatures=Common",
        "-V",
        "mathfont=texgyreschola-math.otf",
        "-V",
        f"mathfontoptions:Path={TEX_GYRE_MATH_FONT_DIR}",
        "-V",
        "monofont=Menlo",
        "-V",
        "monofontoptions:Ligatures=NoCommon",
    ]


def _rewrite_longtable_specs(latex_path: Path) -> None:
    text = latex_path.read_text(encoding="utf-8")
    replacements = {
        2: (
            r"\begin{longtable}[]{@{}>{\RaggedRight\arraybackslash\hspace{0pt}}p{0.26\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.62\linewidth}@{}}"
        ),
        3: (
            r"\begin{longtable}[]{@{}>{\RaggedRight\arraybackslash\hspace{0pt}}p{0.22\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.34\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.30\linewidth}@{}}"
        ),
        4: (
            r"\begin{longtable}[]{@{}>{\RaggedRight\arraybackslash\hspace{0pt}}p{0.15\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.23\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.22\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.30\linewidth}@{}}"
        ),
        5: (
            r"\begin{longtable}[]{@{}>{\RaggedRight\arraybackslash\hspace{0pt}}p{0.13\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.22\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.14\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.10\linewidth}"
            r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.27\linewidth}@{}}"
        ),
    }

    def bounded_columns(match: re.Match[str]) -> str:
        return replacements[len(match.group("columns"))]

    text = re.sub(
        r"\\begin\{longtable\}\[\]\{@\{\}(?P<columns>[lcr]{2,5})@\{\}\}",
        bounded_columns,
        text,
    )

    layout_spec = (
        r"\begin{longtable}[]{@{}>{\RaggedRight\arraybackslash\hspace{0pt}}p{0.16\linewidth}"
        r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.14\linewidth}"
        r">{\RaggedRight\arraybackslash\hspace{0pt}}p{0.60\linewidth}@{}}"
    )
    longtable_pattern = re.compile(
        r"(\\begin\{longtable\}.*?\\end\{longtable\})", re.DOTALL
    )

    def semantic_columns(match: re.Match[str]) -> str:
        block = match.group(1)
        if all(label in block for label in ("Layout", "Channels", "Use case")):
            return block.replace(replacements[3], layout_spec, 1)
        return block

    text = longtable_pattern.sub(semantic_columns, text)

    text = _rewrite_longtable_value_breaks(text)
    text = _rewrite_highlighting_token_breaks(text)
    text = _rewrite_texttt_breaks(text)
    text = _rewrite_figure_paths(text)
    latex_path.write_text(text, encoding="utf-8")


def _rewrite_figure_paths(text: str) -> str:
    """Resolve guide assets from the temporary LaTeX build directory."""

    figure_root = (ROOT / "docs" / "assets" / "userguide_figures").as_posix()
    asset_root = (ROOT / "docs" / "assets").as_posix()
    text = text.replace(
        r"{assets/userguide_figures/",
        rf"{{{figure_root}/",
    )
    return text.replace(r"{assets/", rf"{{{asset_root}/")


def _rewrite_longtable_value_breaks(text: str) -> str:
    """Add breakpoints to compact slash/comma lists inside tables only."""

    longtable_pattern = re.compile(r"(\\begin\{longtable\}.*?\\end\{longtable\})", re.DOTALL)

    def rewrite_block(match: re.Match[str]) -> str:
        block = match.group(1)
        protected_urls: list[str] = []

        def protect_url(url_match: re.Match[str]) -> str:
            protected_urls.append(url_match.group(0))
            return f"@@VERBX_HREF_URL_{len(protected_urls) - 1}@@"

        block = re.sub(r"\\href\{[^{}]*\}", protect_url, block)
        block = re.sub(r"(?<=[A-Za-z0-9_.\\])/(?=[A-Za-z0-9_.\\])", r"/\\allowbreak{}", block)
        block = re.sub(r"(?<=[A-Za-z0-9_.\\]),(?=[A-Za-z0-9_.\\])", r",\\allowbreak{}", block)
        block = re.sub(r"(?<=[A-Za-z0-9_.\\])=(?=[A-Za-z0-9_.\\])", r"=\\allowbreak{}", block)

        for index, url in enumerate(protected_urls):
            block = block.replace(f"@@VERBX_HREF_URL_{index}@@", url)
        return block

    return longtable_pattern.sub(rewrite_block, text)


def _rewrite_texttt_breaks(text: str) -> str:
    return _rewrite_macro_content_breaks(text, r"\texttt{")


def _rewrite_highlighting_token_breaks(text: str) -> str:
    for marker in (r"\StringTok{", r"\NormalTok{", r"\AttributeTok{"):
        text = _rewrite_macro_content_breaks(text, marker)
    return text


def _rewrite_macro_content_breaks(text: str, marker: str) -> str:
    parts: list[str] = []
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            parts.append(text[start:])
            break
        parts.append(text[start : idx + len(marker)])
        cursor = idx + len(marker)
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        content = text[content_start : cursor - 1]
        content = _insert_inline_breakpoints(content)
        parts.append(content)
        parts.append("}")
        start = cursor
    return "".join(parts)


def _insert_inline_breakpoints(content: str) -> str:
    content = content.replace(r"\_", r"\_\allowbreak{}")
    content = content.replace(r"\/", r"\/\allowbreak{}")
    content = re.sub(r"(?<!\\)/", r"/\\allowbreak{}", content)
    content = re.sub(r"(?<!\\)\.", r".\\allowbreak{}", content)
    content = content.replace(":", r":\allowbreak{}")
    content = content.replace("-", r"-\allowbreak{}")
    content = content.replace(",", r",\allowbreak{}")
    content = content.replace("=", r"=\allowbreak{}")
    return content


def _markdown_with_pdf_targets(markdown: str) -> str:
    """Make GitHub-style HTML anchors available to Pandoc's LaTeX backend."""

    def replace_anchor_run(match: re.Match[str]) -> str:
        anchor_run = match.group(0)
        anchor_ids = re.findall(r'id="([^"]+)"', anchor_run)
        label_lines = "\n".join(
            rf"\phantomsection\label{{{anchor_id}}}" for anchor_id in anchor_ids
        )
        return anchor_run + "\n\n```{=latex}\n" + label_lines + "\n```"

    markdown = _remove_pdf_exclusions(markdown)
    markdown = _remove_generated_pdf_preamble(markdown)
    markdown = _add_book_parts(markdown)
    markdown = _promote_reverb_primer_to_chapter(markdown)
    markdown = _add_chapter_exercises(markdown)
    # Keep each epigraph flush with the chapter it introduces.
    markdown = _add_chapter_epigraphs(markdown)
    markdown = _italicize_musical_titles(markdown)
    markdown = re.sub(r'(?:<a\s+id="[^"]+"></a>)+', replace_anchor_run, markdown)
    markdown = _replace_mermaid_with_static_assets(markdown)
    markdown = _ensure_image_captions(markdown)
    markdown = _keep_code_leads_with_examples(markdown)
    markdown = _compact_illustrated_guide(markdown)
    markdown = _illustrate_operational_cards(markdown)
    markdown = _convert_figure_captions(markdown)
    markdown = _strip_plugin_heading_numbers(markdown)
    markdown = markdown.replace(
        "# Important Musical Pieces",
        "```{=latex}\n\\appendix\n```\n\n# Important Musical Pieces",
        1,
    )
    markdown = markdown.replace(
        "# Research Papers and References",
        "```{=latex}\n\\clearpage\n```\n\n# Research Papers and References",
        1,
    )
    markdown = _add_pdf_index(markdown)
    _validate_figure_sequence(markdown)
    markdown = re.sub(
        r"^\\newpage$",
        lambda _: "```{=latex}\n\\newpage\n```",
        markdown,
        flags=re.MULTILINE,
    )
    _validate_fenced_blocks(markdown)
    return markdown


def _remove_pdf_exclusions(markdown: str) -> str:
    """Remove source passages explicitly marked as inappropriate inside the PDF."""

    start_marker = "<!-- verbx-pdf-exclude-start -->"
    end_marker = "<!-- verbx-pdf-exclude-end -->"
    if markdown.count(start_marker) != markdown.count(end_marker):
        raise ValueError("PDF-exclusion markers are unbalanced")

    pattern = re.compile(
        rf"(?ms)^{re.escape(start_marker)}\n.*?^{re.escape(end_marker)}\n?"
    )
    return pattern.sub("", markdown)


def _replace_mermaid_with_static_assets(markdown: str) -> str:
    """Use checked-in static equivalents for Mermaid diagrams in the PDF."""

    pattern = re.compile(
        r"(?ms)^```mermaid\n(?P<body>.*?)\n```\n\n"
        r"(?P<caption>\*\*Figure:\s+(?P<title>[^\n]+?)\.\*\*)$"
    )

    def replace(match: re.Match[str]) -> str:
        directive = re.search(
            r"(?m)^%% verbx-static:\s+(?P<path>\S+)\s*$",
            match.group("body"),
        )
        if directive is None:
            raise ValueError(f"Mermaid figure lacks verbx-static directive: {match.group('title')}")
        asset = directive.group("path")
        if not (ROOT / asset).is_file():
            raise FileNotFoundError(f"Mermaid static asset does not exist: {asset}")
        return f"![{match.group('title')}]({asset})\n\n{match.group('caption')}"

    markdown = pattern.sub(replace, markdown)
    if "```mermaid" in markdown:
        raise ValueError("Found an unconverted Mermaid block in PDF Markdown")
    return markdown


def _keep_code_leads_with_examples(markdown: str) -> str:
    """Keep short prose introductions with the command examples they introduce."""

    lines = markdown.splitlines()
    insertions: dict[int, list[str]] = {}
    active_fence = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        if active_fence:
            active_fence = False
            continue
        active_fence = True
        if stripped == "```{=latex}" or index < 2 or lines[index - 1].strip():
            continue

        paragraph_end = index - 2
        paragraph_start = paragraph_end
        while paragraph_start > 0 and lines[paragraph_start - 1].strip():
            paragraph_start -= 1
        paragraph = lines[paragraph_start : paragraph_end + 1]
        if not paragraph or len(paragraph) > 4:
            continue
        disallowed_prefixes = (
            "#",
            "-",
            "* ",
            "+ ",
            ">",
            "|",
            "<",
            "\\",
            "![",
            "```",
            "**Figure",
            "**Block diagram",
        )
        if any(
            not value.strip()
            or value.startswith("    ")
            or value.lstrip().startswith(disallowed_prefixes)
            for value in paragraph
        ):
            continue
        if sum(len(value) for value in paragraph) > 420:
            continue

        insertions[paragraph_start] = [
            "```{=latex}",
            "\\Needspace{7\\baselineskip}",
            "```",
            "",
        ]
        insertions[index] = [
            "```{=latex}",
            "\\nopagebreak[4]",
            "```",
            "",
        ]

    output: list[str] = []
    for index, line in enumerate(lines):
        output.extend(insertions.get(index, ()))
        output.append(line)
    return "\n".join(output)


def _validate_figure_sequence(markdown: str) -> None:
    """Require one prose lead before every figure caption in document order."""

    raw_events = re.findall(
        r"\\verbx(?P<event>FigureLead|FigureCaption|PlateCaption)\{",
        markdown,
    )
    events = ["Lead" if event == "FigureLead" else "Caption" for event in raw_events]
    if not events or len(events) % 2:
        raise ValueError("Every figure must have one prose lead and one caption")
    malformed = [
        position // 2 + 1
        for position in range(0, len(events), 2)
        if events[position : position + 2] != ["Lead", "Caption"]
    ]
    if malformed:
        preview = ", ".join(str(number) for number in malformed[:10])
        raise ValueError(f"Figure lead/caption order is invalid at figure(s): {preview}")


def _validate_fenced_blocks(markdown: str) -> None:
    """Reject fence corruption before Pandoc can typeset it as visible text."""

    valid_fence = re.compile(r"```(?:\{=latex\}|[A-Za-z][A-Za-z0-9_+.-]*)?")
    active_fence: tuple[int, str] | None = None
    for line_number, line in enumerate(markdown.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        if valid_fence.fullmatch(stripped) is None:
            raise ValueError(f"Malformed Markdown fence at line {line_number}: {stripped!r}")
        if active_fence is None:
            active_fence = (line_number, stripped)
        elif stripped == "```":
            active_fence = None

    if active_fence is not None:
        line_number, fence = active_fence
        raise ValueError(f"Unclosed Markdown fence at line {line_number}: {fence!r}")


def _remove_generated_pdf_preamble(markdown: str) -> str:
    """Drop the consolidated-source manifest; the book has real front matter."""

    readme_start = markdown.find("\n# verbx\n")
    return markdown[readme_start + 1 :] if readme_start != -1 else markdown


def _add_book_parts(markdown: str) -> str:
    """Insert the three editorial divisions used by the PDF book."""

    divisions = (
        ("# verbx\n", "User Manual and Workflows"),
        ("# VERBX AUv3/VST3 Plug-in Handbook\n", "Plug-in Architecture and Operational Cards"),
        (
            "# IR Synthesis – A Dual-Layer Reference\n",
            "DSP, Impulse Responses, and Technical Reference",
        ),
    )
    for heading, part_title in divisions:
        marker = f"```{{=latex}}\n\\part{{{part_title}}}\n```\n\n"
        markdown = markdown.replace(heading, marker + heading, 1)
    return markdown


def _promote_reverb_primer_to_chapter(markdown: str) -> str:
    """Give the reverb primer a chapter boundary without changing README hierarchy."""

    primer_heading = "## What Is Reverb? (and why verbx sounds different)"
    reference_heading = "## Core Concepts"
    start = markdown.find(primer_heading)
    end = markdown.find(reference_heading, start)
    if start == -1 or end == -1:
        raise ValueError("Cannot locate the reverb-primer chapter boundaries")

    primer = markdown[start:end]
    primer = primer.replace(primer_heading, primer_heading[1:], 1)
    primer = re.sub(r"(?m)^### ", "## ", primer)
    primer = re.sub(r"(?m)^#### ", "### ", primer)
    reference = "# verbx Reference\n\n" + markdown[end:]
    return markdown[:start] + primer + reference


def _add_chapter_exercises(markdown: str) -> str:
    """Add five tailored exercises to the end of every principal book chapter."""

    output: list[str] = []
    fence: str | None = None
    active_heading: str | None = None
    has_exercises = False

    def append_exercises() -> None:
        if active_heading not in CHAPTER_EXERCISES or has_exercises:
            return
        output.extend(["", "## Suggested Exercises", ""])
        output.extend(f"{index}. {exercise}" for index, exercise in enumerate(
            CHAPTER_EXERCISES[active_heading], start=1
        ))
        output.append("")

    for line in markdown.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None

        if fence is None and re.match(r"^# [^#]", line):
            append_exercises()
            active_heading = line[2:].strip()
            has_exercises = False
        elif fence is None and line.strip() == "## Suggested Exercises":
            has_exercises = True
        output.append(line)

    append_exercises()
    return "\n".join(output)


def _epigraph_tex(value: str) -> str:
    """Escape short epigraph text for a two-argument LaTeX macro."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    def escape(text: str) -> str:
        return "".join(replacements.get(character, character) for character in text)

    parts = re.split(r"(\*[^*]+\*)", value)
    return "".join(
        rf"\emph{{{escape(part[1:-1])}}}"
        if part.startswith("*") and part.endswith("*")
        else escape(part)
        for part in parts
    )


def _add_chapter_epigraphs(markdown: str) -> str:
    """Place a dedicated quotation leaf before every H1 outside code fences."""

    output: list[str] = []
    fence: str | None = None
    chapter_index = 0
    for line in markdown.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
        if fence is None and re.match(r"^# [^#]", line):
            quote, attribution, date = CHAPTER_EPIGRAPHS[
                chapter_index % len(CHAPTER_EPIGRAPHS)
            ]
            output.extend(
                [
                    "```{=latex}",
                    rf"\verbxChapterEpigraph{{{_epigraph_tex(quote)}}}"
                    rf"{{{_epigraph_tex(attribution)}}}{{{_epigraph_tex(date)}}}",
                    "```",
                    "",
                ]
            )
            chapter_index += 1
        output.append(line)
    return "\n".join(output)


def _italicize_musical_titles(markdown: str) -> str:
    """Enforce book-style italics for work titles in Appendix A."""

    start = markdown.find("# Important Musical Pieces")
    end = markdown.find("# Educational Exercises and Project Assignments", start)
    if start == -1:
        return markdown
    if end == -1:
        end = len(markdown)
    appendix = markdown[start:end]
    lead_pattern = re.compile(
        r"(?m)^\*\*(?P<creator>[^,*\n]+(?:,\s+[^,*\n]+)*),\s+"
        r"(?P<title>(?!\*)[^\n]+?)\s+\((?P<date>[^)]+)\)\.\*\*"
    )

    def italicize(match: re.Match[str]) -> str:
        return f"**{match.group('creator')}, *{match.group('title')}* ({match.group('date')}).**"

    appendix = lead_pattern.sub(italicize, appendix)
    return markdown[:start] + appendix + markdown[end:]


def _ensure_image_captions(markdown: str) -> str:
    """Add a descriptive caption marker after every otherwise unlabeled image."""

    lines = markdown.splitlines()
    output: list[str] = []
    image_pattern = re.compile(r"^!\[(?P<alt>[^]]*)\]\([^)]+\)\s*$")
    caption_pattern = re.compile(r"^\*\*(?:Figure|Block diagram)")
    for index, line in enumerate(lines):
        output.append(line)
        match = image_pattern.match(line)
        if not match:
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines) and caption_pattern.match(lines[cursor]):
            continue
        title = (
            re.sub(
                r"^(?:Figure|Block diagram)\s+\d+\s*:\s*",
                "",
                match.group("alt"),
                flags=re.IGNORECASE,
            )
            .strip()
            .rstrip(".")
            or "Illustration"
        )
        output.extend(("", f"**Figure: {title}.**"))
    return "\n".join(output)


def _convert_figure_captions(markdown: str) -> str:
    """Place a numbered prose reference before each standard image and its caption below."""

    pattern = re.compile(
        r"(?m)^(?P<image>!\[[^\n]*\]\((?P<path>[^\n)]+)\))\n\n"
        r"\*\*(?:Figure(?:\s+\d+)?|Block diagram\s+\d+)[.:]\s*"
        r"(?P<title>.+?)\.?\*\*(?P<rest>[^\n]*)$"
        r"(?:\n\n(?P<credit>\*Source and license:\*[^\n]+))?"
    )

    def convert(match: re.Match[str]) -> str:
        title = _latex_text_with_inline_math(match.group("title").rstrip("."))
        rest = match.group("rest").strip()
        credit = match.group("credit") or ""
        credit_block = f"\n\n{credit}" if credit else ""
        lead = f"```{{=latex}}\n\\verbxFigureLead{{{title}}}\n```"
        caption = f"```{{=latex}}\n\\verbxFigureCaption{{{title}}}\n```"
        path = match.group("path")
        if "open_source_portfolio/" in path:
            full_page = path.endswith(
                (
                    "01_spem_in_alium_opening.png",
                    "01_spem_in_alium_tutti-22.png",
                )
            )
            page_open = "\\clearpage\n" if full_page else ""
            page_close = "\n\\clearpage" if full_page else ""
            max_height = "0.72\\textheight" if full_page else "0.57\\textheight"
            figure = (
                "```{=latex}\n"
                f"{page_open}\\begin{{samepage}}\n"
                f"\\verbxFigureLead{{{title}}}\n"
                "{\\centering\n"
                f"\\includegraphics[width=\\linewidth,height={max_height},keepaspectratio]"
                f"{{\\detokenize{{{path}}}}}\n"
                "\\par}\n"
                f"\\verbxPlateCaption{{{title}}}\n"
                "```"
            )
            if credit:
                figure += f"\n\n{credit}"
            figure += (
                "\n\n```{=latex}\n"
                f"\\end{{samepage}}{page_close}\n"
                "```"
            )
        elif path.startswith(("docs/assets/reverb_primer/", "assets/reverb_primer/")):
            asset = match.group("path")
            figure = (
                "```{=latex}\n"
                "\\begin{minipage}{\\linewidth}\n"
                f"\\verbxFigureLead{{{title}}}\n"
                "{\\centering\n"
                "\\includegraphics[width=\\linewidth,height=0.64\\textheight,keepaspectratio]"
                f"{{\\detokenize{{{asset}}}}}\n"
                "\\par}\n"
                f"\\verbxFigureCaption{{{title}}}\n"
                "\\end{minipage}\n"
                "```"
            )
        else:
            figure = (
                "```{=latex}\n\\begin{samepage}\n```\n\n"
                f"{match.group('image')}\n\n{caption}{credit_block}\n\n"
                "```{=latex}\n\\end{samepage}\n```"
            )
        converted = (
            figure
            if "reverb_primer/" in path or "open_source_portfolio/" in path
            else f"{lead}\n\n{figure}"
        )
        return converted + (f"\n\n{rest}" if rest else "")

    markdown, count = pattern.subn(convert, markdown)
    leftover = re.search(
        r"(?m)^\*\*(?:Figure(?:\s+\d+)?|Block diagram\s+\d+)[.:]",
        markdown,
    )
    if leftover:
        raise ValueError("Found a figure caption without an immediately preceding image")
    if count == 0:
        raise ValueError("No standard image figures were converted")
    return markdown


def _resolve_pdf_figure_asset(value: str) -> Path | None:
    if re.match(r"^(?:https?:|data:)", value) or any(char in value for char in ('"', "'")):
        return None
    path = Path(value)
    candidates = (path,) if path.is_absolute() else (ROOT / path, ROOT / "docs" / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _write_trimmed_pdf_figure(source: Path, destination: Path) -> bool:
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        return False
    with Image.open(source) as opened:
        rgba = opened.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, "white")
        flattened.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    corner = flattened.getpixel((0, 0))
    background = Image.new("RGB", flattened.size, corner)
    difference = ImageChops.difference(flattened, background).convert("L")
    content_bounds = difference.point(lambda value: 255 if value > 8 else 0).getbbox()
    if content_bounds is None:
        return False
    bottom = min(flattened.height, content_bounds[3] + 12)
    if flattened.height - bottom < 8:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    flattened.crop((0, 0, flattened.width, bottom)).save(destination, optimize=True)
    return True


def _trim_pdf_figure_assets(markdown: str, output_dir: Path) -> str:
    """Replace raster paths with copies that omit trailing background whitespace."""

    cache: dict[Path, Path | None] = {}

    def replace(match: re.Match[str]) -> str:
        source = _resolve_pdf_figure_asset(match.group("path").strip())
        if source is None:
            return match.group(0)
        if source not in cache:
            digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
            destination = output_dir / f"{source.stem}-{digest}.png"
            cache[source] = destination if _write_trimmed_pdf_figure(source, destination) else None
        replacement = cache[source]
        if replacement is None:
            return match.group(0)
        return f"{match.group('prefix')}{replacement.as_posix()}{match.group('suffix')}"

    patterns = (
        re.compile(r"(?P<prefix>!\[[^\n]*\]\()(?P<path>[^)\n]+)(?P<suffix>\))"),
        re.compile(r"(?P<prefix>\\detokenize\{)(?P<path>[^}\n]+)(?P<suffix>\})"),
    )
    for pattern in patterns:
        markdown = pattern.sub(replace, markdown)
    return markdown


def _illustrate_operational_cards(markdown: str) -> str:
    """Give every generated operational card a dedicated vector illustration."""

    start = markdown.find("## 19. Production Starting-Point Cards")
    end = markdown.find("## 32. Closing Checklist", start)
    if start == -1 or end == -1:
        return markdown

    handbook = markdown[start:end]
    handbook = re.sub(r"(?m)^###\s+(?:19|2[0-9]|3[01])\.\d+\s+.*?\n+", "", handbook)

    section_pattern = re.compile(r"(?m)^##\s+(?:19|2[0-9]|3[01])\.\s+(.+?)\s*$")

    def section_visual(match: re.Match[str]) -> str:
        title = _latex_text(match.group(1))
        return match.group(0) + "\n\n```{=latex}\n" + f"\\verbxCardSection{{{title}}}\n" + "```"

    handbook = section_pattern.sub(section_visual, handbook)

    card_pattern = re.compile(
        r"(?ms)^(?P<title>\*\*(?:Production|Automation|Quality|Validation|"
        r"Troubleshooting|Preset|Interaction|Audition|Asset|Release|Bus|"
        r"Signal-test|Triage) card.*?\*\*)\n"
        r"(?P<body>.*?)(?=^\\newpage\s*$)"
    )

    def illustrate_card(match: re.Match[str]) -> str:
        title_line = match.group("title")
        title = title_line.removeprefix("**").removesuffix("**")
        command = _card_visual_command(title)
        return (
            "```{=latex}\n\\clearpage\n```\n\n"
            + title_line
            + "\n"
            + match.group("body").rstrip()
            + "\n\n```{=latex}\n"
            + "\\vfill\n"
            + f"\\verbxFigureLead{{{_latex_text(title)} illustration}}\n"
            + command
            + f"\n\\verbxFigureCaption{{{_latex_text(title)} illustration}}"
            + "\n```\n\n"
        )

    handbook, card_count = card_pattern.subn(illustrate_card, handbook)
    if card_count != 588:
        raise ValueError(f"Expected 588 operational cards, found {card_count}")

    handbook = re.sub(
        r"(?m)^(#{2,3})\s+(?:\d+(?:\.\d+)*)\.?(?:\s+)(.+?)$",
        r"\1 \2",
        handbook,
    )
    return markdown[:start] + handbook + markdown[end:]


def _strip_plugin_heading_numbers(markdown: str) -> str:
    """Let the book class number structural headings without duplicated labels."""

    return re.sub(
        r"(?m)^(#{2,3})\s+(?:\d+(?:\.\d+)*)\.?(?:\s+)(.+?)$",
        r"\1 \2",
        markdown,
    )


def _card_visual_command(title: str) -> str:
    """Map a generated card title to one of the book's diagram grammars."""

    patterns: tuple[tuple[str, str], ...] = (
        (r"Production card: (.+) in (.+)", "verbxProductionVisual"),
        (r"Automation card: (.+): (.+)", "verbxAutomationVisual"),
        (r"Validation card: (.+): (.+)", "verbxValidationVisual"),
        (r"Preset card: (.+) / (.+)", "verbxPresetVisual"),
        (r"Audition card: (.+) on (.+)", "verbxMonitoringVisual"),
        (r"Asset card: (.+): (.+)", "verbxAssetVisual"),
        (r"Release card: (.+): (.+)", "verbxReleaseVisual"),
        (r"Bus card: (.+): (.+)", "verbxBusVisual"),
        (r"Signal-test card: (.+) with (.+)", "verbxSignalVisual"),
        (r"Triage card: (.+): (.+)", "verbxTriageVisual"),
    )
    for pattern, macro in patterns:
        match = re.fullmatch(pattern, title)
        if match:
            left, right = (_latex_text(value) for value in match.groups())
            return rf"\{macro}{{{left}}}{{{right}}}"

    interaction = re.fullmatch(r"Interaction card \d+: (.+) with (.+)", title)
    if interaction:
        left, right = interaction.groups()
        family = _interaction_visual_family(left, right)
        return rf"\verbxInteractionVisual{{{_latex_text(left)}}}{{{_latex_text(right)}}}{{{family}}}"

    quality = re.fullmatch(r"Quality card \d+: (\d+) Hz, (.+), (\d+) frames", title)
    if quality:
        rate, mode, block = (_latex_text(value) for value in quality.groups())
        return rf"\verbxQualityVisual{{{rate} Hz}}{{{mode}}}{{{block} frames}}"

    troubleshooting = re.fullmatch(r"Troubleshooting card \d+: (.+)", title)
    if troubleshooting:
        return rf"\verbxTroubleshootingVisual{{{_latex_text(troubleshooting.group(1))}}}"
    raise ValueError(f"No illustration mapping for card title: {title}")


def _interaction_visual_family(left: str, right: str) -> int:
    """Choose a diagram grammar that reflects the interaction being described."""

    pair = f"{left} {right}".lower()
    if any(term in pair for term in ("freeze", "reverse", "quality")):
        return 4  # State and quality: routing / decision diagram.
    if any(term in pair for term in ("wet", "dry", "duck", "mix", "gain")):
        return 3  # Level: faders and gain structure.
    if any(term in pair for term in ("damping", "shimmer", "bloom", "tone")):
        return 1  # Spectral: frequency-domain response.
    if any(term in pair for term in ("room size", "width", "diffusion", "spread")):
        return 2  # Spatial: field geometry and distribution.
    return 0  # Timing: delay and decay envelopes.


def _latex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "|": r"\textbar{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex_url(value: str) -> str:
    """Escape URL characters that have special meaning in LaTeX."""

    return value.replace("%", r"\%").replace("#", r"\#")


def _latex_text_with_inline_math(value: str) -> str:
    """Escape caption prose while preserving delimited LaTeX math."""

    segments = value.split("$")
    if len(segments) % 2 == 0:
        raise ValueError(f"Unbalanced inline-math delimiter in figure caption: {value!r}")
    def prose(segment: str) -> str:
        parts = segment.split("*")
        if len(parts) % 2 == 0:
            raise ValueError(f"Unbalanced emphasis delimiter in figure caption: {value!r}")
        return "".join(
            f"\\emph{{{_latex_text(part)}}}" if index % 2 else _latex_text(part)
            for index, part in enumerate(parts)
        )

    return "".join(
        f"${segment}$" if index % 2 else prose(segment)
        for index, segment in enumerate(segments)
    )


def _compact_illustrated_guide(markdown: str) -> str:
    """Typeset each illustrated-guide image beside its long description."""

    start = markdown.find("# Illustrated Guide")
    if start == -1:
        return markdown
    end = markdown.find("\n\\newpage\n", start)
    if end == -1:
        end = len(markdown)

    chapter = markdown[start:end]
    entry_pattern = re.compile(
        r"(?P<lead>The figure below introduces.*?)(?=\n\n)\n\n"
        r"(?P<image>!\[Figure\s+\d+:[^\n]+\]\([^\n]+\))\n\n"
        r"\*\*Figure\s+\d+[.:]\s+(?P<title>.+?)\.\*\*\s*"
        r"(?P<description>Read (?:the figure|each plan).*?)"
        r"(?=\n\n(?:The figure below|##\s)|\Z)",
        re.DOTALL,
    )

    def compact_entry(match: re.Match[str]) -> str:
        title = match.group("title")
        latex_title = _latex_text_with_inline_math(title)
        if title == "Loudspeaker Layouts: Plan and Elevation":
            return (
                f"{match.group('lead').strip()}\n\n"
                "```{=latex}\n"
                f"\\verbxFigureLead{{{latex_title}}}\n"
                "\\par\\medskip\\noindent\n"
                "\\begin{minipage}[t]{\\linewidth}\\vspace{0pt}\n"
                "```\n\n"
                f"{match.group('image')}\n\n"
                "```{=latex}\n"
                f"\\verbxFigureCaption{{{latex_title}}}\n"
                "\\end{minipage}\\par\\medskip\n"
                "```\n\n"
                f"{match.group('description')}"
            )
        return (
            f"{match.group('lead').strip()}\n\n"
            "```{=latex}\n"
            f"\\verbxFigureLead{{{latex_title}}}\n"
            "\\par\\medskip\\noindent\n"
            "\\begin{minipage}[t]{0.43\\textwidth}\\vspace{0pt}\n"
            "```\n\n"
            f"{match.group('image')}\n\n"
            "```{=latex}\n"
            f"\\verbxFigureCaption{{{latex_title}}}\n"
            "```\n\n"
            "```{=latex}\n"
            "\\end{minipage}\\hfill\n"
            "\\begin{minipage}[t]{0.54\\textwidth}\\vspace{0pt}\n"
            "```\n\n"
            f"{match.group('description')}\n\n"
            "```{=latex}\n"
            "\\end{minipage}\\par\\medskip\n"
            "```"
        )

    compacted, count = entry_pattern.subn(compact_entry, chapter)
    if count != 100:
        raise ValueError(f"Expected 100 illustrated-guide entries, found {count}")
    return markdown[:start] + compacted + markdown[end:]


def _add_pdf_index(markdown: str) -> str:
    """Build a flat textbook index from structure, controls, cards, and citations."""

    markdown = _index_markdown_headings(markdown)
    markdown = _index_cli_terms(markdown)
    markdown = _index_operational_cards(markdown)
    markdown = _index_bibliography(markdown)
    markdown = _index_glossary_terms(markdown)
    markdown = _typeset_research_references(markdown)

    figure_pattern = re.compile(
        r"(?m)^(?P<command>\\verbx(?:Figure|Plate)Caption\{"
        r"(?P<title>(?:[^{}]|\{[^{}]*\})*)\})$"
    )

    def index_figure(match: re.Match[str]) -> str:
        term = _figure_index_term(match.group("title"))
        return match.group("command") + f"\n\\index{{{term}}}"

    markdown = figure_pattern.sub(index_figure, markdown)

    appendix_start = markdown.find("# Important Musical Pieces")
    if appendix_start != -1:
        appendix = markdown[appendix_start:]
        work_pattern = re.compile(r"(?m)^\*\*(?P<term>.+?\([^)]+\))\.\*\*")

        def index_work(match: re.Match[str]) -> str:
            term = _musical_index_term(match.group("term"))
            marker = (
                "```{=latex}\n"
                f"\\index{{{term}}}\n"
                f"\\verbxWorkIndex{{{term}}}\n"
                "```\n\n"
            )
            return marker + match.group(0)

        appendix = work_pattern.sub(index_work, appendix)
        markdown = markdown[:appendix_start] + appendix
    return (
        markdown.rstrip()
        + "\n\n```{=latex}\n"
        + "\\backmatter\n"
        + "\\verbxAfterword\n"
        + "\\verbxAboutAuthor\n"
        + "\\verbxErrata\n"
        + "\\verbxPrintMusicalWorksIndex\n"
        + "\\clearpage\n"
        + "\\phantomsection\n"
        + "\\addcontentsline{toc}{chapter}{Index}\n"
        + "\\printindex\n"
        + "\\verbxColophon\n"
        + "\\verbxRepositoryQR\n"
        + "```\n"
    )


def _index_glossary_terms(markdown: str) -> str:
    """Add every glossary headword to the flat book index."""

    start = markdown.find("# Glossary")
    if start == -1:
        return markdown

    glossary = markdown[start:]
    entry_pattern = re.compile(
        r"(?m)^(?P<entry>\*\*(?P<term>[^*\n]+)\.\*\*\s+.+)$"
    )

    def index_entry(match: re.Match[str]) -> str:
        term = _latex_index_term(match.group("term"))
        return match.group("entry") + f"\n\n```{{=latex}}\n\\index{{{term}}}\n```"

    return markdown[:start] + entry_pattern.sub(index_entry, glossary)


def _index_cli_terms(markdown: str) -> str:
    """Index each distinct CLI flag and command at its first useful occurrence."""

    flag_pattern = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+")
    command_pattern = re.compile(r"\bverbx\s+[a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*)?")
    fence_pattern = re.compile(r"^\s*(`{3,}|~{3,})")
    seen: set[str] = set()
    pending: list[str] = []
    output: list[str] = []
    active_fence: str | None = None
    active_table = False

    def markers(terms: list[str]) -> list[str]:
        if not terms:
            return []
        commands = "\n".join(rf"\index{{{_latex_index_term(term)}}}" for term in terms)
        return ["", "```{=latex}", commands, "```", ""]

    for line in markdown.splitlines():
        table_line = line.lstrip().startswith("|")
        if active_table and not table_line:
            output.extend(markers(pending))
            pending.clear()
            active_table = False
        if table_line:
            active_table = True

        fence = fence_pattern.match(line)
        terms = [*flag_pattern.findall(line), *command_pattern.findall(line)]
        fresh = [term for term in terms if term not in seen]
        seen.update(fresh)
        if active_fence is None and not active_table and fresh:
            output.extend(markers(fresh))
        elif fresh:
            pending.extend(fresh)
        output.append(line)
        if fence:
            marker = fence.group(1)[0]
            if active_fence == marker:
                active_fence = None
                output.extend(markers(pending))
                pending.clear()
            elif active_fence is None:
                active_fence = marker
    output.extend(markers(pending))
    return "\n".join(output)


def _index_operational_cards(markdown: str) -> str:
    """Make all 588 operational-card subjects discoverable."""

    pattern = re.compile(
        r"(?m)^(?P<title>\*\*(?:Production|Automation|Quality|Validation|"
        r"Troubleshooting|Preset|Interaction|Audition|Asset|Release|Bus|"
        r"Signal-test|Triage) card.*?\*\*)$"
    )

    def add_marker(match: re.Match[str]) -> str:
        term = _plain_index_term(match.group("title"))
        return (
            "```{=latex}\n"
            + rf"\index{{{_latex_index_term(term)}}}"
            + "\n"
            + "```\n\n"
            + match.group(0)
        )

    return pattern.sub(add_marker, markdown)


_BIBLIOGRAPHY_INDEX_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Acoustic measurement", (r"\bacoustic measurements?\b", r"\broom measurements?\b")),
    ("Acoustic simulation", (r"\bacoustic simulations?\b", r"\broom simulations?\b")),
    ("Active acoustics", (r"\bactive acoustics?\b", r"\bactive reverberation\b")),
    ("Algorithmic reverberation", (r"\balgorithmic reverberation\b",)),
    ("Ambience extraction", (r"\bambience extraction\b", r"\bambient sound extraction\b")),
    ("Ambisonics", (r"\bambisonics?\b", r"\bhigher[- ]order ambisonics?\b")),
    ("Artificial reverberation", (r"\bartificial reverberation\b", r"\breverberators?\b")),
    ("Audio AI", (r"\baudio ai\b", r"\bartificial intelligence\b")),
    ("Auralization", (r"\bauraliz(?:ation|ations|ing)\b",)),
    ("Binaural rendering", (r"\bbinaural (?:audio|rendering|reproduction|synthesis)\b",)),
    ("Blind estimation", (r"\bblind (?:estimation|identification)\b",)),
    ("Convolution reverb", (r"\bconvolution(?:al)? reverberation\b", r"\bconvolution reverb\b")),
    ("Decay analysis", (r"\bdecay (?:analysis|curve|rate|slope)\b",)),
    ("Dereverberation", (r"\bdereverberation\b", r"\bde[- ]reverberation\b",)),
    ("Diffuse field", (r"\bdiffuse (?:field|sound field|reverberation)\b",)),
    (
        "Direct-to-reverberant ratio",
        (r"\bdirect[- ]to[- ]reverberant ratio\b", r"\bdr r?\b"),
    ),
    ("Early reflections", (r"\bearly reflections?\b",)),
    ("Energy decay", (r"\benergy decay (?:curve|relief|function)\b",)),
    ("Feedback delay network", (r"\bfeedback delay networks?\b", r"\bfdns?\b")),
    ("Filter design", (r"\bfilter design\b", r"\bdigital filters?\b")),
    (
        "Head-related transfer function",
        (r"\bhead[- ]related transfer functions?\b", r"\bhrtfs?\b"),
    ),
    ("Immersive audio", (r"\bimmersive audio\b", r"\bimmersive sound\b", r"\bdolby atmos\b")),
    (
        "Impulse-response measurement",
        (r"\bimpulse response measurements?\b", r"\bmeasur(?:e|ing) impulse responses?\b"),
    ),
    (
        "Impulse-response processing",
        (r"\bimpulse response (?:processing|equalization|manipulation)\b",),
    ),
    ("Machine learning", (r"\bmachine learning\b", r"\bdeep learning\b", r"\bneural networks?\b")),
    ("Multichannel audio", (r"\bmultichannel (?:audio|sound|reproduction|reverberation)\b",)),
    ("Object-based audio", (r"\bobject[- ]based audio\b", r"\baudio objects?\b")),
    ("Pole-zero analysis", (r"\bpole[- ]zero\b", r"\bpoles? and zeros?\b")),
    ("Reverberation fingerprint", (r"\breverberation fingerprints?\b",)),
    ("Reverberation time", (r"\breverberation times?\b", r"\brt60\b", r"\bt60\b")),
    ("Room acoustics", (r"\broom acoustics?\b",)),
    ("Room impulse response", (r"\broom impulse responses?\b", r"\brirs?\b")),
    ("Sound propagation", (r"\bsound propagation\b", r"\bacoustic propagation\b")),
    ("Spatial audio", (r"\bspatial audio\b", r"\bspatial sound\b", r"\b3d audio\b")),
    ("Speech dereverberation", (r"\bspeech dereverberation\b", r"\bdereverberat(?:e|ing) speech\b")),
    ("Statistical reverberation", (r"\bstatistical reverberation\b", r"\bstochastic reverberation\b")),
    ("Underwater reverberation", (r"\bunderwater reverberation\b", r"\bsonar reverberation\b")),
    ("Virtual acoustics", (r"\bvirtual acoustics?\b", r"\bvirtual rooms?\b")),
)


def _bibliography_index_phrases(title: str) -> list[str]:
    """Map a paper title to concise, controlled index phrases."""

    normalized = _plain_index_term(title).casefold()
    phrases = [
        phrase
        for phrase, patterns in _BIBLIOGRAPHY_INDEX_RULES
        if any(re.search(pattern, normalized) for pattern in patterns)
    ]
    if phrases:
        return phrases[:6]

    fallbacks = (
        ("Reverberation", r"\breverber(?:ation|ant|ance|ator|ators)\b"),
        ("Acoustics", r"\bacoustic(?:s|al|ally)?\b"),
        ("Audio signal processing", r"\baudio\b"),
    )
    return [phrase for phrase, pattern in fallbacks if re.search(pattern, normalized)][:1]


def _index_bibliography(markdown: str) -> str:
    """Index paper authors and controlled subject phrases, never full titles."""

    start = markdown.find("# Research Papers and References")
    if start == -1:
        return markdown
    references = markdown[start:]
    entry_pattern = re.compile(
        r"(?m)^(?P<entry>\*\*\[[^]]+\]\*\*\s+"
        r"(?P<authors>.+?)\s+\((?P<year>(?:18|19|20)\d{2}[a-z]?|n\.d\.)\)\.\s+"
        r"(?P<title>.+?)\.\s+\*[^*]+\*\..*)$"
    )

    def index_entry(match: re.Match[str]) -> str:
        authors = re.split(r"\s*;\s*|\s+&\s+|\s+and\s+", match.group("authors"))
        normalized: list[str] = []
        for author in authors:
            author = re.sub(r"\s+", " ", author).strip(" ,")
            if author.lower() == "et al.":
                continue
            if author in {"Jot, J.-M.", "Jot, J. M."}:
                author = "Jot, Jean-Marc"
            if author and author not in normalized:
                normalized.append(author)
        terms = [*normalized, *_bibliography_index_phrases(match.group("title"))]
        commands = "\n".join(rf"\index{{{_latex_index_term(term)}}}" for term in terms)
        return f"```{{=latex}}\n{commands}\n```\n\n{match.group('entry')}"

    references = entry_pattern.sub(index_entry, references)
    return markdown[:start] + references


def _typeset_research_references(markdown: str) -> str:
    """Render DOI records as consistent hanging bibliography entries."""

    start = markdown.find("# Research Papers and References")
    if start == -1:
        return markdown
    metadata: dict[str, dict[str, str]] = {}
    if REFERENCE_METADATA.exists():
        metadata = json.loads(REFERENCE_METADATA.read_text(encoding="utf-8"))
    references = markdown[start:]
    total_match = re.search(r"(?m)^Total entries:\s+([\d,]+)\b", references)
    if total_match is None:
        raise ValueError("Research bibliography is missing its declared entry total")
    expected_count = int(total_match.group(1).replace(",", ""))

    def typeset(match: re.Match[str]) -> str:
        doi = match.group("doi") or ""
        record = metadata.get(doi.lower(), {})
        detail = _reference_detail(record)
        authors = _reference_authors(match.group("authors"))
        title = _latex_text(match.group("title").strip('"').rstrip("."))
        venue = _latex_text(match.group("venue"))
        if match.group("key").startswith("BOOK"):
            citation_body = ". \\textit{" + title + "}. " + venue
        else:
            citation_body = (
                ". ``"
                + title
                + ("'' " if match.group("title").rstrip().endswith(("?", "!")) else ".'' ")
                + "\\textit{"
                + venue
                + "}"
            )
        identifier = (
            ". DOI: \\href{https://doi.org/" + doi + "}{\\nolinkurl{" + doi + "}}"
            if doi
            else (
                ". Source: \\href{"
                + _latex_url(match.group("source_url") or "")
                + "}{"
                + _latex_text(match.group("source_label") or "Primary source")
                + "}"
                if match.group("source_url")
                else ". " + _latex_text(match.group("note") or "No identifier supplied")
            )
        )
        return (
            "```{=latex}\n"
            "\\par\\noindent\\hangindent=1.7em\\hangafter=1\n"
            + _latex_text(authors.rstrip("."))
            + citation_body
            + detail
            + identifier
            + ", "
            + _latex_text(match.group("year"))
            + ".\\hfill{\\scriptsize\\color{verbxCover} ["
            + _latex_text(match.group("key"))
            + "]}\\par\\smallskip\n"
            "```"
        )

    references, count = RESEARCH_REFERENCE_PATTERN.subn(typeset, references)
    if count != expected_count:
        raise ValueError(
            f"Expected {expected_count} research references, formatted {count}"
        )
    return markdown[:start] + references


def _reference_authors(value: str) -> str:
    formatted: list[str] = []
    for author in re.split(r"\s*;\s*", value):
        author = author.strip()
        role = ""
        role_match = re.search(r",\s*(ed\.?|editor)$", author, flags=re.IGNORECASE)
        if role_match:
            role = ", ed."
            author = author[: role_match.start()].rstrip()
        if not author or author.lower() == "et al.":
            formatted.append("et al.")
            continue
        if author == "Unknown authors" or "," not in author:
            formatted.append(author)
            continue
        family, given = (part.strip() for part in author.split(",", 1))
        initials = " ".join(
            token
            if re.fullmatch(r"(?:[A-ZÀ-ÖØ-Þ]\.?(?:-[A-ZÀ-ÖØ-Þ]\.?)*)", token)
            else token[0].upper() + "."
            for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)?\.?", given)
            if token
        )
        formatted.append((f"{family}, {initials}" if initials else family) + role)
    return "; ".join(formatted)


def _reference_detail(record: dict[str, str]) -> str:
    volume = record.get("volume", "")
    issue = record.get("issue", "")
    page = record.get("page", "").replace("-", "--")
    if not any((volume, issue, page)):
        return ""
    volume_issue = volume + (f"({issue})" if issue else "")
    if volume_issue and page:
        return f" {_latex_text(volume_issue)}: {_latex_text(page)}"
    return " " + _latex_text(volume_issue or page)


def _index_markdown_headings(markdown: str) -> str:
    """Add index markers to headings without touching fenced code examples."""

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    fence_pattern = re.compile(r"^\s*(`{3,}|~{3,})")
    output: list[str] = []
    active_fence: str | None = None

    for line in markdown.splitlines():
        fence_match = fence_pattern.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            active_fence = None if active_fence == marker else marker
            output.append(line)
            continue

        output.append(line)
        if active_fence is not None:
            continue

        heading_match = heading_pattern.match(line)
        if not heading_match:
            continue
        heading = heading_match.group(2).strip()
        if heading.startswith(("\"", "'", "“", "‘")) and heading.endswith(
            ("\"", "'", "”", "’")
        ):
            continue
        term = _plain_index_term(heading)
        if not term or term.lower() in {"contents", "index"}:
            continue
        output.extend(("", "```{=latex}", f"\\index{{{_latex_index_term(term)}}}", "```"))

    return "\n".join(output)


def _plain_index_term(value: str) -> str:
    """Reduce Markdown heading or caption text to a readable index term."""

    value = re.sub(r"\{#[^}]+\}\s*$", "", value)
    value = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\\(?:emph|textit|textbf|texttt)\{([^{}]*)\}", r"\1", value)
    value = value.replace("`", "").replace("*", "").replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip(" .")
    value = re.sub(r"^(?:\d+(?:\.\d+)*|\d+[A-Za-z])[.)]?\s+", "", value)
    if re.fullmatch(r"(?:5\.1|7\.1|7\.1\.4)", value):
        value = f"Surround format {value}"
    return value


def _figure_index_term(value: str) -> str:
    """Reduce a descriptive figure caption to a concise index subject."""

    musical = re.search(
        r"(?:Opening measures|Score page|Page) of (?P<creator>.+?)(?:'s|’s)\s+"
        r"\\emph\{(?P<title>[^{}]+)\}\s+\((?P<date>[^)]+)\)",
        value,
        flags=re.IGNORECASE,
    )
    if musical is not None:
        creator = _musical_creator_index_name(musical.group("creator"))
        plain = f"{creator}. {musical.group('title')} ({musical.group('date')})"
        display = (
            _latex_text(creator)
            + ". \\textit{"
            + _latex_text(musical.group("title"))
            + "} ("
            + _latex_text(musical.group("date"))
            + ")"
        )
        return f"{_latex_index_term(plain)}@{display}"

    plain = _plain_index_term(value)
    folded = plain.casefold()
    controlled_subjects = (
        ("Energy decay relief", ("energy decay relief",)),
        ("Energy decay curve", ("energy decay curve",)),
        ("Pole-zero plot", ("pole-zero", "pole zero")),
        ("Feedback comb filter", ("feedback comb",)),
        ("Feedback delay network", ("feedback delay network", "fdn")),
        ("Schroeder allpass filter", ("schroeder allpass",)),
        ("Schroeder reverberator", ("schroeder reverberator", "schroeder-style")),
        ("Reverberation time", ("rt60", "t_{60}", "t60")),
        ("Spectrogram", ("spectrogram", "sonogram")),
        ("Signal flowgraph", ("flowgraph", "signal flow")),
    )
    for subject, needles in controlled_subjects:
        if any(needle in folded for needle in needles):
            return _latex_index_term(subject)

    concise = re.split(
        r"\s*(?:[,;:]|\bwith\b|\bshowing\b|\bused to\b|\bcompared with\b|"
        r"\bthrough\b|\bagainst\b|\bwhere\b|\bwhose\b)\s*",
        plain,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    concise = re.sub(r"^(?:Implementation-level|Illustrative|Expanded)\s+", "", concise)
    words = concise.split()
    if len(words) > 7:
        concise = " ".join(words[:7])
    return _latex_index_term(concise or "Figure")


def _musical_index_term(value: str) -> str:
    """Sort personal creators surname-first while keeping work titles italic."""

    match = re.fullmatch(
        r"(?P<creator>.+),\s+\*(?P<title>.+)\*\s+\((?P<date>[^)]+)\)",
        value,
    )
    if match is None:
        return _latex_index_term(_plain_index_term(value))

    creator = _musical_creator_index_name(match.group("creator"))
    plain = f"{creator}. {match.group('title')} ({match.group('date')})"
    sort_key = _latex_index_term(_plain_index_term(plain))
    display = (
        _latex_text(creator)
        + ". \\textit{"
        + _latex_text(match.group("title"))
        + "} ("
        + _latex_text(match.group("date"))
        + ")"
    )
    return f"{sort_key}@{display}"


_MUSICAL_NATURAL_ORDER_NAMES = frozenset(
    {
        "Actress",
        "Aphex Twin",
        "Basic Channel",
        "Bill Evans Trio",
        "Biosphere",
        "Black Sabbath",
        "Boris",
        "Burial",
        "Cocteau Twins",
        "Cowboy Junkies",
        "Cult of Luna",
        "Deafheaven",
        "Deepchord Presents Echospace",
        "FKA twigs",
        "Floating Points",
        "Gas",
        "Godspeed You! Black Emperor",
        "Joy Division",
        "Laraaji",
        "Led Zeppelin",
        "Loscil",
        "Massive Attack",
        "Maurizio",
        "Mogwai",
        "My Bloody Valentine",
        "Oneohtrix Point Never",
        "Pat Metheny Group",
        "Pink Floyd",
        "Pole",
        "Porter Ricks",
        "Portishead",
        "Prince and the Revolution",
        "Public Enemy",
        "Radiohead",
        "Rhythm & Sound with Cornel Campbell",
        "Sigur Rós",
        "Slowdive",
        "SOPHIE",
        "Stars of the Lid",
        "Sunn O)))",
        "Swans",
        "Talk Talk",
        "Tangerine Dream",
        "The Beatles",
        "The Ronettes",
        "The Staple Singers",
        "Tricky",
        "U2",
        "Vangelis",
        "Weyes Blood",
        "Yagya",
    }
)
_MUSICAL_NAME_OVERRIDES = {
    "Ralph Vaughan Williams": "Vaughan Williams, Ralph",
}
_MUSICAL_SURNAME_PARTICLES = frozenset(
    {"da", "de", "del", "della", "der", "di", "du", "la", "le", "van", "von"}
)


def _musical_creator_index_name(value: str) -> str:
    """Return creator credits in a stable surname-first index form."""

    creator = re.sub(r"\s+", " ", value).strip()
    if creator in _MUSICAL_NATURAL_ORDER_NAMES:
        return creator

    contributors = re.split(r",\s+(?:and\s+)?|\s+and\s+", creator)
    return "; ".join(
        _musical_contributor_index_name(contributor)
        for contributor in contributors
        if contributor
    )


def _musical_contributor_index_name(value: str) -> str:
    """Invert one personal name without inverting ensembles or stage names."""

    contributor = value.strip()
    if contributor in _MUSICAL_NAME_OVERRIDES:
        return _MUSICAL_NAME_OVERRIDES[contributor]
    if contributor in _MUSICAL_NATURAL_ORDER_NAMES:
        return contributor

    words = contributor.split()
    lowered = contributor.lower()
    ensemble_markers = (
        "ensemble",
        "orchestra",
        "quartet",
        "choir",
        "chorus",
        "trio",
        "collective",
        "group",
        "the revolution",
        "the upsetters",
    )
    if (
        len(words) < 2
        or contributor.startswith("The ")
        or any(marker in lowered for marker in ensemble_markers)
    ):
        return contributor

    surname_start = len(words) - 1
    while (
        surname_start > 0
        and words[surname_start - 1].casefold() in _MUSICAL_SURNAME_PARTICLES
    ):
        surname_start -= 1

    given = " ".join(words[:surname_start])
    surname = " ".join(words[surname_start:])
    return f"{surname}, {given}" if given else surname


def _latex_index_term(value: str) -> str:
    """Escape text for both makeindex syntax and the resulting LaTeX file."""

    value = value.replace('"', "").replace("!", " - ").replace("@", " at ").replace("|", " / ")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _force_table_of_figures_page_break(latex_path: Path) -> None:
    latex = latex_path.read_text(encoding="utf-8")
    figure_marker = "\n\\listoffigures\n"
    table_marker = "\n\\listoftables\n"
    if latex.count(figure_marker) != 1:
        raise ValueError("Expected exactly one body-level \\listoffigures command")
    if latex.count(table_marker) != 1:
        raise ValueError("Expected exactly one body-level \\listoftables command")
    latex = latex.replace(figure_marker, "\n\\clearpage\n\\listoffigures\n", 1)
    latex = latex.replace(
        table_marker,
        "\n\\clearpage\n\\listoftables\n\\clearpage\n\\listofplates\n",
        1,
    )
    latex_path.write_text(latex, encoding="utf-8")


def _render_pdf(markdown_path: Path, pdf_path: Path, author: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verbx_userguide_") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        pdf_markdown_path = tmpdir / "userguide_pdf.md"
        latex_path = tmpdir / "userguide.tex"
        pdf_markdown = _markdown_with_pdf_targets(
            markdown_path.read_text(encoding="utf-8")
        )
        pdf_markdown = _trim_pdf_figure_assets(pdf_markdown, tmpdir / "trimmed_figures")
        pdf_markdown_path.write_text(
            pdf_markdown,
            encoding="utf-8",
        )
        pandoc_command = [
            *_pandoc_base_command(pdf_markdown_path, author),
            "--to=latex",
            "-o",
            str(latex_path),
        ]
        subprocess.run(pandoc_command, cwd=ROOT, check=True)
        _force_table_of_figures_page_break(latex_path)
        _rewrite_longtable_specs(latex_path)
        for pass_index in range(5):
            subprocess.run(
                [
                    "xelatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(tmpdir),
                    str(latex_path),
                ],
                cwd=ROOT,
                check=True,
            )
            if pass_index == 0:
                subprocess.run(
                    ["makeindex", "-s", str(INDEX_STYLE), "userguide.idx"],
                    cwd=tmpdir,
                    check=True,
                )
                subprocess.run(
                    [
                        "makeindex",
                        "-s",
                        str(INDEX_STYLE),
                        "-o",
                        "userguide.mwi",
                        "userguide.mwx",
                    ],
                    cwd=tmpdir,
                    check=True,
                )
        shutil.copy2(tmpdir / "userguide.pdf", pdf_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--pdf-out", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument(
        "--md-only",
        action="store_true",
        help="Only write the consolidated Markdown file and skip PDF rendering.",
    )
    args = parser.parse_args()

    subprocess.run([sys.executable, str(PLUGIN_GUIDE_GENERATOR)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(BOOK_SUPPLEMENT_GENERATOR)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(GLOSSARY_GENERATOR)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(REVERB_PRIMER_ASSET_GENERATOR)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(TERMINAL_ASSET_GENERATOR)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(IMMERSIVE_AUDIO_ASSET_GENERATOR)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(AI_AUGMENTATION_ASSET_GENERATOR)], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, str(LITERATURE_SORTER), "--check"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(COMPOSITION_YEAR_NORMALIZER), "--check"],
        cwd=ROOT,
        check=True,
    )

    missing = [path for path in USERGUIDE_INCLUDED_SOURCES if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing USERGUIDE source: {_rel(path)}", file=sys.stderr)
        return 2

    markdown_out = args.markdown_out.resolve()
    pdf_out = args.pdf_out.resolve()

    _write_markdown(markdown_out, str(args.author))
    print(f"Wrote {markdown_out}")
    subprocess.run(
        [sys.executable, str(TYPOGRAPHY_CHECKER)],
        cwd=ROOT,
        check=True,
    )

    if args.md_only:
        return 0

    try:
        _render_pdf(markdown_out, pdf_out, str(args.author))
    except subprocess.CalledProcessError as exc:
        print(f"Failed to render PDF with pandoc/xelatex: {exc}", file=sys.stderr)
        return exc.returncode or 1

    print(f"Wrote {pdf_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
