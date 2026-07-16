from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "scripts_generate_docs_pdf",
    REPO_ROOT / "scripts_generate_docs_pdf.py",
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
DOCS_PDF = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(DOCS_PDF)

ASSET_SPEC = importlib.util.spec_from_file_location(
    "generate_reverb_primer_assets",
    REPO_ROOT / "scripts/generate_reverb_primer_assets.py",
)
assert ASSET_SPEC is not None and ASSET_SPEC.loader is not None
PRIMER_ASSETS = importlib.util.module_from_spec(ASSET_SPEC)
ASSET_SPEC.loader.exec_module(PRIMER_ASSETS)

COMPOSITION_PROJECT_TITLES = (
    "Compose with Infinite Sustain",
    "Reverse-Reverb Phrase Study",
    "Shimmer Canon and Harmonic Field",
    "Ducked-Reverb Rhythmic Counterpoint",
    "Morphing-Space Miniature",
    "Spatial Automation as Musical Form",
    "Compose a Reverb-Preset Etude",
    "Ambisonic Rotational Counterpoint",
    "Original Spatial Composition",
    "Harmonic Pedals from Decay",
    "Tempo-Synchronized Decay Canon",
    "Antiphonal Virtual Architecture",
    "Multiband Tail Orchestration",
    "Percussive Reflection Groove",
    "Vocal Space Dramaturgy",
    "Dub Send Performance",
    "Silence and Cadential Decay",
    "Room-Swap Theme and Variations",
    "Site-Specific IR Portrait",
    "Modulated-Reverb Timbre Study",
    "Spatial Fugue Across Rooms",
    "Adaptive Reverb Cue",
    "Reverb-Orchestrated Song",
    "Portfolio Capstone: Spatial Composition Cycle",
)


def test_parenthesized_doi_reference_is_matched_atomically() -> None:
    reference = (
        "**[XREF0794]** Rutkowski, Leon (1996). A comparison of transfer functions. "
        "*Applied Acoustics*. DOI: "
        "[10.1016/s0003-682x(96)00028-x]"
        "(https://doi.org/10.1016/s0003-682x(96)00028-x)"
    )

    match = DOCS_PDF.RESEARCH_REFERENCE_PATTERN.fullmatch(reference)

    assert match is not None
    assert match.group("doi") == "10.1016/s0003-682x(96)00028-x"


def test_pdf_markdown_has_no_parenthesized_doi_fence_artifacts() -> None:
    source = (REPO_ROOT / "docs/USERGUIDE.md").read_text(encoding="utf-8")

    rendered = DOCS_PDF._markdown_with_pdf_targets(source)

    assert "```00028-x)" not in rendered
    assert "```{=latex}\n\\par\\noindent" in rendered
    DOCS_PDF._validate_fenced_blocks(rendered)


def test_code_example_leads_reserve_space_and_forbid_boundary_breaks() -> None:
    source = (
        "Installation text.\n\n"
        "**With Homebrew (macOS):**\n\n"
        "```bash\nbrew install verbx\n```\n"
    )

    rendered = DOCS_PDF._keep_code_leads_with_examples(source)

    assert rendered.index(r"\Needspace{7\baselineskip}") < rendered.index(
        "**With Homebrew (macOS):**"
    )
    assert rendered.index("**With Homebrew (macOS):**") < rendered.index(
        r"\nopagebreak[4]"
    )
    assert rendered.index(r"\nopagebreak[4]") < rendered.index("```bash")
    DOCS_PDF._validate_fenced_blocks(rendered)


def test_reverb_primer_is_promoted_to_a_standalone_pdf_chapter() -> None:
    source = (
        "# verbx\n\n"
        "## What Is Reverb? (and Why Does verbx Sound Different)\n\n"
        "Primer.\n\n"
        "### Musical Examples\n\n"
        "#### Listening Test\n\n"
        "## Core Concepts\n\n"
        "Reference.\n"
    )

    rendered = DOCS_PDF._promote_reverb_primer_to_chapter(source)

    assert "# What Is Reverb? (and Why Does verbx Sound Different)" in rendered
    assert "## Musical Examples" in rendered
    assert "### Listening Test" in rendered
    assert "# verbx Reference\n\n## Core Concepts" in rendered


def test_malformed_fence_suffix_is_rejected() -> None:
    with pytest.raises(ValueError, match="Malformed Markdown fence"):
        DOCS_PDF._validate_fenced_blocks("```{=latex}\n\\index{Example}\n```00028-x)\n")


def test_homework_appendix_contains_twenty_four_composition_projects() -> None:
    homework = (REPO_ROOT / "docs/HOMEWORK_ASSIGNMENTS.md").read_text(encoding="utf-8")

    assert len(re.findall(r"^## Project \d+:", homework, flags=re.MULTILINE)) == 48
    assert homework.count("**Project mode:** Musical composition and production.") == 24
    for title in COMPOSITION_PROJECT_TITLES:
        assert title in homework


def test_all_literature_lists_are_alphabetized() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts_sort_literature.py"), "--check"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_main_bibliography_preserves_all_reference_ids() -> None:
    references = (REPO_ROOT / "docs/REFERENCES.md").read_text(encoding="utf-8")
    reference_ids = re.findall(r"^\*\*\[([^]]+)\]\*\*", references, flags=re.MULTILINE)
    xref_ids = [reference_id for reference_id in reference_ids if reference_id.startswith("XREF")]

    assert len(reference_ids) == 1002
    assert len(set(reference_ids)) == 1002
    assert len(xref_ids) == 900
    assert "102 curated annotated entries + 900 extended Crossref entries" in references
    assert "bringing the guide bibliography to 1,002 total entries" in references


def test_reference_reading_note_is_not_a_setext_heading() -> None:
    references = (REPO_ROOT / "docs/REFERENCES.md").read_text(encoding="utf-8")

    assert "The best survey of the field in existence" not in references
    assert (
        "A historical survey connecting mechanical, algorithmic, convolution, "
        "and perceptual reverberation.\n\n---"
    ) in references


def test_title_page_uses_white_background() -> None:
    preamble = (REPO_ROOT / "docs/assets/pandoc_pdf_preamble.tex").read_text(encoding="utf-8")
    title_page = preamble.split(r"\begin{titlepage}", 1)[1].split(r"\end{titlepage}", 1)[0]

    assert r"\pagecolor{white}\color{verbxCover}" in title_page
    assert r"\pagecolor{verbxCover}" not in title_page


def test_pdf_preamble_prevents_widows_orphans_and_stranded_headings() -> None:
    preamble = (REPO_ROOT / "docs/assets/pandoc_pdf_preamble.tex").read_text(
        encoding="utf-8"
    )

    for rule in (
        r"\raggedbottom",
        r"\clubpenalty=10000",
        r"\widowpenalty=10000",
        r"\displaywidowpenalty=10000",
        r"\brokenpenalty=10000",
        r"\interfootnotelinepenalty=10000",
        r"\pretocmd{\section}{\Needspace{6\baselineskip}}{}{}",
        r"\pretocmd{\subsection}{\Needspace{5\baselineskip}}{}{}",
        r"\pretocmd{\subsubsection}{\Needspace{4\baselineskip}}{}{}",
    ):
        assert rule in preamble


def test_reverb_primer_has_textbook_depth_and_complete_figure_set() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## What Is Reverb? (and Why Does verbx Sound Different)")
    end = readme.index("\n---", start)
    primer = readme[start:end]

    words = re.findall(r"\b[\w'-]+\b", primer)
    assert len(words) >= 8000
    assert "### Musical Examples" in primer
    assert "### DSP Overview" in primer
    assert primer.count("```mermaid") == 10
    assert len(re.findall(r"^!\[", primer, flags=re.MULTILINE)) == 18
    assert len(re.findall(r"^\*\*Figure:", primer, flags=re.MULTILINE)) == 28
    assert "Schroeder_Reverberators.html" in primer

    for topic in (
        "Feedback Comb Filters",
        "Schroeder Allpass Filters",
        "Allpass Networks",
        "The Schroeder Reverberator",
        "Feedback Delay Networks",
        "Convolution and Partitioned FFT Processing",
    ):
        assert topic in primer


def test_reverb_primer_mermaid_assets_convert_for_pdf() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## What Is Reverb? (and Why Does verbx Sound Different)")
    end = readme.index("\n---", start)
    primer = readme[start:end]

    paths = re.findall(r"^%% verbx-static:\s+(\S+)$", primer, flags=re.MULTILINE)
    assert len(paths) == 10
    for path in paths:
        assert (REPO_ROOT / path).is_file()

    converted = DOCS_PDF._replace_mermaid_with_static_assets(primer)
    assert "```mermaid" not in converted
    assert converted.count("docs/assets/reverb_primer/") == 28
    assert converted.count("**Figure:") == 28

    generated_paths = re.findall(r"\]\((docs/assets/reverb_primer/[^)]+)\)", converted)
    assert len(generated_paths) == 28
    for path in generated_paths:
        assert (REPO_ROOT / path).is_file()

    pdf_ready = DOCS_PDF._convert_figure_captions(converted)
    assert pdf_ready.count(r"\begin{minipage}{\linewidth}") == 28
    assert pdf_ready.count(r"\end{minipage}") == 28
    assert pdf_ready.count(r"\includegraphics") == 28
    assert "![" not in pdf_ready
    first_group = pdf_ready.index(r"\begin{minipage}{\linewidth}")
    first_lead = pdf_ready.index(r"\verbxFigureLead")
    first_image = pdf_ready.index(r"\includegraphics")
    first_caption = pdf_ready.index(r"\verbxFigureCaption")
    first_group_end = pdf_ready.index(r"\end{minipage}")
    assert first_group < first_lead < first_image < first_caption < first_group_end

    consolidated = converted.replace("(docs/assets/reverb_primer/", "(assets/reverb_primer/")
    consolidated_ready = DOCS_PDF._convert_figure_captions(consolidated)
    assert consolidated_ready.count(r"\begin{minipage}{\linewidth}") == 28
    assert consolidated_ready.count(r"\includegraphics") == 28


def test_pdf_figure_assets_trim_trailing_background(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGB", (240, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 130), fill="black")
    image.save(source)

    markdown = f"![Synthetic figure]({source})"
    rendered = DOCS_PDF._trim_pdf_figure_assets(markdown, tmp_path / "trimmed")
    match = re.search(r"\]\(([^)]+)\)", rendered)

    assert match is not None
    trimmed_path = Path(match.group(1))
    assert trimmed_path != source
    with Image.open(trimmed_path) as trimmed:
        assert trimmed.width == 240
        assert trimmed.height == 143


def test_reverb_primer_assets_have_tight_caption_edges() -> None:
    asset_dir = REPO_ROOT / "docs" / "assets" / "reverb_primer"
    for path in asset_dir.glob("*.png"):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        difference = ImageChops.difference(
            image, Image.new("RGB", image.size, "white")
        )
        bounds = difference.getbbox()
        assert bounds is not None
        assert image.height - bounds[3] <= 24, path.name


def test_reverb_primer_math_labels_use_positioned_scripts() -> None:
    image = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(image)

    exponent_runs, _, _, _ = PRIMER_ASSETS._math_layout(
        draw, "z^{-M}", PRIMER_ASSETS.F_FLOW
    )
    matrix_runs, _, _, _ = PRIMER_ASSETS._math_layout(
        draw, "C_L^T", PRIMER_ASSETS.F_FLOW
    )

    assert any(
        text == "\N{MINUS SIGN}M" and y < 0 for text, _, _, y in exponent_runs
    )
    assert any(text == "L" and y > 0 for text, _, _, y in matrix_runs)
    assert any(text == "T" and y < 0 for text, _, _, y in matrix_runs)
    assert all("^" not in text and "_" not in text for text, *_ in exponent_runs + matrix_runs)


def test_documentation_avoids_plaintext_caret_delay_notation() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    generator = (REPO_ROOT / "scripts/generate_reverb_primer_assets.py").read_text(
        encoding="utf-8"
    )

    assert "z^-" not in readme
    assert "z^-" not in generator
    assert '"$z^{-M}$"' in generator
    assert "10^(" not in readme
    assert "10^(" not in generator


def test_speaker_layout_figure_uses_dedicated_plan_and_elevation_renderer() -> None:
    generator = (REPO_ROOT / "scripts/generate_userguide_figures.py").read_text(
        encoding="utf-8"
    )
    figure_guide = (REPO_ROOT / "scripts/generate_figure_guide.py").read_text(
        encoding="utf-8"
    )
    figures = (REPO_ROOT / "docs/FIGURES.md").read_text(encoding="utf-8")

    assert "def fig_speaker_layout_coverage" in generator
    assert '"72_speaker_layout_coverage.png",\n        "layout",' in generator
    assert "rng.uniform(size=(9, 2))" in generator
    assert 'if kind == "layout":' in generator
    assert '"layout": (' in figure_guide
    assert '"Listener-centered plan views encode nominal azimuth' in figure_guide
    assert "Loudspeaker Layouts: Plan and Elevation" in figures
    assert "Radial lines indicate nominal bearing only" in figures
    assert "Coverage state (category)" not in figures


def test_illustrated_guide_compactor_accepts_layout_specific_reading_instructions() -> None:
    figures = (REPO_ROOT / "docs/FIGURES.md").read_text(encoding="utf-8")

    compacted = DOCS_PDF._compact_illustrated_guide(figures)

    assert compacted.count(r"\begin{minipage}[t]{0.43\textwidth}") == 99
    assert compacted.count(r"\begin{minipage}[t]{\linewidth}") == 1
    assert "Read each plan with front at the top" in compacted
    assert "Loudspeaker Layouts: Plan and Elevation" in compacted
