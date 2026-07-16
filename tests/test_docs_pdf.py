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

FIGURE_SPEC = importlib.util.spec_from_file_location(
    "generate_userguide_figures",
    REPO_ROOT / "scripts/generate_userguide_figures.py",
)
assert FIGURE_SPEC is not None and FIGURE_SPEC.loader is not None
USERGUIDE_FIGURES = importlib.util.module_from_spec(FIGURE_SPEC)
FIGURE_SPEC.loader.exec_module(USERGUIDE_FIGURES)

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


def test_musical_work_titles_remain_italic_in_pdf_index() -> None:
    term = DOCS_PDF._musical_index_term(
        "Karlheinz Stockhausen, *Kontakte* (1958–1960)"
    )

    assert term.startswith("Karlheinz Stockhausen, Kontakte (1958–1960)@")
    assert r"Karlheinz Stockhausen, \textit{Kontakte} (1958–1960)" in term


def test_musical_workflow_titles_are_italicized_in_sources() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    cookbook = (REPO_ROOT / "docs/EXTREME_COOKBOOK.md").read_text(encoding="utf-8")

    assert "Alvin Lucier's *I Am Sitting in a Room* technique" in readme
    assert "Alvin Lucier / *I Am Sitting in a Room*" in cookbook
    assert "Brian Eno / *Discreet Music*" in cookbook
    assert "Pauline Oliveros / *Deep Listening*" in cookbook


def test_every_extreme_cookbook_recipe_has_a_title() -> None:
    cookbook = (REPO_ROOT / "docs/EXTREME_COOKBOOK.md").read_text(encoding="utf-8")
    recipes = re.findall(r"^### Recipe (\d+): (\S.+)$", cookbook, flags=re.MULTILINE)

    assert [int(number) for number, _title in recipes] == list(range(1, 101))
    assert all(title.strip() for _number, title in recipes)
    assert re.search(r"^\*\*Recipe \d+\*\*$", cookbook, flags=re.MULTILINE) is None


def test_reader_facing_numeric_ranges_use_en_dashes() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    cookbook = (REPO_ROOT / "docs/EXTREME_COOKBOOK.md").read_text(encoding="utf-8")
    figure_generator = (REPO_ROOT / "scripts/generate_userguide_figures.py").read_text(
        encoding="utf-8"
    )

    for start in range(1, 100, 10):
        assert f"Recipes {start}–{start + 9}" in cookbook
    assert re.search(r"Recipes \d+-\d+", cookbook) is None
    assert "Minute 0–5" in readme
    assert "Early reflections<br/>10–80 ms" in readme
    assert "(0-1)" not in figure_generator
    assert "(0–1)" in figure_generator


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


def test_pdf_author_credit_includes_academic_credential() -> None:
    preamble = (REPO_ROOT / "docs/assets/pandoc_pdf_preamble.tex").read_text(
        encoding="utf-8"
    )

    assert DOCS_PDF.DEFAULT_AUTHOR == "Colby Leider, PhD"
    assert r"Documentation and software by \@author." in preamble
    assert r"\@author \quad / \quad \@date" in preamble


def test_cli_reference_title_omits_autogenerated_label() -> None:
    cli_reference = (REPO_ROOT / "docs/CLI_REFERENCE.md").read_text(encoding="utf-8")

    assert cli_reference.startswith("# CLI Reference\n")
    assert "CLI Reference (Autogenerated)" not in cli_reference


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


def test_table_of_figures_starts_on_a_new_page(tmp_path: Path) -> None:
    latex_path = tmp_path / "guide.tex"
    latex_path.write_text(
        "\\tableofcontents\n}\n\\listoffigures\n\\mainmatter\n",
        encoding="utf-8",
    )

    DOCS_PDF._force_table_of_figures_page_break(latex_path)

    latex = latex_path.read_text(encoding="utf-8")
    assert "\\tableofcontents\n}\n\\clearpage\n\\listoffigures" in latex


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

    assert any(text == "\N{EN DASH}" and y < 0 for text, _, _, y in exponent_runs)
    assert any(text == "L" and y > 0 for text, _, _, y in matrix_runs)
    assert any(text == "T" and y < 0 for text, _, _, y in matrix_runs)
    variable_runs = [
        (text, selected_font)
        for text, selected_font, _, _ in exponent_runs + matrix_runs
        if any(character.isalpha() for character in text)
    ]
    assert variable_runs
    assert all(
        "italic" in selected_font.getname()[1].lower()
        for _, selected_font in variable_runs
    )
    assert all("^" not in text and "_" not in text for text, *_ in exponent_runs + matrix_runs)


def test_illustrated_guide_math_segments_use_true_italic_fonts() -> None:
    segments = USERGUIDE_FIGURES._rich_segments(
        "Ambisonics order $N$; absorption $\\alpha$",
        USERGUIDE_FIGURES.F_SMALL,
    )

    roman_runs = [selected_font for text, selected_font in segments if "order" in text]
    variable_runs = [
        selected_font
        for text, selected_font in segments
        if text in {"N", "α"}
    ]

    assert roman_runs
    assert variable_runs
    assert all("italic" not in selected_font.getname()[1].lower() for selected_font in roman_runs)
    assert all("italic" in selected_font.getname()[1].lower() for selected_font in variable_runs)

    image = Image.new("RGB", (400, 120), "white")
    draw = ImageDraw.Draw(image)
    script_runs, _, _, _ = USERGUIDE_FIGURES._math_layout(
        draw,
        "T_{60}",
        USERGUIDE_FIGURES.F_SMALL,
    )
    indexed_runs, _, _, _ = USERGUIDE_FIGURES._math_layout(
        draw,
        "A_k",
        USERGUIDE_FIGURES.F_SMALL,
    )
    powered_runs, _, _, _ = USERGUIDE_FIGURES._math_layout(
        draw,
        "2^{7/12}",
        USERGUIDE_FIGURES.F_SMALL,
    )

    assert any(text == "60" and y > 0 for text, _, _, y in script_runs)
    assert any(
        text == "T" and "italic" in selected_font.getname()[1].lower()
        for text, selected_font, _, _ in script_runs
    )
    assert any(text == "k" and y > 0 for text, _, _, y in indexed_runs)
    assert any(
        text == "A" and "italic" in selected_font.getname()[1].lower()
        for text, selected_font, _, _ in indexed_runs
    )
    assert any(text == "7/12" and y < 0 for text, _, _, y in powered_runs)
    assert all("_" not in text and "^" not in text for text, *_ in indexed_runs + powered_runs)


def test_book_sources_mark_mathematical_variables_as_inline_math() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    ir_synthesis = (REPO_ROOT / "docs/IR_SYNTHESIS.md").read_text(encoding="utf-8")
    references = (REPO_ROOT / "docs/REFERENCES.md").read_text(encoding="utf-8")
    primer_generator = (
        REPO_ROOT / "scripts/generate_reverb_primer_assets.py"
    ).read_text(encoding="utf-8")
    figure_guide = (REPO_ROOT / "scripts/generate_figure_guide.py").read_text(
        encoding="utf-8"
    )
    figure_generator = (
        REPO_ROOT / "scripts/generate_userguide_figures.py"
    ).read_text(encoding="utf-8")

    for expected in (
        "full $M$-input-to-$N$-output matrix routing",
        "delay length $M$ and loop gain $g$",
        "an explicit internal state, $M$-sample delay",
        "fully coupled $N$-line FDN",
    ):
        assert expected in readme

    for expected in (
        "An impulse response $h[n]$",
        "a set of $N$\ndelay lines",
        "mixing matrix $M$",
        "degree $k$ introduces $k$ cross-connections",
        "Lower $Q$ (5–15)",
        "$A_k$, frequency $f_k$, phase $\\phi_k$, and time constant $\\tau_k$",
    ):
        assert expected in ir_synthesis

    assert "h_k[n] = A_k" in ir_synthesis
    assert "e^{-n/\\tau_k}" in ir_synthesis
    assert "$T_{60}=0.161V/A$" in references
    assert "$T_{60}=0.161V/[-S\\ln(1-\\bar{\\alpha})]$" in references
    assert '"IR partitions\\n$H_0$  $H_1$  ...  $H_K$"' in primer_generator
    assert '("fft", "parts", "$X[k]$")' in primer_generator
    assert '"Ambisonics order $N$ (integer)"' in figure_guide
    assert '"Blend coordinate $A$' in figure_guide
    assert '"$T_{60}$ Decay Families"' in figure_generator
    assert '("$C_{80}$", "–3.8 dB", GOLD)' in figure_generator
    assert '("$G$", "$T_{60}$ gains")' in primer_generator


def test_expanded_fdn_projection_and_gain_boxes_do_not_overlap() -> None:
    projection = PRIMER_ASSETS.FDN_OUTPUT_PROJECTION_BOX
    gain = PRIMER_ASSETS.FDN_GAIN_BOX

    assert projection[3] + 20 <= gain[1]


def test_documentation_avoids_plaintext_caret_delay_notation() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    generator = (REPO_ROOT / "scripts/generate_reverb_primer_assets.py").read_text(
        encoding="utf-8"
    )

    assert "z^-" not in readme
    assert "z^-" not in generator
    assert '"$z^{-M}$"' in generator
    assert '"$M$\\N{EN DASH}sample delay"' in generator
    assert "\\N{MINUS SIGN}" not in generator
    assert "10^(" not in readme
    assert "10^(" not in generator


def test_userguide_sources_typeset_power_expressions_as_math() -> None:
    fenced_code = re.compile(r"```.*?```", re.DOTALL)
    display_math = re.compile(r"\$\$.*?\$\$", re.DOTALL)
    inline_math = re.compile(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", re.DOTALL)
    plaintext_power = re.compile(
        r"(?:\b[0-9A-Za-z]+|\([^()\n]+\))\s*\^\s*"
        r"(?:\([^()\n]+\)|[-+]?[0-9A-Za-z]+)"
    )

    for source_path in DOCS_PDF.USERGUIDE_SOURCES:
        source = source_path.read_text(encoding="utf-8")
        prose = fenced_code.sub("", source)
        prose = display_math.sub("", prose)
        prose = inline_math.sub("", prose)
        match = plaintext_power.search(prose)
        assert match is None, (
            f"{source_path.relative_to(REPO_ROOT)} contains a plaintext power "
            f"expression: {match.group(0)!r}"
        )

    cookbook = (REPO_ROOT / "docs/EXTREME_COOKBOOK.md").read_text(encoding="utf-8")
    assert r"$2^{7/12} \approx 1.498$" in cookbook
    assert r"$2^{n}$" in cookbook
    assert r"$2^{3} = 8$" in cookbook


def test_userguide_sources_use_en_dash_for_displayed_negative_values() -> None:
    fenced_code = re.compile(r"```.*?```", re.DOTALL)
    display_math = re.compile(r"\$\$.*?\$\$", re.DOTALL)
    inline_math = re.compile(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", re.DOTALL)
    inline_code = re.compile(r"`[^`\n]+`")
    ascii_negative = re.compile(r"(?<![A-Za-z0-9_/.])-(?=\d)")
    standalone_numeric_code = re.compile(r"`-[0-9][^`\n]*`")

    for source_path in DOCS_PDF.USERGUIDE_SOURCES:
        source = source_path.read_text(encoding="utf-8")
        prose = fenced_code.sub("", source)
        prose = display_math.sub("", prose)
        prose = inline_math.sub("", prose)
        prose = inline_code.sub("", prose)
        match = ascii_negative.search(prose)
        assert match is None, (
            f"{source_path.relative_to(REPO_ROOT)} contains an ASCII negative "
            f"value in displayed prose near: {prose[match.start():match.start() + 24]!r}"
        )

        match = standalone_numeric_code.search(source)
        assert match is None, (
            f"{source_path.relative_to(REPO_ROOT)} contains an ASCII negative "
            f"display value: {match.group(0)!r}"
        )


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
    assert r"\verbxFigureLead{$T_{60}$ decay families}" in compacted
    assert r"\verbxFigureCaption{$T_{60}$ decay families}" in compacted
    assert r"\$T\_\{60\}\$ decay families" not in compacted


def test_latex_caption_text_preserves_positioned_scripts() -> None:
    rendered = DOCS_PDF._latex_text_with_inline_math(
        r"Amplitude $A_k$ & decay $e^{-n/\tau_k}$"
    )

    assert rendered == r"Amplitude $A_k$ \& decay $e^{-n/\tau_k}$"


def test_immersive_audio_chapter_distinguishes_beds_objects_and_monitoring() -> None:
    chapter_path = REPO_ROOT / "docs/IMMERSIVE_AUDIO.md"
    chapter = chapter_path.read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert chapter_path in DOCS_PDF.USERGUIDE_SOURCES
    assert "standard Dolby 7.1.2 bed" in chapter
    assert "7.1.4 monitoring array" in chapter
    assert "does not currently author Dolby object metadata" in chapter
    assert "ADM BWF, DAMF, or IMF IAB" in chapter
    assert "Ltm, Rtm" in chapter
    assert "Ltf/Rtf" in chapter
    assert "Full Atmos bed format" not in readme
    assert "Common Atmos monitoring/render layout" in readme

    figures = [
        "01_bed_vs_monitor_layout.png",
        "02_atmos_renderer_architecture.png",
        "03_hybrid_reverb_topology.png",
        "04_translation_qc_loop.png",
        "05_delivery_boundary.png",
    ]
    for filename in figures:
        assert f"assets/immersive_audio/{filename}" in chapter
        assert (REPO_ROOT / "docs/assets/immersive_audio" / filename).is_file()

    assert chapter.count("**Figure:") == len(figures)
