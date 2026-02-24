import json
from pathlib import Path
from typing import Optional

import soundfile as soundfile_lib
import typer
from rich.console import Console
from rich.table import Table

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.core.pipeline import process_pipeline
from verbx.io.audio import read_audio
from verbx.io.progress import create_progress
from verbx.presets.default_presets import list_presets

app = typer.Typer()
console = Console()


@app.command()
def render(
    infile: Path = typer.Argument(..., help="Input audio file"),
    outfile: Path = typer.Argument(..., help="Output audio file"),
    engine: str = typer.Option("algo", help="Engine type: algo, conv"),
    rt60: float = typer.Option(2.0, help="Reverb time (s)"),
    wet: float = typer.Option(0.5, help="Wet mix"),
    dry: float = typer.Option(0.5, help="Dry mix"),
    repeat: int = typer.Option(1, help="Repeat count"),
    freeze: bool = typer.Option(False, help="Enable freeze mode"),
    start: float = typer.Option(0.0, help="Start time (s) for freeze"),
    end: float = typer.Option(0.0, help="End time (s) for freeze"),
    analysis_out: Optional[str] = typer.Option(None, help="Output analysis JSON path"),
    silent: bool = typer.Option(False, help="Suppress output"),
    progress: bool = typer.Option(True, help="Show progress bar"),
    impulse_response: Optional[str] = typer.Option(
        None, help="Impulse response path for conv engine"
    ),
):
    """Render audio with reverb."""
    if not silent:
        console.print(f"[bold green]Rendering[/bold green] {infile} -> {outfile}")
        console.print(
            f"Engine: {engine}, RT60: {rt60}, Wet: {wet}, Dry: {dry}, Repeat: {repeat}"
        )
        if freeze:
            console.print(f"Freeze Enabled: Start={start}, End={end}")

    with create_progress() as p:
        task = p.add_task("Processing...", total=100)

        def progress_cb(advance_samples):
            if progress:
                p.update(task, advance=advance_samples)

        try:
            info = soundfile_lib.info(str(infile))
            total_samples = info.frames * repeat
            p.update(task, total=total_samples)
        except Exception:
            p.update(task, total=None)

        result_stats = process_pipeline(
            infile=infile,
            outfile=outfile,
            engine_type=engine,
            rt60=rt60,
            wet=wet,
            dry=dry,
            repeat=repeat,
            freeze=freeze,
            start=start,
            end=end,
            analysis_out=analysis_out,
            silent=silent,
            progress_callback=progress_cb,
            impulse_response=impulse_response,
        )

    if not silent and result_stats:
        console.print("[bold green]Complete![/bold green]")
        table = Table(title="Output Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        for k, v in result_stats.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, str(v))
        console.print(table)


@app.command()
def analyze(
    infile: Path = typer.Argument(..., help="Input audio file"),
    json_out: Optional[str] = typer.Option(None, help="Output JSON path"),
):
    """Analyze audio file."""
    console.print(f"[bold blue]Analyzing[/bold blue] {infile}")

    audio, sr = read_audio(infile)

    analyzer = AudioAnalyzer()
    results = analyzer.analyze(audio, sr)

    table = Table(title="Audio Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in results.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)

    if json_out:
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"Analysis saved to {json_out}")


@app.command()
def presets():
    """List available presets."""
    console.print("[bold yellow]Available Presets:[/bold yellow]")
    for p in list_presets():
        console.print(f"- {p}")


@app.command()
def suggest(infile: Path):
    """Suggest reverb settings based on analysis."""
    console.print(f"[bold magenta]Suggesting for[/bold magenta] {infile}")

    audio, sr = read_audio(infile)
    analyzer = AudioAnalyzer()
    stats = analyzer.analyze(audio, sr)

    centroid = float(stats.get("spectral_centroid_mean", 0))
    peak = float(stats.get("peak_dbfs", -100))

    console.print(f"Spectral Centroid: {centroid:.2f} Hz")
    console.print(f"Peak Level: {peak:.2f} dBFS")

    if centroid < 1000:
        console.print(
            "Suggestion: Dark content. Try 'cathedral' preset (long tail, low damping)."
        )
        console.print("Settings: --engine algo --rt60 4.0 --wet 0.6")
    elif centroid > 3000:
        console.print(
            "Suggestion: Bright content. Try 'ambience' preset (short tail, high damping)."
        )
        console.print("Settings: --engine algo --rt60 1.5 --wet 0.4 --damping 0.6")
    else:
        console.print("Suggestion: Balanced content. Try 'plate' style.")
        console.print("Settings: --engine algo --rt60 2.5 --wet 0.5")


if __name__ == "__main__":
    app()
