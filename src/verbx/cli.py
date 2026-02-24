import typer
from rich import print
import librosa
import soundfile as sf
import numpy as np

app = typer.Typer(help="verbx: Extreme ambient audio reverberator")

@app.command()
def process(
    input_file: str = typer.Argument(..., help="Path to the input audio file"),
    output_file: str = typer.Argument(..., help="Path to the output audio file"),
):
    """
    Apply extreme reverberation to an audio file.
    """
    print(f"[bold green]Processing[/bold green] {input_file} to {output_file}...")

    try:
        y, sr = librosa.load(input_file, sr=None)
        print(f"Loaded audio with sample rate {sr}Hz, duration {librosa.get_duration(y=y, sr=sr):.2f}s")

        # TODO: Implement actual reverb logic here.
        # For now, just pass through (or maybe add a simple delay/echo to verify)

        sf.write(output_file, y, sr)
        print(f"Saved to {output_file}")

    except Exception as e:
        print(f"[bold red]Error processing file:[/bold red] {e}")
        raise typer.Exit(code=1)

    print("[bold blue]Done![/bold blue]")

if __name__ == "__main__":
    app()
