import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def version():
    """Show the application version."""
    console.print("[bold green]verbx v0.1.0[/bold green]")

@app.command()
def process(input_file: str, output_file: str):
    """Process an audio file with reverberation."""
    console.print(f"Processing [bold cyan]{input_file}[/bold cyan] -> [bold cyan]{output_file}[/bold cyan]")
    # Placeholder for actual processing logic

if __name__ == "__main__":
    app()
