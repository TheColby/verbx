"""Render pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.config import RenderConfig
from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine
from verbx.core.convolution_reverb import ConvolutionReverbConfig, ConvolutionReverbEngine
from verbx.core.engine_base import ReverbEngine
from verbx.core.freeze import freeze_segment
from verbx.core.repeat import repeat_process
from verbx.io.audio import read_audio, validate_audio_path, write_audio
from verbx.io.progress import RenderProgress


def run_render_pipeline(infile: Path, outfile: Path, config: RenderConfig) -> dict[str, Any]:
    """Run the complete render pipeline and return report metadata."""
    validate_audio_path(str(infile))

    with RenderProgress(enabled=(config.progress and not config.silent)) as progress:
        audio, sr = read_audio(str(infile))
        progress.mark_read()

        engine_name, engine = _resolve_engine(config)
        progress.set_passes(max(1, config.repeat))

        input_for_engine = audio
        if config.freeze:
            input_for_engine = freeze_segment(
                audio=audio,
                sr=sr,
                start=config.start,
                end=config.end,
                mode="loop",
                xfade_ms=100.0,
            )

        rendered = repeat_process(
            engine=engine,
            audio=input_for_engine,
            sr=sr,
            n=config.repeat,
            target_dbfs=-1.0,
            limiter=True,
            progress_callback=lambda idx, total: progress.mark_process_pass(idx),
        )

        write_audio(str(outfile), rendered, sr)
        progress.mark_write()

        report: dict[str, Any] = {
            "engine": engine_name,
            "sample_rate": sr,
            "input_samples": int(audio.shape[0]),
            "output_samples": int(rendered.shape[0]),
            "channels": int(rendered.shape[1]),
            "config": asdict(config),
        }

        if not config.silent:
            analyzer = AudioAnalyzer()
            report["input"] = analyzer.analyze(audio, sr)
            report["output"] = analyzer.analyze(rendered, sr)
            analysis_path = _resolve_analysis_path(outfile, config.analysis_out)
            analysis_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["analysis_path"] = str(analysis_path)

        progress.mark_analyze()

    return report


def _resolve_engine(config: RenderConfig) -> tuple[str, ReverbEngine]:
    engine_name = config.engine
    if engine_name == "auto":
        engine_name = "conv" if config.ir is not None else "algo"

    if engine_name == "conv":
        if config.ir is None:
            msg = "Convolution engine requires --ir when --engine conv is selected"
            raise ValueError(msg)
        return "conv", ConvolutionReverbEngine(
            ConvolutionReverbConfig(
                wet=config.wet,
                dry=config.dry,
                ir_path=config.ir,
                ir_normalize=config.ir_normalize,
                partition_size=config.partition_size,
                tail_limit=config.tail_limit,
                threads=config.threads,
            )
        )

    return "algo", AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=config.rt60,
            pre_delay_ms=config.pre_delay_ms,
            damping=config.damping,
            width=config.width,
            mod_depth_ms=config.mod_depth_ms,
            mod_rate_hz=config.mod_rate_hz,
            wet=config.wet,
            dry=config.dry,
            block_size=config.block_size,
        )
    )


def _resolve_analysis_path(outfile: Path, analysis_out: str | None) -> Path:
    if analysis_out is not None:
        return Path(analysis_out)
    return Path(f"{outfile}.analysis.json")
