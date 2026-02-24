import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

import soundfile as soundfile_lib

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.core.algo_reverb import AlgoReverbEngine
from verbx.core.convolution_reverb import ConvolutionReverbEngine
from verbx.core.freeze import create_crossfaded_loop, freeze_generator
from verbx.io.audio import (
    iter_audio_blocks,
    read_audio,
    soft_limiter,
)


def process_pipeline(
    infile: Path,
    outfile: Path,
    engine_type: str = "algo",
    rt60: float = 2.0,
    wet: float = 0.5,
    dry: float = 0.5,
    repeat: int = 1,
    freeze: bool = False,
    start: float = 0.0,
    end: float = 0.0,
    analysis_out: Optional[str] = None,
    silent: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    impulse_response: Optional[str] = None,
):
    """
    Main processing pipeline.
    """
    # 1. Analyze Input (if requested)
    # We analyze the full file for stats? Or just stream?
    # Requirement: "Always produce analysis JSON for input and output unless --silent."
    # For long files, full analysis takes time.
    # We'll analyze input first.

    input_stats = {}
    analyzer = AudioAnalyzer()

    # Load basic info first
    try:
        # Just read the whole file for analysis for v0.1
        # If very large, this might be slow/heavy.
        # But we need stats.
        if not silent:
            # print("Analyzing input...")
            pass

        full_audio, sr = read_audio(infile)
        input_stats = analyzer.analyze(full_audio, sr)

        if analysis_out:
            in_json = Path(analysis_out).with_suffix(".input.json")
            with open(in_json, "w") as f:
                json.dump(input_stats, f, indent=2)

    except Exception as e:
        print(f"Input analysis failed: {e}")
        # Proceed with processing even if analysis fails?
        # Maybe we should load audio for processing differently (streaming).
        # We already loaded it. Use it if memory permits.
        # But for pipeline robustness on large files, we should stream processing.
        # We loaded it just for analysis.
        # If we loaded it, we can use it.
        # But let's assume we release memory if possible, or use it.
        pass

    # 2. Setup Engine
    if engine_type == "conv":
        engine = ConvolutionReverbEngine(
            impulse_response=impulse_response or "", wet=wet, dry=dry
        )
    else:
        engine = AlgoReverbEngine(rt60=rt60, wet=wet, dry=dry)

    # 3. Setup Input Source (Freeze or File)
    # If freeze, we use the loop generator.
    # If not, we stream from file.

    current_infile = infile

    # Logic for Repeat Chaining
    # We iterate N times.
    # Each time we process from current_infile to temp_outfile (or final outfile).

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / "temp.wav"

    try:
        for pass_idx in range(repeat):
            is_last_pass = pass_idx == repeat - 1
            target_outfile = outfile if is_last_pass else temp_path

            # Setup Input Iterator
            if freeze and pass_idx == 0:
                # First pass with freeze: generate from input file segment
                # We need to extract the segment once.
                # If we loaded full_audio, use it.
                # Else read segment?
                # We have full_audio from analysis step.
                # If not analyzed (silent?), we might need to read.
                if "full_audio" not in locals():
                    full_audio, sr = read_audio(infile)

                loop = create_crossfaded_loop(full_audio, sr, start, end)
                # How long to render?
                # Infinite? No, we need a duration.
                # Use input duration? Or explicit length?
                # User req: "Output audio = reverb of the frozen loop, length = (original length + tail-limit) OR a fixed --tail-limit."
                # We'll use original length for now.
                duration_samples = len(full_audio)
                source_iter = freeze_generator(loop)

                # We need to limit the generator
                # Helper to limit iterator
                def limit_iter(it, max_samples):
                    count = 0
                    for block in it:
                        if count >= max_samples:
                            break
                        yield block
                        count += len(block)

                input_iterator = limit_iter(source_iter, duration_samples)

            else:
                # Read from current_infile
                # If pass > 0, current_infile is temp_path (which was written in prev pass).
                # We use iter_audio_blocks
                input_iterator = iter_audio_blocks(current_infile, block_size=4096)

            # Open Output File
            # We need sr. If we read blocks, we don't get sr.
            # We assume sr matches input.
            if "sr" not in locals():
                _, sr = read_audio(current_infile)  # This reads whole file? No.
                # Use sf.info
                info = soundfile_lib.info(str(current_infile))
                sr = info.samplerate

            # Process Stream
            # We need to handle writing.
            # We can't use write_audio for streaming.
            # We use sf.SoundFile for writing.

            with soundfile_lib.SoundFile(
                str(target_outfile),
                mode="w",
                samplerate=sr,
                channels=2,
                subtype="FLOAT",
            ) as out_f:
                # engine.process returns blocks.
                # engine might change channels (mono -> stereo).
                # We force output to stereo for safety or check engine output.

                for block in input_iterator:
                    processed = engine.process(block, sr)

                    # Normalize/Limit for safety in repeat
                    # Soft limit peaks > 0dB
                    processed = soft_limiter(processed, threshold_dbfs=0.0)

                    # Write
                    out_f.write(processed)

                    if progress_callback:
                        progress_callback(len(block))

            # After pass, set current_infile to target_outfile (which is temp_path or outfile)
            if not is_last_pass:
                # Move temp_path to a new temp name to avoid overwrite issues reading/writing same file?
                # We wrote to temp_path.
                # Next pass reads from temp_path.
                # But we overwrite temp_path in next pass?
                # We need 2 temp files to ping-pong.
                next_temp = temp_path.with_name(f"temp_{pass_idx}.wav")
                shutil.move(str(temp_path), str(next_temp))
                current_infile = next_temp

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 4. Analyze Output
    if not silent:
        try:
            out_audio, out_sr = read_audio(outfile)
            out_stats = analyzer.analyze(out_audio, out_sr)

            if analysis_out:
                out_json = Path(analysis_out).with_suffix(".output.json")
                with open(out_json, "w") as f:
                    json.dump(out_stats, f, indent=2)

            # Print summary? CLI handles that.
            return out_stats
        except Exception as e:
            print(f"Output analysis failed: {e}")
            return {}

    return {}
