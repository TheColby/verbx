from typing import Iterator

import numpy as np


def create_crossfaded_loop(
    audio: np.ndarray, sr: int, start: float, end: float, xfade_ms: float = 100.0
) -> np.ndarray:
    """
    Create a seamless loop from an audio segment using crossfading.

    Args:
        audio: Input audio array.
        sr: Sample rate.
        start: Start time (s).
        end: End time (s).
        xfade_ms: Crossfade duration (ms).

    Returns:
        The loop buffer (numpy array).
    """
    start_sample = int(start * sr)
    end_sample = int(end * sr)

    if start_sample < 0:
        start_sample = 0
    if end_sample > len(audio):
        end_sample = len(audio)

    if start_sample >= end_sample:
        # Fallback to silence or full audio
        return np.zeros_like(audio[:1024])

    segment = audio[start_sample:end_sample]

    xfade_samples = int(xfade_ms * sr / 1000.0)

    # If segment is too short for crossfade
    if len(segment) < 2 * xfade_samples:
        xfade_samples = len(segment) // 2

    if xfade_samples <= 0:
        return segment

    # Perform crossfade
    # Mix end of segment into start
    # Overlap-add style for loop?
    # Standard crossfade loop:
    # We want a loop of length L = len(segment) - xfade_samples.
    # The last xfade_samples overlap with the first xfade_samples.

    loop_len = len(segment) - xfade_samples
    loop = np.zeros((loop_len, segment.shape[1]), dtype=np.float32)

    # Main body (excluding fade region at end)
    loop[:loop_len] = segment[:loop_len]

    # Fade in/out region
    # The last xfade_samples of segment are faded out.
    # The first xfade_samples of segment are faded in (implied by loop start).
    # Wait, seamless loop means:
    # transition from end of loop back to start is smooth.
    # So we mix segment[-xfade:] with segment[:xfade].

    fade_in = np.linspace(0, 1, xfade_samples)[:, np.newaxis]
    fade_out = 1.0 - fade_in

    # The loop start (first xfade samples) is:
    # segment[:xfade] * fade_in + segment[-xfade:] * fade_out?
    # No, usually we want to preserve the transient at start if possible, or smooth it.
    # Let's say we overlap the end onto the start.

    loop[:xfade_samples] = (
        segment[:xfade_samples] * fade_in + segment[-xfade_samples:] * fade_out
    )

    # The rest of the loop is the middle of the segment
    loop[xfade_samples:] = segment[xfade_samples:loop_len]

    # Note: segment has length loop_len + xfade_samples.
    # loop has length loop_len.
    # We used segment[:loop_len] initially.
    # Then we modified loop[:xfade] by mixing segment[-xfade:] (which is segment[loop_len:]).

    return loop


def freeze_generator(
    loop_buffer: np.ndarray, block_size: int = 4096
) -> Iterator[np.ndarray]:
    """
    Yield infinite blocks from the loop buffer.
    """
    idx = 0
    loop_len = len(loop_buffer)

    while True:
        # If block fits
        if idx + block_size <= loop_len:
            yield loop_buffer[idx : idx + block_size]
            idx += block_size
            if idx >= loop_len:
                idx = 0
        else:
            # Wrap around
            part1 = loop_buffer[idx:]
            remaining = block_size - len(part1)
            # Handle case where remaining > loop_len (very short loop)
            # We construct block by tiling
            out_parts = [part1]
            current_rem = remaining

            while current_rem > 0:
                take = min(current_rem, loop_len)
                out_parts.append(loop_buffer[:take])
                current_rem -= take

            yield np.concatenate(out_parts, axis=0)

            # Update idx
            # We consumed 'remaining' from start of loop.
            idx = remaining % loop_len
