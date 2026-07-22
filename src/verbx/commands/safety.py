"""Shared render/realtime safety estimates for CLI preflight output."""

from __future__ import annotations

from dataclasses import dataclass

from verbx.config import RenderConfig


@dataclass(frozen=True)
class PreflightItem:
    """One compact machine/human-readable preflight summary row."""

    key: str
    value: str


def algorithmic_render_selected(config: RenderConfig) -> bool:
    """Return true when a render config resolves to the algorithmic path."""

    engine = str(config.engine).strip().lower()
    has_conv_source = (
        config.ir is not None
        or bool(config.ir_gen)
        or bool(config.self_convolve)
        or len(config.ir_blend) > 0
    )
    return engine in {"algo", "ism-fdn"} or (engine == "auto" and not has_conv_source)


def estimate_algorithmic_tail_seconds(config: RenderConfig) -> float:
    """Estimate appended algorithmic tail duration for planning and safeguards."""

    tail_seconds = max(
        0.25,
        float(config.rt60) + (max(0.0, float(config.pre_delay_ms)) / 1000.0),
    )
    if config.tail_limit is not None:
        tail_seconds = min(tail_seconds, max(0.0, float(config.tail_limit)))
    return tail_seconds


def render_preflight_items(config: RenderConfig) -> tuple[PreflightItem, ...]:
    """Summarize render risk/behavior without touching audio."""

    if algorithmic_render_selected(config):
        tail_seconds = estimate_algorithmic_tail_seconds(config)
        if float(config.rt60) >= 600.0:
            profile = "extreme-tail"
        elif float(config.rt60) >= 60.0:
            profile = "long-tail"
        else:
            profile = "standard"
        tail_limit = (
            "unset"
            if config.tail_limit is None
            else f"{max(0.0, float(config.tail_limit)):.3f}s"
        )
        return (
            PreflightItem("render_path", "algorithmic"),
            PreflightItem("safety_profile", profile),
            PreflightItem("estimated_algo_tail_s", f"{tail_seconds:.3f}"),
            PreflightItem("tail_limit", tail_limit),
        )

    return (
        PreflightItem("render_path", "convolution"),
        PreflightItem("safety_profile", "input/IR bounded"),
    )


def realtime_freeze_proxy_required_seconds(config: RenderConfig) -> float:
    """Return the proxy-IR duration needed for the current freeze approximation."""

    return max(120.0, float(config.rt60) * 4.0)


def realtime_preflight_items(
    *,
    config: RenderConfig,
    live_mode: str,
    sample_rate: int,
    block_size: int,
    duration_seconds: float | None,
) -> tuple[PreflightItem, ...]:
    """Summarize realtime startup risk before opening the audio stream."""

    block_ms = 1000.0 * float(block_size) / float(max(1, sample_rate))
    items = [
        PreflightItem("live_mode", str(live_mode)),
        PreflightItem("block_latency", f"{block_ms:.1f}ms"),
    ]
    if duration_seconds is None:
        items.append(PreflightItem("duration", "until interrupted"))
    else:
        items.append(PreflightItem("duration", f"{float(duration_seconds):.3f}s"))

    if str(live_mode) != "dereverb" and algorithmic_render_selected(config):
        items.append(PreflightItem("render_path", "algorithmic proxy"))
        items.append(
            PreflightItem(
                "proxy_ir_budget",
                f"{float(config.algo_proxy_ir_max_seconds):.3f}s",
            )
        )
        if config.freeze:
            items.append(
                PreflightItem(
                    "freeze_proxy_required",
                    f"{realtime_freeze_proxy_required_seconds(config):.3f}s",
                )
            )
    elif str(live_mode) == "dereverb":
        items.append(PreflightItem("render_path", "dereverb only"))
    else:
        items.append(PreflightItem("render_path", "convolution"))

    return tuple(items)
