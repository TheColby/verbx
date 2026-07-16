#include "verbx_c/plugin_realtime.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int fail_message(const char *message) {
    fprintf(stderr, "test_plugin_realtime: %s\n", message);
    return 1;
}

static int require_true(int condition, const char *message) {
    if (!condition) {
        return fail_message(message);
    }

    return 0;
}

static int require_close(double actual, double expected, double tolerance, const char *message) {
    if (fabs(actual - expected) > tolerance) {
        fprintf(
            stderr,
            "test_plugin_realtime: %s (actual=%0.17g expected=%0.17g tolerance=%0.17g)\n",
            message,
            actual,
            expected,
            tolerance
        );
        return 1;
    }

    return 0;
}

static int require_string_contains(const char *actual, const char *expected, const char *message) {
    if ((actual == NULL) || (expected == NULL) || (strstr(actual, expected) == NULL)) {
        fprintf(
            stderr,
            "test_plugin_realtime: %s (actual=%s expected substring=%s)\n",
            message,
            actual != NULL ? actual : "(null)",
            expected != NULL ? expected : "(null)"
        );
        return 1;
    }

    return 0;
}

int main(void) {
    verbx_plugin_realtime_context context;
    verbx_plugin_realtime_config config;
    verbx_plugin_realtime_params params;
    verbx_plugin_realtime_status status;
    char error[256];
    float in_l[4] = {0.25f, -0.50f, 0.75f, -1.0f};
    float in_r[4] = {-0.25f, 0.50f, -0.75f, 1.0f};
    float out_l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out_r[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float *inputs[2] = {in_l, in_r};
    float *outputs[2] = {out_l, out_r};
    size_t frame;
    double wet_energy = 0.0;

    memset(&context, 0, sizeof(context));
    memset(&config, 0, sizeof(config));
    memset(&params, 0, sizeof(params));
    memset(&status, 0, sizeof(status));
    memset(error, 0, sizeof(error));

    config.host_sample_rate = 48000U;
    config.max_block_frames = 512U;
    config.channel_count = 2U;
    config.quality_mode = VERBX_PLUGIN_QUALITY_TARGET_192K;

    params.rt60_coarse_normalized = 0.5;
    params.rt60_fine_bipolar = 1.0;
    params.dry = 1.0;
    params.wet = 0.0;
    params.freeze = 1;
    params.reverse = 1;

    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0, "prepare should succeed") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_internal_sample_rate(&context) == 192000U, "internal sample rate mismatch") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_oversampling_factor(&context) == 4U, "target oversampling factor mismatch") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_latency_frames(&context) == 0U, "latency frames mismatch") != 0) {
        return 1;
    }

    if (require_true(verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 2U, &params, &status) == 0, "process should succeed") != 0) {
        return 1;
    }
    for (frame = 0U; frame < 4U; ++frame) {
        if (require_true(out_l[frame] == in_l[frame], "left output should pass through input") != 0) {
            return 1;
        }
        if (require_true(out_r[frame] == in_r[frame], "right output should pass through input") != 0) {
            return 1;
        }
    }
    if (require_close(status.effective_rt60_seconds, sqrt(0.01 * 360.0) * 1.20, 1e-12, "effective rt60 mismatch") != 0) {
        return 1;
    }
    if (require_true(status.freeze_enabled == 1, "freeze status mismatch") != 0) {
        return 1;
    }
    if (require_true(status.reverse_enabled == 1, "reverse status mismatch") != 0) {
        return 1;
    }
    if (require_true(status.quality_mode == VERBX_PLUGIN_QUALITY_TARGET_192K, "quality mode status mismatch") != 0) {
        return 1;
    }
    if (require_true(status.host_sample_rate == 48000U, "host sample rate status mismatch") != 0) {
        return 1;
    }
    if (require_true(status.internal_sample_rate == 192000U, "internal sample rate status mismatch") != 0) {
        return 1;
    }
    if (require_true(status.latency_frames == 0U, "latency status mismatch") != 0) {
        return 1;
    }
    if (require_true(status.oversampling_factor == 4U, "status oversampling factor mismatch") != 0) {
        return 1;
    }

    verbx_plugin_realtime_reset(&context);
    memset(in_l, 0, sizeof(in_l));
    memset(in_r, 0, sizeof(in_r));
    params.pre_delay_ms = 0.0;
    params.room_size = 0.72;
    params.rt60_coarse_normalized = 0.5;
    params.rt60_fine_bipolar = 0.0;
    params.damping = 0.41;
    params.diffusion = 0.65;
    params.width = 1.35;
    params.dry = 0.0;
    params.wet = 1.0;
    params.freeze = 0;
    params.reverse = 0;
    for (frame = 0U; frame < 4096U; frame += 4U) {
        size_t local;
        in_l[0] = frame == 0U ? 1.0f : 0.0f;
        in_r[0] = frame == 0U ? 1.0f : 0.0f;
        in_l[1] = in_l[2] = in_l[3] = 0.0f;
        in_r[1] = in_r[2] = in_r[3] = 0.0f;
        if (require_true(
                verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 2U, &params, &status) == 0,
                "wet impulse processing should succeed"
            ) != 0) {
            return 1;
        }
        for (local = 0U; local < 4U; ++local) {
            wet_energy += fabs((double)out_l[local]) + fabs((double)out_r[local]);
        }
    }
    if (require_true(wet_energy > 0.01, "wet impulse should produce a reverb tail") != 0) {
        return 1;
    }

    verbx_plugin_realtime_reset(&context);
    memset(in_l, 0, sizeof(in_l));
    memset(in_r, 0, sizeof(in_r));
    if (require_true(
            verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 2U, &params, &status) == 0,
            "processing after reset should succeed"
        ) != 0) {
        return 1;
    }
    for (frame = 0U; frame < 4U; ++frame) {
        if (require_close(out_l[frame], 0.0, 1e-12, "reset should clear left reverb state") != 0) {
            return 1;
        }
        if (require_close(out_r[frame], 0.0, 1e-12, "reset should clear right reverb state") != 0) {
            return 1;
        }
    }

    params.dry = 1.0;
    params.wet = 0.0;
    if (require_true(verbx_plugin_realtime_process(&context, inputs, outputs, 513U, 2U, &params, &status) != 0, "process should reject oversized blocks") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 0U, &params, &status) != 0, "process should reject zero channels") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 2U, NULL, &status) != 0, "process should reject null params") != 0) {
        return 1;
    }
    inputs[1] = NULL;
    if (require_true(verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 2U, &params, &status) != 0, "process should reject null channel pointers") != 0) {
        return 1;
    }
    inputs[1] = in_r;

    verbx_plugin_realtime_release(&context);
    if (require_true(verbx_plugin_realtime_internal_sample_rate(&context) == 0U, "released context internal sample rate should reset") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_latency_frames(&context) == 0U, "released context latency should reset") != 0) {
        return 1;
    }
    if (require_true(context.prepared == 0, "released context should clear prepared flag") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_internal_sample_rate(NULL) == 0U, "null context internal sample rate should be zero") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_latency_frames(NULL) == 0U, "null context latency should be zero") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_oversampling_factor(NULL) == 0U, "null context oversampling factor should be zero") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 2U, &params, &status) != 0, "process should reject an unprepared context") != 0) {
        return 1;
    }

    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(&context, 0, error, sizeof(error)) != 0, "prepare should reject null config") != 0) {
        return 1;
    }
    if (require_string_contains(error, "config", "null config error should name config") != 0) {
        return 1;
    }

    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(0, &config, error, sizeof(error)) != 0, "prepare should reject null context") != 0) {
        return 1;
    }
    if (require_string_contains(error, "context", "null context error should name context") != 0) {
        return 1;
    }

    config.channel_count = 0U;
    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) != 0, "prepare should reject zero channels") != 0) {
        return 1;
    }
    if (require_string_contains(error, "channel_count", "zero channel error should name channel_count") != 0) {
        return 1;
    }

    config.channel_count = 2U;
    config.host_sample_rate = 0U;
    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) != 0, "prepare should reject zero sample rate") != 0) {
        return 1;
    }
    if (require_string_contains(error, "host_sample_rate", "zero sample rate error should name host_sample_rate") != 0) {
        return 1;
    }

    config.host_sample_rate = 48000U;
    config.max_block_frames = 0U;
    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) != 0, "prepare should reject zero max block") != 0) {
        return 1;
    }
    if (require_string_contains(error, "max_block_frames", "zero max block error should name max_block_frames") != 0) {
        return 1;
    }

    config.max_block_frames = 512U;
    config.quality_mode = VERBX_PLUGIN_QUALITY_HOST;
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0, "host quality prepare should succeed") != 0) {
        return 1;
    }
    if (require_true(context.internal_sample_rate == 48000U, "host quality rate mismatch") != 0) {
        return 1;
    }
    if (require_true(context.oversampling_factor == 1U, "host quality factor mismatch") != 0) {
        return 1;
    }

    config.quality_mode = VERBX_PLUGIN_QUALITY_2X;
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0, "2x quality prepare should succeed") != 0) {
        return 1;
    }
    if (require_true(context.internal_sample_rate == 96000U, "2x quality rate mismatch") != 0) {
        return 1;
    }
    if (require_true(context.oversampling_factor == 2U, "2x quality factor mismatch") != 0) {
        return 1;
    }

    config.quality_mode = VERBX_PLUGIN_QUALITY_4X;
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0, "4x quality prepare should succeed") != 0) {
        return 1;
    }
    if (require_true(context.internal_sample_rate == 192000U, "4x quality rate mismatch") != 0) {
        return 1;
    }
    if (require_true(context.oversampling_factor == 4U, "4x quality factor mismatch") != 0) {
        return 1;
    }

    config.host_sample_rate = 44100U;
    config.quality_mode = VERBX_PLUGIN_QUALITY_TARGET_192K;
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0, "44.1 kHz target quality prepare should succeed") != 0) {
        return 1;
    }
    if (require_true(context.internal_sample_rate == 220500U, "44.1 kHz target quality rate mismatch") != 0) {
        return 1;
    }
    if (require_true(context.oversampling_factor == 5U, "44.1 kHz target quality factor mismatch") != 0) {
        return 1;
    }

    config.host_sample_rate = 384000U;
    config.quality_mode = VERBX_PLUGIN_QUALITY_TARGET_192K;
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0, "high host rate prepare should succeed") != 0) {
        return 1;
    }
    if (require_true(context.internal_sample_rate == 384000U, "target quality should preserve a higher host rate") != 0) {
        return 1;
    }
    if (require_true(context.oversampling_factor == 1U, "target quality should not downsample a higher host rate") != 0) {
        return 1;
    }

    config.host_sample_rate = 48000U;
    config.quality_mode = (verbx_plugin_quality_mode)999;
    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) != 0, "prepare should reject invalid quality") != 0) {
        return 1;
    }
    if (require_string_contains(error, "quality_mode", "invalid quality error should name quality_mode") != 0) {
        return 1;
    }

    config.host_sample_rate = UINT_MAX;
    config.quality_mode = VERBX_PLUGIN_QUALITY_4X;
    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) != 0, "prepare should reject internal-rate overflow") != 0) {
        return 1;
    }
    if (require_string_contains(error, "overflow", "overflow error should identify overflow") != 0) {
        return 1;
    }

    return 0;
}
