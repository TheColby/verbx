#include "verbx_c/plugin_realtime.h"

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

    verbx_plugin_realtime_release(&context);

    config.channel_count = 0U;
    memset(error, 0, sizeof(error));
    if (require_true(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) != 0, "prepare should reject zero channels") != 0) {
        return 1;
    }
    if (require_string_contains(error, "channel_count", "zero channel error should name channel_count") != 0) {
        return 1;
    }

    return 0;
}
