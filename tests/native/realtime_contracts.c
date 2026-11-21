/* Executable signal/lifecycle contracts, also run under ASan and UBSan in CI. */
#include "verbx_c/plugin_realtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAMES 65536U
#define BLOCK 257U
#define REQUIRE(condition, message) do { if (!(condition)) { \
    fprintf(stderr, "%s (line %d)\n", message, __LINE__); return 1; } } while (0)

static verbx_plugin_realtime_params defaults(int model) {
    verbx_plugin_realtime_params params = {0};
    params.room_size = 0.65;
    params.rt60_coarse_normalized = 0.5;
    params.damping = 0.4;
    params.width = 1.0;
    params.diffusion = 0.7;
    params.wet = 1.0;
    params.spring_tension = 0.5;
    params.plate_brightness = 0.5;
    params.reverb_model = (verbx_plugin_reverb_model)model;
    return params;
}

static int render(verbx_plugin_realtime_context *context, int model,
                  size_t block, int impulse, float *result) {
    float left[BLOCK] = {0}, right[BLOCK] = {0};
    float out_left[BLOCK], out_right[BLOCK];
    const float *inputs[2] = {left, right};
    float *outputs[2] = {out_left, out_right};
    verbx_plugin_realtime_params params = defaults(model);
    for (size_t start = 0; start < FRAMES; start += block) {
        size_t count = FRAMES - start < block ? FRAMES - start : block;
        memset(left, 0, sizeof(left));
        memset(right, 0, sizeof(right));
        if (start == 0 && impulse) left[0] = right[0] = 1.0f;
        REQUIRE(verbx_plugin_realtime_process(context, inputs, outputs, count, 2,
                                             &params, NULL) == 0, "render failed");
        for (size_t i = 0; i < count; ++i) {
            REQUIRE(isfinite(out_left[i]) && isfinite(out_right[i]), "nonfinite impulse");
            result[2 * (start + i)] = out_left[i];
            result[2 * (start + i) + 1] = out_right[i];
        }
    }
    return 0;
}

static int impulse_reset_contract(int model, int quality) {
    verbx_plugin_realtime_context context = {0};
    verbx_plugin_realtime_config config = {48000, BLOCK, 2,
                                           (verbx_plugin_quality_mode)quality};
    char error[256] = {0};
    float *reference = calloc(2 * FRAMES, sizeof(float));
    float *actual = calloc(2 * FRAMES, sizeof(float));
    double energy = 0.0;
    REQUIRE(reference != NULL && actual != NULL, "test allocation failed");
    REQUIRE(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0,
            error);
    REQUIRE(render(&context, model, BLOCK, 1, reference) == 0, "impulse failed");
    for (size_t i = 4096; i < 2 * FRAMES; ++i) energy += fabs(reference[i]);
    REQUIRE(energy > 0.001, "impulse has no late reverb tail");
    verbx_plugin_realtime_reset(&context);
    REQUIRE(render(&context, model, BLOCK, 0, actual) == 0, "reset silence failed");
    for (size_t i = 0; i < 2 * FRAMES; ++i)
        REQUIRE(actual[i] == 0.0f, "reset leaked delayed audio");
    verbx_plugin_realtime_reset(&context);
    REQUIRE(render(&context, model, 113, 1, actual) == 0, "partitioned impulse failed");
    for (size_t i = 0; i < 2 * FRAMES; ++i)
        REQUIRE(actual[i] == reference[i], "reset/block partition changed impulse");
    verbx_plugin_realtime_release(&context);
    verbx_plugin_realtime_release(&context);
    REQUIRE(context.dsp_state == NULL && context.prepared == 0, "release not idempotent");
    free(reference);
    free(actual);
    return 0;
}

static int automation_contract(void) {
    verbx_plugin_realtime_context context = {0};
    verbx_plugin_realtime_config config = {44100, BLOCK, 2, VERBX_PLUGIN_QUALITY_TARGET_192K};
    verbx_plugin_realtime_status status = {0};
    char error[256] = {0};
    float left[BLOCK], right[BLOCK], out_left[BLOCK], out_right[BLOCK];
    const float *inputs[2] = {left, right};
    float *outputs[2] = {out_left, out_right};
    REQUIRE(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0,
            error);
    for (size_t step = 0; step < 256; ++step) {
        verbx_plugin_realtime_params params = defaults((int)(step % 3));
        double target = step % 2 ? 1.0 : 0.0;
        size_t count = 1 + step % BLOCK;
        params.pre_delay_ms = target * 1000;
        params.room_size = params.damping = params.diffusion = target;
        params.rt60_coarse_normalized = target;
        params.rt60_fine_bipolar = target * 2 - 1;
        params.width = target * 2;
        params.wet = target;
        params.dry = 1 - target;
        params.spring_tension = params.plate_brightness = target;
        params.freeze = (int)(step % 2);
        params.reverse = (int)((step / 2) % 2);
        for (size_t i = 0; i < count; ++i) {
            left[i] = (float)(0.1 * sin((double)(step * BLOCK + i) * 0.03));
            right[i] = -left[i];
            out_left[i] = out_right[i] = NAN;
        }
        REQUIRE(verbx_plugin_realtime_process(&context, inputs, outputs, count, 2,
                                             &params, &status) == 0, "automation failed");
        REQUIRE(status.reverb_model == params.reverb_model, "model status stale");
        REQUIRE(status.freeze_enabled == params.freeze, "freeze status stale");
        REQUIRE(status.reverse_enabled == params.reverse, "reverse status stale");
        REQUIRE(isfinite(status.effective_rt60_seconds) && status.effective_rt60_seconds > 0,
                "invalid RT60 status");
        for (size_t i = 0; i < count; ++i)
            REQUIRE(isfinite(out_left[i]) && isfinite(out_right[i]) &&
                    fabs(out_left[i]) < 16 && fabs(out_right[i]) < 16,
                    "automation produced nonfinite or unbounded output");
    }
    verbx_plugin_realtime_release(&context);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "automation") == 0) return automation_contract();
    if (argc != 3) return 2;
    return impulse_reset_contract(atoi(argv[1]), atoi(argv[2]));
}
