#include "verbx_c/algo_reverb.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static verbx_render_options defaults(void) {
    verbx_render_options options = {0};
    options.rt60 = 0.3;
    options.wet = 1.0;
    options.damping = 0.3;
    options.tail_limit_seconds = 0.3;
    options.tail_threshold_db = -120.0;
    options.tail_hold_ms = 10.0;
    return options;
}

int main(void) {
    double samples[400] = {1.0};
    double wet[400], dry[400], damping[400], rt60[400];
    verbx_audio_buffer input = {0}, first = {0}, second = {0}, baseline = {0};
    verbx_render_options options = defaults();
    char error[256];
    size_t i;
    double delta = 0.0;
    input.samples = samples;
    input.frames = 400;
    input.channels = 1;
    input.sample_rate = 8000;
    REQUIRE(verbx_algo_render(&input, &options, &baseline, error, sizeof(error)) == 0);
    /* A constant RT60 lane must reproduce the static feedback coefficients. */
    for (i = 0; i < 400; ++i) rt60[i] = options.rt60;
    options.automation.rt60 = rt60;
    options.automation.frames = 400;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) == 0);
    REQUIRE(first.frames == baseline.frames);
    REQUIRE(memcmp(first.samples, baseline.samples, first.frames * sizeof(double)) == 0);
    verbx_audio_buffer_free(&first);
    for (i = 0; i < 400; ++i) rt60[i] = 0.1 + 0.2 * (double)i / 400.0;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) == 0);
    delta = 0.0;
    for (i = 0; i < first.frames && i < baseline.frames; ++i)
        delta += fabs(first.samples[i] - baseline.samples[i]);
    REQUIRE(delta > 0.001);
    verbx_audio_buffer_free(&first);
    options.automation.rt60 = NULL;
    options.automation.frames = 0;
    delta = 0.0;
    options.mod_depth_ms = 2.0;
    options.mod_rate_hz = 0.7;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) == 0);
    delta = 0.0;
    for (i = 0; i < first.frames && i < baseline.frames; ++i)
        delta += fabs(first.samples[i] - baseline.samples[i]);
    REQUIRE(delta > 0.001);
    verbx_audio_buffer_free(&first);
    options.mod_depth_ms = 0.0;
    options.rt60_low = 0.4;
    options.rt60_mid = 0.3;
    options.rt60_high = 0.15;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) == 0);
    delta = 0.0;
    for (i = 0; i < first.frames && i < baseline.frames; ++i)
        delta += fabs(first.samples[i] - baseline.samples[i]);
    REQUIRE(delta > 0.001);
    verbx_audio_buffer_free(&first);
    options.mod_depth_ms = 2.0;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) == 0);
    REQUIRE(verbx_algo_render(&input, &options, &second, error, sizeof(error)) == 0);
    REQUIRE(first.frames == second.frames);
    REQUIRE(memcmp(first.samples, second.samples, first.frames * sizeof(double)) == 0);
    for (i = 0; i < first.frames; ++i) {
        REQUIRE(isfinite(first.samples[i]));
        if (i < baseline.frames) delta += fabs(first.samples[i] - baseline.samples[i]);
    }
    REQUIRE(delta > 0.001);
    verbx_audio_buffer_free(&first);
    verbx_audio_buffer_free(&second);
    verbx_audio_buffer_free(&baseline);

    options = defaults();
    for (i = 0; i < 400; ++i) {
        samples[i] = 0.1;
        wet[i] = 0.0;
        dry[i] = (double)i / 400.0;
        damping[i] = 0.2 + 0.6 * (double)i / 400.0;
        rt60[i] = 0.2 + 0.2 * (double)i / 400.0;
    }
    options.automation.wet = wet;
    options.automation.dry = dry;
    options.automation.damping = damping;
    options.automation.rt60 = rt60;
    options.automation.frames = 400;
    REQUIRE(verbx_algo_resolved_tail_padding_seconds(&options) == 0.3);
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) == 0);
    /* Away from tail finalization's short fade, mixing follows each lane exactly. */
    for (i = 0; i < 350; ++i) REQUIRE(fabs(first.samples[i] - 0.1 * dry[i]) < 1e-12);
    verbx_audio_buffer_free(&first);
    rt60[2] = NAN;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) != 0);
    rt60[2] = 0.3;
    options.rt60_low = options.rt60_mid = options.rt60_high = 0.3;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) != 0);
    options = defaults();
    options.rt60_low = 0.2;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) != 0);
    options = defaults();
    options.mod_depth_ms = 21.0;
    REQUIRE(verbx_algo_render(&input, &options, &first, error, sizeof(error)) != 0);
    return 0;
}
