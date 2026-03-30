#include "verbx_c/algo_reverb.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VERBX_COMB_COUNT = 4,
    VERBX_ALLPASS_COUNT = 2
};

static const double VERBX_PI = 3.14159265358979323846;

typedef struct {
    double *buffer;
    size_t length;
    size_t index;
    double feedback;
    double filter_state;
} verbx_comb_state;

typedef struct {
    double *buffer;
    size_t length;
    size_t index;
    double gain;
} verbx_allpass_state;

typedef struct {
    double *predelay;
    size_t predelay_length;
    size_t predelay_index;
    verbx_comb_state combs[VERBX_COMB_COUNT];
    verbx_allpass_state allpasses[VERBX_ALLPASS_COUNT];
} verbx_channel_state;

static void set_error(char *error_message, size_t error_message_size, const char *message) {
    if ((error_message == NULL) || (error_message_size == 0U)) {
        return;
    }
    snprintf(error_message, error_message_size, "%s", message);
}

static size_t delay_from_ms(unsigned int sample_rate, double ms, double offset_scale) {
    double frames = ((double)sample_rate * ms * offset_scale) / 1000.0;
    if (frames < 1.0) {
        frames = 1.0;
    }
    return (size_t)llround(frames);
}

static int init_channel_state(
    verbx_channel_state *state,
    unsigned int sample_rate,
    double rt60,
    double pre_delay_ms,
    double stereo_scale
) {
    static const double comb_delay_ms[VERBX_COMB_COUNT] = {29.7, 37.1, 41.1, 43.7};
    static const double allpass_delay_ms[VERBX_ALLPASS_COUNT] = {5.0, 1.7};
    size_t i;

    memset(state, 0, sizeof(*state));
    state->predelay_length = delay_from_ms(sample_rate, pre_delay_ms, 1.0);
    state->predelay = (double *)calloc(state->predelay_length, sizeof(double));
    if (state->predelay == NULL) {
        return -1;
    }

    for (i = 0U; i < VERBX_COMB_COUNT; ++i) {
        double delay_ms = comb_delay_ms[i] * stereo_scale;
        double delay_seconds;
        state->combs[i].length = delay_from_ms(sample_rate, delay_ms, 1.0);
        state->combs[i].buffer = (double *)calloc(state->combs[i].length, sizeof(double));
        if (state->combs[i].buffer == NULL) {
            return -1;
        }
        delay_seconds = (double)state->combs[i].length / (double)sample_rate;
        state->combs[i].feedback = pow(10.0, (-3.0 * delay_seconds) / rt60);
        if (state->combs[i].feedback > 0.995) {
            state->combs[i].feedback = 0.995;
        }
        state->combs[i].filter_state = 0.0;
    }

    for (i = 0U; i < VERBX_ALLPASS_COUNT; ++i) {
        state->allpasses[i].length = delay_from_ms(sample_rate, allpass_delay_ms[i] * stereo_scale, 1.0);
        state->allpasses[i].buffer = (double *)calloc(state->allpasses[i].length, sizeof(double));
        if (state->allpasses[i].buffer == NULL) {
            return -1;
        }
        state->allpasses[i].gain = 0.7 - (0.05 * (double)i);
    }

    return 0;
}

static void free_channel_state(verbx_channel_state *state) {
    size_t i;
    free(state->predelay);
    state->predelay = NULL;
    for (i = 0U; i < VERBX_COMB_COUNT; ++i) {
        free(state->combs[i].buffer);
        state->combs[i].buffer = NULL;
    }
    for (i = 0U; i < VERBX_ALLPASS_COUNT; ++i) {
        free(state->allpasses[i].buffer);
        state->allpasses[i].buffer = NULL;
    }
}

static double process_comb(verbx_comb_state *state, double input, double damping) {
    double output = state->buffer[state->index];
    state->filter_state = (output * (1.0 - damping)) + (state->filter_state * damping);
    state->buffer[state->index] = input + (state->filter_state * state->feedback);
    state->index = (state->index + 1U) % state->length;
    return output;
}

static double process_allpass(verbx_allpass_state *state, double input) {
    double buffered = state->buffer[state->index];
    double output = (-input) + buffered;
    state->buffer[state->index] = input + (buffered * state->gain);
    state->index = (state->index + 1U) % state->length;
    return output;
}

static double process_channel(verbx_channel_state *state, double input, double damping) {
    size_t i;
    double delayed = state->predelay[state->predelay_index];
    double comb_sum = 0.0;
    double wet;

    state->predelay[state->predelay_index] = input;
    state->predelay_index = (state->predelay_index + 1U) % state->predelay_length;

    for (i = 0U; i < VERBX_COMB_COUNT; ++i) {
        comb_sum += process_comb(&state->combs[i], delayed, damping);
    }
    wet = comb_sum / (double)VERBX_COMB_COUNT;
    for (i = 0U; i < VERBX_ALLPASS_COUNT; ++i) {
        wet = process_allpass(&state->allpasses[i], wet);
    }
    return wet;
}

static double threshold_from_db(double threshold_db) {
    double clamped = threshold_db;
    if (clamped < -240.0) {
        clamped = -240.0;
    }
    if (clamped > 0.0) {
        clamped = 0.0;
    }
    return pow(10.0, clamped / 20.0);
}

static size_t hold_samples(unsigned int sample_rate, double hold_ms) {
    double frames = ((double)sample_rate * hold_ms) / 1000.0;
    if (frames < 1.0) {
        frames = 1.0;
    }
    return (size_t)llround(frames);
}

static size_t fade_samples(unsigned int sample_rate, size_t first_zero_frame) {
    double base = ((double)sample_rate * 3.0) / 1000.0;
    size_t base_frames = (size_t)llround(base < 1.0 ? 1.0 : base);
    size_t local_cap;
    if (first_zero_frame <= 1U) {
        return 1U;
    }
    local_cap = first_zero_frame / 4U;
    if (local_cap < 1U) {
        local_cap = 1U;
    }
    return base_frames < local_cap ? base_frames : local_cap;
}

static void apply_tail_fade_and_zero(
    const verbx_audio_buffer *rendered,
    verbx_audio_buffer *out,
    double threshold_db,
    double hold_ms
) {
    size_t frame;
    size_t channel;
    size_t last_active = 0U;
    int active_found = 0;
    double threshold = threshold_from_db(threshold_db);
    size_t hold = hold_samples(rendered->sample_rate, hold_ms);
    size_t first_zero_frame;
    size_t target_frames;
    size_t fade_len;
    size_t fade_start;

    for (frame = 0U; frame < rendered->frames; ++frame) {
        double peak = 0.0;
        for (channel = 0U; channel < rendered->channels; ++channel) {
            double value = fabs(rendered->samples[(frame * rendered->channels) + channel]);
            if (value > peak) {
                peak = value;
            }
        }
        if (peak > threshold) {
            last_active = frame;
            active_found = 1;
        }
    }

    if (!active_found) {
        target_frames = hold;
        out->samples = (double *)calloc(target_frames * rendered->channels, sizeof(double));
        out->frames = target_frames;
        out->sample_rate = rendered->sample_rate;
        out->channels = rendered->channels;
        return;
    }

    first_zero_frame = last_active + 1U;
    target_frames = first_zero_frame + hold;
    out->samples = (double *)calloc(target_frames * rendered->channels, sizeof(double));
    out->frames = target_frames;
    out->sample_rate = rendered->sample_rate;
    out->channels = rendered->channels;
    if (out->samples == NULL) {
        out->frames = 0U;
        return;
    }

    fade_len = fade_samples(rendered->sample_rate, first_zero_frame);
    fade_start = first_zero_frame > fade_len ? (first_zero_frame - fade_len) : 0U;
    if (fade_start > 0U) {
        memcpy(
            out->samples,
            rendered->samples,
            fade_start * rendered->channels * sizeof(double)
        );
    }
    for (frame = fade_start; frame < first_zero_frame; ++frame) {
        double ratio;
        double env;
        if (first_zero_frame - fade_start <= 1U) {
            env = 0.0;
        } else {
            ratio = (double)(frame - fade_start) / (double)((first_zero_frame - fade_start) - 1U);
            env = 0.5 * (1.0 + cos(VERBX_PI * ratio));
        }
        for (channel = 0U; channel < rendered->channels; ++channel) {
            size_t index = (frame * rendered->channels) + channel;
            out->samples[index] = rendered->samples[index] * env;
        }
    }
}

int verbx_algo_render(
    const verbx_audio_buffer *input,
    const verbx_render_options *options,
    verbx_audio_buffer *output,
    char *error_message,
    size_t error_message_size
) {
    verbx_channel_state states[2];
    verbx_audio_buffer rendered = {0};
    size_t max_tail_frames;
    size_t total_frames;
    size_t frame;
    size_t channel;
    double damping;
    double wet;
    double dry;
    double rt60;
    double pre_delay_ms;

    if ((input == NULL) || (input->samples == NULL) || (options == NULL) || (output == NULL)) {
        set_error(error_message, error_message_size, "invalid render arguments");
        return -1;
    }
    if ((input->channels < 1U) || (input->channels > 2U)) {
        set_error(error_message, error_message_size, "native algorithmic render supports mono/stereo only");
        return -1;
    }
    if (input->sample_rate == 0U) {
        set_error(error_message, error_message_size, "input sample rate must be non-zero");
        return -1;
    }

    memset(states, 0, sizeof(states));
    memset(output, 0, sizeof(*output));
    damping = options->damping;
    wet = options->wet;
    dry = options->dry;
    rt60 = options->rt60;
    pre_delay_ms = options->pre_delay_ms;

    if (rt60 < 0.1) {
        rt60 = 0.1;
    }
    if (wet < 0.0) {
        wet = 0.0;
    }
    if (dry < 0.0) {
        dry = 0.0;
    }
    if (damping < 0.0) {
        damping = 0.0;
    }
    if (damping > 0.98) {
        damping = 0.98;
    }
    if (pre_delay_ms < 0.0) {
        pre_delay_ms = 0.0;
    }
    max_tail_frames = (size_t)llround(
        (double)input->sample_rate *
        ((rt60 * 2.0) < 1.0 ? 1.0 : ((rt60 * 2.0) > 60.0 ? 60.0 : (rt60 * 2.0)))
    );
    total_frames = input->frames + max_tail_frames;

    rendered.samples = (double *)calloc(total_frames * input->channels, sizeof(double));
    if (rendered.samples == NULL) {
        set_error(error_message, error_message_size, "failed to allocate render buffer");
        return -1;
    }
    rendered.frames = total_frames;
    rendered.sample_rate = input->sample_rate;
    rendered.channels = input->channels;

    for (channel = 0U; channel < input->channels; ++channel) {
        double stereo_scale = (channel == 0U) ? 1.0 : 1.09;
        if (init_channel_state(
                &states[channel],
                input->sample_rate,
                rt60,
                pre_delay_ms,
                stereo_scale
            ) != 0) {
            set_error(error_message, error_message_size, "failed to initialize native reverb state");
            goto cleanup;
        }
    }

    for (frame = 0U; frame < total_frames; ++frame) {
        for (channel = 0U; channel < input->channels; ++channel) {
            double dry_in = 0.0;
            double wet_out;
            if (frame < input->frames) {
                dry_in = input->samples[(frame * input->channels) + channel];
            }
            wet_out = process_channel(&states[channel], dry_in, damping);
            rendered.samples[(frame * input->channels) + channel] =
                (dry * dry_in) + (wet * wet_out);
        }
    }

    apply_tail_fade_and_zero(&rendered, output, options->tail_threshold_db, options->tail_hold_ms);
    if (output->samples == NULL) {
        set_error(error_message, error_message_size, "failed to finalize native tail completion");
        goto cleanup;
    }

    for (channel = 0U; channel < input->channels; ++channel) {
        free_channel_state(&states[channel]);
    }
    verbx_audio_buffer_free(&rendered);
    return 0;

cleanup:
    for (channel = 0U; channel < input->channels; ++channel) {
        free_channel_state(&states[channel]);
    }
    verbx_audio_buffer_free(&rendered);
    return -1;
}
