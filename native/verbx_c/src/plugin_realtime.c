#include "verbx_c/plugin_realtime.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VERBX_PLUGIN_COMB_COUNT = 4,
    VERBX_PLUGIN_ALLPASS_COUNT = 2,
    VERBX_PLUGIN_MAX_CHANNELS = 2,
    VERBX_PLUGIN_TARGET_SAMPLE_RATE = 192000
};

typedef struct {
    float *buffer;
    size_t capacity;
    size_t index;
    float filter_state;
} verbx_plugin_comb;

typedef struct {
    float *buffer;
    size_t capacity;
    size_t index;
} verbx_plugin_allpass;

typedef struct {
    float *predelay;
    size_t predelay_capacity;
    size_t predelay_index;
    verbx_plugin_comb combs[VERBX_PLUGIN_COMB_COUNT];
    verbx_plugin_allpass allpasses[VERBX_PLUGIN_ALLPASS_COUNT];
} verbx_plugin_channel;

typedef struct {
    verbx_plugin_channel channels[VERBX_PLUGIN_MAX_CHANNELS];
    double reverse_envelope;
    double previous_input_level;
    double pre_delay_ms;
    double room_scale;
    double rt60;
    double damping;
    double diffusion;
    double width;
    double wet_mix;
    double dry_mix;
    float previous_input[VERBX_PLUGIN_MAX_CHANNELS];
    int smoothing_initialized;
} verbx_plugin_dsp_state;

static const double VERBX_PLUGIN_COMB_MS[VERBX_PLUGIN_COMB_COUNT] = {29.7, 37.1, 41.1, 43.7};
static const double VERBX_PLUGIN_ALLPASS_MS[VERBX_PLUGIN_ALLPASS_COUNT] = {5.0, 1.7};

static void set_error(char *error_message, size_t error_message_size, const char *message) {
    if ((error_message == 0) || (error_message_size == 0U)) {
        return;
    }
    snprintf(error_message, error_message_size, "%s", message);
}

static unsigned int internal_rate_for_quality(
    unsigned int host_rate,
    verbx_plugin_quality_mode quality_mode,
    size_t *oversampling_factor
) {
    size_t factor = 1U;
    if (quality_mode == VERBX_PLUGIN_QUALITY_2X) {
        factor = 2U;
    } else if (quality_mode == VERBX_PLUGIN_QUALITY_4X) {
        factor = 4U;
    } else if ((quality_mode == VERBX_PLUGIN_QUALITY_TARGET_192K)
               && (host_rate < VERBX_PLUGIN_TARGET_SAMPLE_RATE)) {
        factor = ((size_t)VERBX_PLUGIN_TARGET_SAMPLE_RATE + (size_t)host_rate - 1U)
            / (size_t)host_rate;
    }
    if (oversampling_factor != NULL) {
        *oversampling_factor = factor;
    }
    return host_rate * (unsigned int)factor;
}

static size_t frames_for_ms(unsigned int sample_rate, double milliseconds, double scale) {
    double frames = ((double)sample_rate * milliseconds * scale) / 1000.0;
    if (frames < 1.0) {
        frames = 1.0;
    }
    return (size_t)llround(frames);
}

static int allocate_float_buffer(float **buffer, size_t frames) {
    *buffer = (float *)calloc(frames, sizeof(float));
    return *buffer == NULL ? -1 : 0;
}

static void free_dsp_state(verbx_plugin_dsp_state *state) {
    size_t channel;
    size_t index;
    if (state == NULL) {
        return;
    }
    for (channel = 0U; channel < VERBX_PLUGIN_MAX_CHANNELS; ++channel) {
        free(state->channels[channel].predelay);
        for (index = 0U; index < VERBX_PLUGIN_COMB_COUNT; ++index) {
            free(state->channels[channel].combs[index].buffer);
        }
        for (index = 0U; index < VERBX_PLUGIN_ALLPASS_COUNT; ++index) {
            free(state->channels[channel].allpasses[index].buffer);
        }
    }
    free(state);
}

static verbx_plugin_dsp_state *create_dsp_state(unsigned int sample_rate, size_t channels) {
    verbx_plugin_dsp_state *state;
    size_t channel;
    size_t index;
    state = (verbx_plugin_dsp_state *)calloc(1U, sizeof(*state));
    if (state == NULL) {
        return NULL;
    }
    state->reverse_envelope = 1.0;
    for (channel = 0U; channel < channels; ++channel) {
        double stereo_scale = channel == 0U ? 1.0 : 1.09;
        verbx_plugin_channel *current = &state->channels[channel];
        current->predelay_capacity = frames_for_ms(sample_rate, 1000.0, 1.0) + 1U;
        if (allocate_float_buffer(&current->predelay, current->predelay_capacity) != 0) {
            free_dsp_state(state);
            return NULL;
        }
        for (index = 0U; index < VERBX_PLUGIN_COMB_COUNT; ++index) {
            current->combs[index].capacity = frames_for_ms(
                sample_rate,
                VERBX_PLUGIN_COMB_MS[index],
                stereo_scale * 2.0
            ) + 1U;
            if (allocate_float_buffer(
                    &current->combs[index].buffer,
                    current->combs[index].capacity
                ) != 0) {
                free_dsp_state(state);
                return NULL;
            }
        }
        for (index = 0U; index < VERBX_PLUGIN_ALLPASS_COUNT; ++index) {
            current->allpasses[index].capacity = frames_for_ms(
                sample_rate,
                VERBX_PLUGIN_ALLPASS_MS[index],
                stereo_scale * 2.0
            ) + 1U;
            if (allocate_float_buffer(
                    &current->allpasses[index].buffer,
                    current->allpasses[index].capacity
                ) != 0) {
                free_dsp_state(state);
                return NULL;
            }
        }
    }
    return state;
}

static float process_predelay(verbx_plugin_channel *channel, float input, size_t length) {
    float output;
    if (length == 0U) {
        return input;
    }
    if (channel->predelay_index >= length) {
        channel->predelay_index = 0U;
    }
    output = channel->predelay[channel->predelay_index];
    channel->predelay[channel->predelay_index] = input;
    channel->predelay_index = (channel->predelay_index + 1U) % length;
    return output;
}

static float process_comb(
    verbx_plugin_comb *comb,
    float input,
    size_t length,
    double feedback,
    double damping
) {
    float output;
    if (comb->index >= length) {
        comb->index = 0U;
    }
    output = comb->buffer[comb->index];
    comb->filter_state = (float)(
        ((double)output * (1.0 - damping)) + ((double)comb->filter_state * damping)
    );
    comb->buffer[comb->index] = input + (comb->filter_state * (float)feedback);
    comb->index = (comb->index + 1U) % length;
    return output;
}

static float process_allpass(
    verbx_plugin_allpass *allpass,
    float input,
    size_t length,
    double gain
) {
    float buffered;
    float output;
    if (allpass->index >= length) {
        allpass->index = 0U;
    }
    buffered = allpass->buffer[allpass->index];
    output = -input + buffered;
    allpass->buffer[allpass->index] = input + (buffered * (float)gain);
    allpass->index = (allpass->index + 1U) % length;
    return output;
}

static float process_wet_channel(
    verbx_plugin_channel *channel,
    float input,
    unsigned int sample_rate,
    double stereo_scale,
    double room_scale,
    double rt60,
    double pre_delay_ms,
    double damping,
    double diffusion,
    int freeze
) {
    float delayed;
    float wet = 0.0f;
    size_t index;
    size_t predelay_length = pre_delay_ms <= 0.0
        ? 0U
        : frames_for_ms(sample_rate, pre_delay_ms, 1.0);
    if (predelay_length >= channel->predelay_capacity) {
        predelay_length = channel->predelay_capacity - 1U;
    }
    delayed = process_predelay(channel, freeze ? 0.0f : input, predelay_length);
    for (index = 0U; index < VERBX_PLUGIN_COMB_COUNT; ++index) {
        size_t length = frames_for_ms(
            sample_rate,
            VERBX_PLUGIN_COMB_MS[index],
            stereo_scale * room_scale
        );
        double delay_seconds;
        double feedback;
        if (length >= channel->combs[index].capacity) {
            length = channel->combs[index].capacity - 1U;
        }
        delay_seconds = (double)length / (double)sample_rate;
        feedback = freeze ? 0.9995 : pow(10.0, (-3.0 * delay_seconds) / rt60);
        if (feedback > 0.9995) {
            feedback = 0.9995;
        }
        wet += process_comb(
            &channel->combs[index],
            delayed,
            length,
            feedback,
            damping
        );
    }
    wet /= (float)VERBX_PLUGIN_COMB_COUNT;
    for (index = 0U; index < VERBX_PLUGIN_ALLPASS_COUNT; ++index) {
        size_t length = frames_for_ms(
            sample_rate,
            VERBX_PLUGIN_ALLPASS_MS[index],
            stereo_scale * room_scale
        );
        double gain = 0.20 + (0.55 * diffusion) - (0.04 * (double)index);
        if (length >= channel->allpasses[index].capacity) {
            length = channel->allpasses[index].capacity - 1U;
        }
        wet = process_allpass(&channel->allpasses[index], wet, length, gain);
    }
    return wet;
}

int verbx_plugin_realtime_prepare(
    verbx_plugin_realtime_context *context,
    const verbx_plugin_realtime_config *config,
    char *error_message,
    size_t error_message_size
) {
    verbx_plugin_dsp_state *dsp_state;
    unsigned int internal_sample_rate;
    size_t oversampling_factor;
    if (context == 0) {
        set_error(error_message, error_message_size, "context must be non-null");
        return -1;
    }
    if (config == 0) {
        set_error(error_message, error_message_size, "config must be non-null");
        return -1;
    }
    if (config->host_sample_rate == 0U) {
        set_error(error_message, error_message_size, "host_sample_rate must be non-zero");
        return -1;
    }
    if (config->max_block_frames == 0U) {
        set_error(error_message, error_message_size, "max_block_frames must be non-zero");
        return -1;
    }
    if (config->channel_count == 0U) {
        set_error(error_message, error_message_size, "channel_count must be non-zero");
        return -1;
    }
    if ((config->quality_mode < VERBX_PLUGIN_QUALITY_HOST)
        || (config->quality_mode > VERBX_PLUGIN_QUALITY_TARGET_192K)) {
        set_error(error_message, error_message_size, "quality_mode is invalid");
        return -1;
    }
    if (((config->quality_mode == VERBX_PLUGIN_QUALITY_2X)
         && (config->host_sample_rate > (UINT_MAX / 2U)))
        || ((config->quality_mode == VERBX_PLUGIN_QUALITY_4X)
            && (config->host_sample_rate > (UINT_MAX / 4U)))) {
        set_error(error_message, error_message_size, "internal sample rate would overflow");
        return -1;
    }
    if (config->channel_count > VERBX_PLUGIN_MAX_CHANNELS) {
        set_error(error_message, error_message_size, "realtime DSP supports mono/stereo only");
        return -1;
    }

    if ((context->prepared == 1) && (context->dsp_state != NULL)) {
        free_dsp_state((verbx_plugin_dsp_state *)context->dsp_state);
        context->dsp_state = NULL;
        context->prepared = 0;
    }
    internal_sample_rate = internal_rate_for_quality(
        config->host_sample_rate,
        config->quality_mode,
        &oversampling_factor
    );
    dsp_state = create_dsp_state(internal_sample_rate, config->channel_count);
    if (dsp_state == NULL) {
        set_error(error_message, error_message_size, "failed to allocate realtime DSP state");
        return -1;
    }

    memset(context, 0, sizeof(*context));
    context->host_sample_rate = config->host_sample_rate;
    context->internal_sample_rate = internal_sample_rate;
    context->max_block_frames = config->max_block_frames;
    context->channel_count = config->channel_count;
    context->latency_frames = 0U;
    context->oversampling_factor = oversampling_factor;
    context->quality_mode = config->quality_mode;
    context->dsp_state = dsp_state;
    context->prepared = 1;
    return 0;
}

int verbx_plugin_realtime_process(
    verbx_plugin_realtime_context *context,
    const float *const *inputs,
    float *const *outputs,
    size_t frames,
    size_t channels,
    const verbx_plugin_realtime_params *params,
    verbx_plugin_realtime_status *status
) {
    verbx_plugin_dsp_state *dsp_state;
    double damping;
    double diffusion;
    double room_scale;
    double rt60;
    double width;
    double wet_mix;
    double dry_mix;
    double pre_delay_ms;
    double smoothing;
    size_t oversampling_factor;
    size_t channel;
    size_t frame;

    if ((context == 0) || (context->prepared == 0) || (inputs == 0) || (outputs == 0) || (params == 0)) {
        return -1;
    }
    if ((channels == 0U) || (channels > context->channel_count) || (frames > context->max_block_frames)) {
        return -1;
    }
    dsp_state = (verbx_plugin_dsp_state *)context->dsp_state;
    if (dsp_state == NULL) {
        return -1;
    }

    for (channel = 0U; channel < channels; ++channel) {
        if ((inputs[channel] == 0) || (outputs[channel] == 0)) {
            return -1;
        }
    }

    damping = verbx_plugin_clamp(params->damping, 0.0, 0.98);
    diffusion = verbx_plugin_clamp(params->diffusion, 0.0, 1.0);
    room_scale = 0.55 + (1.45 * verbx_plugin_clamp(params->room_size, 0.0, 1.0));
    rt60 = verbx_plugin_map_rt60_seconds(
        params->rt60_coarse_normalized,
        params->rt60_fine_bipolar
    );
    width = verbx_plugin_clamp(params->width, 0.0, 2.0);
    wet_mix = verbx_plugin_clamp(params->wet, 0.0, 1.0);
    dry_mix = verbx_plugin_clamp(params->dry, 0.0, 1.0);
    pre_delay_ms = verbx_plugin_clamp(params->pre_delay_ms, 0.0, 1000.0);
    oversampling_factor = context->oversampling_factor;
    if (oversampling_factor == 0U) {
        return -1;
    }
    smoothing = 1.0 - exp(-1.0 / (0.020 * (double)context->internal_sample_rate));

    for (frame = 0U; frame < frames; ++frame) {
        float current_input[VERBX_PLUGIN_MAX_CHANNELS] = {0.0f, 0.0f};
        double wet_accumulator[VERBX_PLUGIN_MAX_CHANNELS] = {0.0, 0.0};
        size_t subframe;
        for (channel = 0U; channel < channels; ++channel) {
            current_input[channel] = inputs[channel][frame];
        }

        for (subframe = 0U; subframe < oversampling_factor; ++subframe) {
            float oversampled_input[VERBX_PLUGIN_MAX_CHANNELS] = {0.0f, 0.0f};
            float wet_values[VERBX_PLUGIN_MAX_CHANNELS] = {0.0f, 0.0f};
            double input_level = 0.0;
            double position = (double)(subframe + 1U) / (double)oversampling_factor;

            if (dsp_state->smoothing_initialized == 0) {
                dsp_state->pre_delay_ms = pre_delay_ms;
                dsp_state->room_scale = room_scale;
                dsp_state->rt60 = rt60;
                dsp_state->damping = damping;
                dsp_state->diffusion = diffusion;
                dsp_state->width = width;
                dsp_state->wet_mix = wet_mix;
                dsp_state->dry_mix = dry_mix;
                dsp_state->smoothing_initialized = 1;
            } else {
                dsp_state->pre_delay_ms += smoothing * (pre_delay_ms - dsp_state->pre_delay_ms);
                dsp_state->room_scale += smoothing * (room_scale - dsp_state->room_scale);
                dsp_state->rt60 += smoothing * (rt60 - dsp_state->rt60);
                dsp_state->damping += smoothing * (damping - dsp_state->damping);
                dsp_state->diffusion += smoothing * (diffusion - dsp_state->diffusion);
                dsp_state->width += smoothing * (width - dsp_state->width);
                dsp_state->wet_mix += smoothing * (wet_mix - dsp_state->wet_mix);
                dsp_state->dry_mix += smoothing * (dry_mix - dsp_state->dry_mix);
            }

            for (channel = 0U; channel < channels; ++channel) {
                double stereo_scale = channel == 0U ? 1.0 : 1.09;
                oversampled_input[channel] = (float)(
                    (double)dsp_state->previous_input[channel]
                    + (((double)current_input[channel]
                        - (double)dsp_state->previous_input[channel]) * position)
                );
                if (fabs((double)oversampled_input[channel]) > input_level) {
                    input_level = fabs((double)oversampled_input[channel]);
                }
                wet_values[channel] = process_wet_channel(
                    &dsp_state->channels[channel],
                    oversampled_input[channel],
                    context->internal_sample_rate,
                    stereo_scale,
                    dsp_state->room_scale,
                    dsp_state->rt60,
                    dsp_state->pre_delay_ms,
                    dsp_state->damping,
                    dsp_state->diffusion,
                    params->freeze ? 1 : 0
                );
            }

            if (params->reverse) {
                double attack_samples = (double)context->internal_sample_rate * 0.25;
                if ((input_level >= 0.15) && (dsp_state->previous_input_level < 0.15)) {
                    dsp_state->reverse_envelope = 0.0;
                }
                dsp_state->reverse_envelope += 1.0 / attack_samples;
                if (dsp_state->reverse_envelope > 1.0) {
                    dsp_state->reverse_envelope = 1.0;
                }
            } else {
                dsp_state->reverse_envelope = 1.0;
            }
            dsp_state->previous_input_level = input_level;

            if (channels == 2U) {
                double mid = 0.5 * ((double)wet_values[0] + (double)wet_values[1]);
                double side = 0.5 * ((double)wet_values[0] - (double)wet_values[1])
                    * dsp_state->width;
                wet_values[0] = (float)(mid + side);
                wet_values[1] = (float)(mid - side);
            }
            for (channel = 0U; channel < channels; ++channel) {
                wet_accumulator[channel] += dsp_state->wet_mix
                    * (double)wet_values[channel]
                    * dsp_state->reverse_envelope;
            }
        }

        for (channel = 0U; channel < channels; ++channel) {
            double mixed = (dsp_state->dry_mix * (double)current_input[channel])
                + (wet_accumulator[channel] / (double)oversampling_factor);
            outputs[channel][frame] = (float)verbx_plugin_clamp(mixed, -4.0, 4.0);
            dsp_state->previous_input[channel] = current_input[channel];
        }
    }

    if (status != 0) {
        status->host_sample_rate = context->host_sample_rate;
        status->internal_sample_rate = context->internal_sample_rate;
        status->latency_frames = context->latency_frames;
        status->oversampling_factor = context->oversampling_factor;
        status->effective_rt60_seconds = verbx_plugin_map_rt60_seconds(
            params->rt60_coarse_normalized,
            params->rt60_fine_bipolar
        );
        status->quality_mode = context->quality_mode;
        status->freeze_enabled = params->freeze ? 1 : 0;
        status->reverse_enabled = params->reverse ? 1 : 0;
    }

    return 0;
}

void verbx_plugin_realtime_reset(verbx_plugin_realtime_context *context) {
    verbx_plugin_dsp_state *state;
    size_t channel;
    size_t index;
    if (context == 0) {
        return;
    }
    state = (verbx_plugin_dsp_state *)context->dsp_state;
    if (state == NULL) {
        return;
    }
    for (channel = 0U; channel < context->channel_count; ++channel) {
        verbx_plugin_channel *current = &state->channels[channel];
        memset(current->predelay, 0, current->predelay_capacity * sizeof(float));
        current->predelay_index = 0U;
        for (index = 0U; index < VERBX_PLUGIN_COMB_COUNT; ++index) {
            memset(
                current->combs[index].buffer,
                0,
                current->combs[index].capacity * sizeof(float)
            );
            current->combs[index].index = 0U;
            current->combs[index].filter_state = 0.0f;
        }
        for (index = 0U; index < VERBX_PLUGIN_ALLPASS_COUNT; ++index) {
            memset(
                current->allpasses[index].buffer,
                0,
                current->allpasses[index].capacity * sizeof(float)
            );
            current->allpasses[index].index = 0U;
        }
    }
    state->reverse_envelope = 1.0;
    state->previous_input_level = 0.0;
    memset(state->previous_input, 0, sizeof(state->previous_input));
    state->smoothing_initialized = 0;
}

void verbx_plugin_realtime_release(verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return;
    }
    free_dsp_state((verbx_plugin_dsp_state *)context->dsp_state);
    memset(context, 0, sizeof(*context));
}

size_t verbx_plugin_realtime_latency_frames(const verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return 0U;
    }
    return context->latency_frames;
}

size_t verbx_plugin_realtime_oversampling_factor(const verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return 0U;
    }
    return context->oversampling_factor;
}

unsigned int verbx_plugin_realtime_internal_sample_rate(const verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return 0U;
    }
    return context->internal_sample_rate;
}
