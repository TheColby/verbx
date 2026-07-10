#include "verbx_c/plugin_realtime.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error_message, size_t error_message_size, const char *message) {
    if ((error_message == 0) || (error_message_size == 0U)) {
        return;
    }
    snprintf(error_message, error_message_size, "%s", message);
}

static unsigned int internal_rate_for_quality(unsigned int host_rate, verbx_plugin_quality_mode quality_mode) {
    if (quality_mode == VERBX_PLUGIN_QUALITY_2X) {
        return host_rate * 2U;
    }
    if (quality_mode == VERBX_PLUGIN_QUALITY_4X) {
        return host_rate * 4U;
    }
    if (quality_mode == VERBX_PLUGIN_QUALITY_TARGET_192K) {
        return host_rate >= 192000U ? host_rate : 192000U;
    }
    return host_rate;
}

int verbx_plugin_realtime_prepare(
    verbx_plugin_realtime_context *context,
    const verbx_plugin_realtime_config *config,
    char *error_message,
    size_t error_message_size
) {
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

    memset(context, 0, sizeof(*context));
    context->host_sample_rate = config->host_sample_rate;
    context->internal_sample_rate = internal_rate_for_quality(config->host_sample_rate, config->quality_mode);
    context->max_block_frames = config->max_block_frames;
    context->channel_count = config->channel_count;
    context->latency_frames = 0U;
    context->quality_mode = config->quality_mode;
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
    size_t channel;
    size_t frame;

    if ((context == 0) || (context->prepared == 0) || (inputs == 0) || (outputs == 0) || (params == 0)) {
        return -1;
    }
    if ((channels == 0U) || (channels > context->channel_count) || (frames > context->max_block_frames)) {
        return -1;
    }

    for (channel = 0U; channel < channels; ++channel) {
        if ((inputs[channel] == 0) || (outputs[channel] == 0)) {
            return -1;
        }
        for (frame = 0U; frame < frames; ++frame) {
            outputs[channel][frame] = inputs[channel][frame];
        }
    }

    if (status != 0) {
        status->host_sample_rate = context->host_sample_rate;
        status->internal_sample_rate = context->internal_sample_rate;
        status->latency_frames = context->latency_frames;
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
    if (context == 0) {
        return;
    }
}

void verbx_plugin_realtime_release(verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return;
    }
    memset(context, 0, sizeof(*context));
}

size_t verbx_plugin_realtime_latency_frames(const verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return 0U;
    }
    return context->latency_frames;
}

unsigned int verbx_plugin_realtime_internal_sample_rate(const verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return 0U;
    }
    return context->internal_sample_rate;
}
