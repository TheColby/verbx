#ifndef VERBX_C_PLUGIN_REALTIME_H
#define VERBX_C_PLUGIN_REALTIME_H

#include "verbx_c/plugin_params.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned int host_sample_rate;
    size_t max_block_frames;
    size_t channel_count;
    verbx_plugin_quality_mode quality_mode;
} verbx_plugin_realtime_config;

typedef struct {
    double pre_delay_ms;
    double room_size;
    double rt60_coarse_normalized;
    double rt60_fine_bipolar;
    double damping;
    double width;
    double diffusion;
    double wet;
    double dry;
    int freeze;
    int reverse;
} verbx_plugin_realtime_params;

typedef struct {
    unsigned int host_sample_rate;
    unsigned int internal_sample_rate;
    size_t latency_frames;
    double effective_rt60_seconds;
    verbx_plugin_quality_mode quality_mode;
    int freeze_enabled;
    int reverse_enabled;
} verbx_plugin_realtime_status;

typedef struct {
    unsigned int host_sample_rate;
    unsigned int internal_sample_rate;
    size_t max_block_frames;
    size_t channel_count;
    size_t latency_frames;
    verbx_plugin_quality_mode quality_mode;
    void *dsp_state;
    int prepared;
} verbx_plugin_realtime_context;

int verbx_plugin_realtime_prepare(
    verbx_plugin_realtime_context *context,
    const verbx_plugin_realtime_config *config,
    char *error_message,
    size_t error_message_size
);

int verbx_plugin_realtime_process(
    verbx_plugin_realtime_context *context,
    const float *const *inputs,
    float *const *outputs,
    size_t frames,
    size_t channels,
    const verbx_plugin_realtime_params *params,
    verbx_plugin_realtime_status *status
);

void verbx_plugin_realtime_reset(verbx_plugin_realtime_context *context);
void verbx_plugin_realtime_release(verbx_plugin_realtime_context *context);
size_t verbx_plugin_realtime_latency_frames(const verbx_plugin_realtime_context *context);
unsigned int verbx_plugin_realtime_internal_sample_rate(const verbx_plugin_realtime_context *context);

#ifdef __cplusplus
}
#endif

#endif
