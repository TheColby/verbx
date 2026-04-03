#ifndef VERBX_C_RENDER_H
#define VERBX_C_RENDER_H

#include "verbx_c/audio.h"

#include <stddef.h>

typedef enum {
    VERBX_STATUS_OK = 0,
    VERBX_STATUS_INVALID_ARGUMENT = 1,
    VERBX_STATUS_IO_ERROR = 2,
    VERBX_STATUS_DSP_ERROR = 3
} verbx_status_code;

typedef enum {
    VERBX_TAIL_METRIC_PEAK = 0,
    VERBX_TAIL_METRIC_RMS = 1
} verbx_tail_metric;

typedef struct {
    double rt60;
    double wet;
    double dry;
    double damping;
    double pre_delay_ms;
    double tail_threshold_db;
    double tail_hold_ms;
    verbx_tail_metric tail_metric;
    verbx_wav_format out_format;
} verbx_render_options;

typedef struct {
    size_t input_frames;
    size_t output_frames;
    unsigned int sample_rate;
    unsigned short channels;
    verbx_wav_format out_format;
    verbx_tail_metric tail_metric;
    verbx_status_code status_code;
} verbx_render_report;

const char *verbx_status_code_name(verbx_status_code code);
const char *verbx_tail_metric_name(verbx_tail_metric metric);

int verbx_render_file(
    const char *input_path,
    const char *output_path,
    const verbx_render_options *options,
    verbx_render_report *report,
    char *error_message,
    size_t error_message_size
);

#endif
