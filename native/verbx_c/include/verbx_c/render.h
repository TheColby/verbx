#ifndef VERBX_C_RENDER_H
#define VERBX_C_RENDER_H

#include "verbx_c/audio.h"

#include <stddef.h>

typedef struct {
    double rt60;
    double wet;
    double dry;
    double damping;
    double pre_delay_ms;
    double tail_threshold_db;
    double tail_hold_ms;
    verbx_wav_format out_format;
} verbx_render_options;

typedef struct {
    size_t input_frames;
    size_t output_frames;
    unsigned int sample_rate;
    unsigned short channels;
    verbx_wav_format out_format;
} verbx_render_report;

int verbx_render_file(
    const char *input_path,
    const char *output_path,
    const verbx_render_options *options,
    verbx_render_report *report,
    char *error_message,
    size_t error_message_size
);

#endif
