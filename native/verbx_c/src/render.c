#include "verbx_c/render.h"
#include "verbx_c/algo_reverb.h"
#include "verbx_c/wav_io.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *error_message, size_t error_message_size, const char *message) {
    if ((error_message == NULL) || (error_message_size == 0U)) {
        return;
    }
    snprintf(error_message, error_message_size, "%s", message);
}

const char *verbx_status_code_name(verbx_status_code code) {
    switch (code) {
        case VERBX_STATUS_OK:
            return "ok";
        case VERBX_STATUS_INVALID_ARGUMENT:
            return "invalid_argument";
        case VERBX_STATUS_IO_ERROR:
            return "io_error";
        case VERBX_STATUS_DSP_ERROR:
            return "dsp_error";
    }
    return "unknown";
}

const char *verbx_tail_metric_name(verbx_tail_metric metric) {
    switch (metric) {
        case VERBX_TAIL_METRIC_PEAK:
            return "peak";
        case VERBX_TAIL_METRIC_RMS:
            return "rms";
    }
    return "unknown";
}

int verbx_render_file(
    const char *input_path,
    const char *output_path,
    const verbx_render_options *options,
    verbx_render_report *report,
    char *error_message,
    size_t error_message_size
) {
    verbx_audio_buffer input = {0};
    verbx_audio_buffer output = {0};

    if ((input_path == NULL) || (output_path == NULL) || (options == NULL) || (report == NULL)) {
        set_error(error_message, error_message_size, "invalid render request");
        if (report != NULL) {
            report->status_code = VERBX_STATUS_INVALID_ARGUMENT;
        }
        return -1;
    }
    if (verbx_wav_read(input_path, &input, error_message, error_message_size) != 0) {
        report->status_code = VERBX_STATUS_IO_ERROR;
        return -1;
    }
    if (verbx_algo_render(&input, options, &output, error_message, error_message_size) != 0) {
        report->status_code = VERBX_STATUS_DSP_ERROR;
        verbx_audio_buffer_free(&input);
        return -1;
    }
    if (verbx_wav_write(output_path, &output, options->out_format, error_message, error_message_size) != 0) {
        report->status_code = VERBX_STATUS_IO_ERROR;
        verbx_audio_buffer_free(&input);
        verbx_audio_buffer_free(&output);
        return -1;
    }

    report->input_frames = input.frames;
    report->output_frames = output.frames;
    report->sample_rate = output.sample_rate;
    report->channels = output.channels;
    report->out_format = options->out_format;
    report->tail_metric = options->tail_metric;
    report->status_code = VERBX_STATUS_OK;

    verbx_audio_buffer_free(&input);
    verbx_audio_buffer_free(&output);
    return 0;
}
