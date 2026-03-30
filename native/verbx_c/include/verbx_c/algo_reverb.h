#ifndef VERBX_C_ALGO_REVERB_H
#define VERBX_C_ALGO_REVERB_H

#include "verbx_c/render.h"

#include <stddef.h>

int verbx_algo_render(
    const verbx_audio_buffer *input,
    const verbx_render_options *options,
    verbx_audio_buffer *output,
    char *error_message,
    size_t error_message_size
);

#endif
