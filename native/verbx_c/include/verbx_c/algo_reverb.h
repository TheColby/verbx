#ifndef VERBX_C_ALGO_REVERB_H
#define VERBX_C_ALGO_REVERB_H

#include "verbx_c/render.h"

#include <stddef.h>

/*
 * Native FDN has no shimmer path.  Its model-derived padding therefore maps
 * directly to Python's non-shimmer algorithmic tail policy.
 */
double verbx_algo_resolved_tail_padding_seconds(const verbx_render_options *options);

int verbx_algo_render(
    const verbx_audio_buffer *input,
    const verbx_render_options *options,
    verbx_audio_buffer *output,
    char *error_message,
    size_t error_message_size
);

#endif
