#ifndef VERBX_C_WAV_IO_H
#define VERBX_C_WAV_IO_H

#include "verbx_c/audio.h"

#include <stddef.h>

const char *verbx_wav_format_name(verbx_wav_format format);

int verbx_wav_read(
    const char *path,
    verbx_audio_buffer *out,
    char *error_message,
    size_t error_message_size
);

int verbx_wav_write(
    const char *path,
    const verbx_audio_buffer *buffer,
    verbx_wav_format format,
    char *error_message,
    size_t error_message_size
);

#endif
