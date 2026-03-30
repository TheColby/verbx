#ifndef VERBX_C_AUDIO_H
#define VERBX_C_AUDIO_H

#include <stddef.h>

typedef struct {
    double *samples;
    size_t frames;
    unsigned int sample_rate;
    unsigned short channels;
} verbx_audio_buffer;

typedef enum {
    VERBX_WAV_FORMAT_PCM16 = 0,
    VERBX_WAV_FORMAT_FLOAT32 = 1,
    VERBX_WAV_FORMAT_FLOAT64 = 2
} verbx_wav_format;

void verbx_audio_buffer_free(verbx_audio_buffer *buffer);

#endif
