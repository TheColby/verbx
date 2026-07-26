#include "verbx_c/audio.h"

#include <stdlib.h>

void verbx_audio_buffer_free(verbx_audio_buffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    free(buffer->samples);
    buffer->samples = NULL;
    buffer->frames = 0;
    buffer->sample_rate = 0;
    buffer->channels = 0;
}
