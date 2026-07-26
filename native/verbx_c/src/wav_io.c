#include "verbx_c/wav_io.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VERBX_WAV_PCM = 1,
    VERBX_WAV_IEEE_FLOAT = 3
};

static void set_error(char *error_message, size_t error_message_size, const char *message) {
    if ((error_message == NULL) || (error_message_size == 0U)) {
        return;
    }
    snprintf(error_message, error_message_size, "%s", message);
}

static void set_errorf(
    char *error_message,
    size_t error_message_size,
    const char *prefix,
    const char *path
) {
    if ((error_message == NULL) || (error_message_size == 0U)) {
        return;
    }
    snprintf(error_message, error_message_size, "%s: %s", prefix, path);
}

static uint16_t read_u16_le(const unsigned char *bytes) {
    return (uint16_t)(bytes[0] | (uint16_t)(bytes[1] << 8));
}

static uint32_t read_u32_le(const unsigned char *bytes) {
    return (uint32_t)(
        bytes[0] |
        (uint32_t)(bytes[1] << 8) |
        (uint32_t)(bytes[2] << 16) |
        (uint32_t)(bytes[3] << 24)
    );
}

static int32_t read_i24_le(const unsigned char *bytes) {
    int32_t value = (int32_t)(
        bytes[0] |
        (uint32_t)(bytes[1] << 8) |
        (uint32_t)(bytes[2] << 16)
    );
    if ((value & 0x00800000) != 0) {
        value |= (int32_t)0xFF000000;
    }
    return value;
}

static void write_u16_le(FILE *stream, uint16_t value) {
    unsigned char bytes[2];
    bytes[0] = (unsigned char)(value & 0xFFU);
    bytes[1] = (unsigned char)((value >> 8) & 0xFFU);
    fwrite(bytes, sizeof(unsigned char), 2U, stream);
}

static void write_u32_le(FILE *stream, uint32_t value) {
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xFFU);
    bytes[1] = (unsigned char)((value >> 8) & 0xFFU);
    bytes[2] = (unsigned char)((value >> 16) & 0xFFU);
    bytes[3] = (unsigned char)((value >> 24) & 0xFFU);
    fwrite(bytes, sizeof(unsigned char), 4U, stream);
}

static int read_exact(FILE *stream, void *buffer, size_t size) {
    return fread(buffer, 1U, size, stream) == size ? 0 : -1;
}

const char *verbx_wav_format_name(verbx_wav_format format) {
    switch (format) {
        case VERBX_WAV_FORMAT_PCM16:
            return "pcm16";
        case VERBX_WAV_FORMAT_FLOAT32:
            return "float32";
        case VERBX_WAV_FORMAT_FLOAT64:
            return "float64";
        default:
            return "unknown";
    }
}

int verbx_wav_read(
    const char *path,
    verbx_audio_buffer *out,
    char *error_message,
    size_t error_message_size
) {
    FILE *stream = NULL;
    unsigned char riff_header[12];
    unsigned short channels = 0U;
    unsigned int sample_rate = 0U;
    unsigned short bits_per_sample = 0U;
    unsigned short audio_format = 0U;
    uint32_t block_align = 0U;
    unsigned char *data = NULL;
    uint32_t data_size = 0U;
    int fmt_found = 0;
    int data_found = 0;
    size_t frame_count = 0U;
    size_t sample_count = 0U;
    size_t frame_index;

    if (out == NULL) {
        set_error(error_message, error_message_size, "output buffer is null");
        return -1;
    }
    out->samples = NULL;
    out->frames = 0U;
    out->sample_rate = 0U;
    out->channels = 0U;

    stream = fopen(path, "rb");
    if (stream == NULL) {
        set_errorf(error_message, error_message_size, "failed to open input WAV", path);
        return -1;
    }
    if (read_exact(stream, riff_header, sizeof(riff_header)) != 0) {
        set_error(error_message, error_message_size, "failed to read WAV header");
        fclose(stream);
        return -1;
    }
    if ((memcmp(riff_header, "RIFF", 4U) != 0) || (memcmp(riff_header + 8U, "WAVE", 4U) != 0)) {
        set_error(error_message, error_message_size, "input is not a RIFF/WAVE file");
        fclose(stream);
        return -1;
    }

    while (!data_found) {
        unsigned char chunk_header[8];
        uint32_t chunk_size;
        long chunk_data_start;
        uint32_t padded_chunk_size;

        if (read_exact(stream, chunk_header, sizeof(chunk_header)) != 0) {
            break;
        }
        chunk_size = read_u32_le(chunk_header + 4U);
        chunk_data_start = ftell(stream);
        if (chunk_data_start < 0L) {
            set_error(error_message, error_message_size, "failed to inspect WAV chunk");
            fclose(stream);
            return -1;
        }
        padded_chunk_size = chunk_size + (chunk_size % 2U);

        if (memcmp(chunk_header, "fmt ", 4U) == 0) {
            unsigned char fmt[40];
            if (chunk_size < 16U) {
                set_error(error_message, error_message_size, "WAV fmt chunk is too small");
                fclose(stream);
                return -1;
            }
            if (chunk_size > sizeof(fmt)) {
                set_error(error_message, error_message_size, "WAV fmt chunk is too large");
                fclose(stream);
                return -1;
            }
            if (read_exact(stream, fmt, chunk_size) != 0) {
                set_error(error_message, error_message_size, "failed to read WAV fmt chunk");
                fclose(stream);
                return -1;
            }
            audio_format = read_u16_le(fmt);
            channels = read_u16_le(fmt + 2U);
            sample_rate = read_u32_le(fmt + 4U);
            block_align = read_u16_le(fmt + 12U);
            bits_per_sample = read_u16_le(fmt + 14U);
            fmt_found = 1;
        } else if (memcmp(chunk_header, "data", 4U) == 0) {
            data = (unsigned char *)malloc(chunk_size > 0U ? (size_t)chunk_size : 1U);
            if (data == NULL) {
                set_error(error_message, error_message_size, "failed to allocate WAV data buffer");
                fclose(stream);
                return -1;
            }
            if ((chunk_size > 0U) && (read_exact(stream, data, chunk_size) != 0)) {
                set_error(error_message, error_message_size, "failed to read WAV data chunk");
                free(data);
                fclose(stream);
                return -1;
            }
            data_size = chunk_size;
            data_found = 1;
        }

        if (!data_found) {
            if (fseek(stream, chunk_data_start + (long)padded_chunk_size, SEEK_SET) != 0) {
                set_error(error_message, error_message_size, "failed to skip WAV chunk");
                fclose(stream);
                return -1;
            }
        }
    }
    fclose(stream);

    if (!fmt_found || !data_found) {
        free(data);
        set_error(error_message, error_message_size, "WAV file is missing fmt or data chunk");
        return -1;
    }
    if ((channels < 1U) || (channels > 2U)) {
        free(data);
        set_error(error_message, error_message_size, "native render currently supports mono/stereo WAV only");
        return -1;
    }
    if ((audio_format != VERBX_WAV_PCM) && (audio_format != VERBX_WAV_IEEE_FLOAT)) {
        free(data);
        set_error(error_message, error_message_size, "unsupported WAV encoding; use PCM or IEEE float");
        return -1;
    }
    if ((block_align == 0U) || ((uint32_t)data_size % block_align != 0U)) {
        free(data);
        set_error(error_message, error_message_size, "invalid WAV block alignment");
        return -1;
    }

    frame_count = (size_t)((uint32_t)data_size / block_align);
    sample_count = frame_count * (size_t)channels;
    out->samples = (double *)calloc(sample_count > 0U ? sample_count : 1U, sizeof(double));
    if (out->samples == NULL) {
        free(data);
        set_error(error_message, error_message_size, "failed to allocate decoded audio buffer");
        return -1;
    }

    if (audio_format == VERBX_WAV_PCM) {
        if (bits_per_sample == 16U) {
            for (frame_index = 0U; frame_index < sample_count; ++frame_index) {
                int16_t sample = (int16_t)read_u16_le(data + (frame_index * 2U));
                out->samples[frame_index] = (double)sample / 32768.0;
            }
        } else if (bits_per_sample == 24U) {
            for (frame_index = 0U; frame_index < sample_count; ++frame_index) {
                int32_t sample = read_i24_le(data + (frame_index * 3U));
                out->samples[frame_index] = (double)sample / 8388608.0;
            }
        } else if (bits_per_sample == 32U) {
            for (frame_index = 0U; frame_index < sample_count; ++frame_index) {
                int32_t sample = (int32_t)read_u32_le(data + (frame_index * 4U));
                out->samples[frame_index] = (double)sample / 2147483648.0;
            }
        } else {
            verbx_audio_buffer_free(out);
            free(data);
            set_error(error_message, error_message_size, "unsupported PCM WAV bit depth");
            return -1;
        }
    } else {
        if (bits_per_sample == 32U) {
            for (frame_index = 0U; frame_index < sample_count; ++frame_index) {
                float sample = 0.0f;
                memcpy(&sample, data + (frame_index * 4U), sizeof(float));
                out->samples[frame_index] = (double)sample;
            }
        } else if (bits_per_sample == 64U) {
            for (frame_index = 0U; frame_index < sample_count; ++frame_index) {
                double sample = 0.0;
                memcpy(&sample, data + (frame_index * 8U), sizeof(double));
                out->samples[frame_index] = sample;
            }
        } else {
            verbx_audio_buffer_free(out);
            free(data);
            set_error(error_message, error_message_size, "unsupported IEEE-float WAV bit depth");
            return -1;
        }
    }

    free(data);
    out->frames = frame_count;
    out->sample_rate = sample_rate;
    out->channels = channels;
    return 0;
}

int verbx_wav_write(
    const char *path,
    const verbx_audio_buffer *buffer,
    verbx_wav_format format,
    char *error_message,
    size_t error_message_size
) {
    FILE *stream = NULL;
    uint16_t audio_format;
    uint16_t bits_per_sample;
    uint16_t bytes_per_sample;
    uint32_t block_align;
    uint32_t byte_rate;
    uint64_t data_bytes_u64;
    uint32_t data_bytes;
    uint32_t riff_size;
    size_t sample_index;

    if ((buffer == NULL) || (buffer->samples == NULL)) {
        set_error(error_message, error_message_size, "nothing to write");
        return -1;
    }
    if ((buffer->channels < 1U) || (buffer->channels > 2U)) {
        set_error(error_message, error_message_size, "native writer currently supports mono/stereo only");
        return -1;
    }

    switch (format) {
        case VERBX_WAV_FORMAT_PCM16:
            audio_format = VERBX_WAV_PCM;
            bits_per_sample = 16U;
            bytes_per_sample = 2U;
            break;
        case VERBX_WAV_FORMAT_FLOAT32:
            audio_format = VERBX_WAV_IEEE_FLOAT;
            bits_per_sample = 32U;
            bytes_per_sample = 4U;
            break;
        case VERBX_WAV_FORMAT_FLOAT64:
            audio_format = VERBX_WAV_IEEE_FLOAT;
            bits_per_sample = 64U;
            bytes_per_sample = 8U;
            break;
        default:
            set_error(error_message, error_message_size, "unsupported output WAV format");
            return -1;
    }

    block_align = (uint32_t)((uint32_t)buffer->channels * (uint32_t)bytes_per_sample);
    byte_rate = (uint32_t)(buffer->sample_rate * block_align);
    data_bytes_u64 = (uint64_t)buffer->frames * (uint64_t)block_align;
    if (data_bytes_u64 > 0xFFFFFFFFULL) {
        set_error(error_message, error_message_size, "WAV output is too large for RIFF");
        return -1;
    }
    data_bytes = (uint32_t)data_bytes_u64;
    riff_size = 36U + data_bytes;

    stream = fopen(path, "wb");
    if (stream == NULL) {
        set_errorf(error_message, error_message_size, "failed to open output WAV", path);
        return -1;
    }

    fwrite("RIFF", sizeof(char), 4U, stream);
    write_u32_le(stream, riff_size);
    fwrite("WAVE", sizeof(char), 4U, stream);
    fwrite("fmt ", sizeof(char), 4U, stream);
    write_u32_le(stream, 16U);
    write_u16_le(stream, audio_format);
    write_u16_le(stream, buffer->channels);
    write_u32_le(stream, buffer->sample_rate);
    write_u32_le(stream, byte_rate);
    write_u16_le(stream, block_align);
    write_u16_le(stream, bits_per_sample);
    fwrite("data", sizeof(char), 4U, stream);
    write_u32_le(stream, data_bytes);

    for (sample_index = 0U; sample_index < (buffer->frames * (size_t)buffer->channels); ++sample_index) {
        double sample = buffer->samples[sample_index];
        if (format == VERBX_WAV_FORMAT_PCM16) {
            int16_t pcm;
            unsigned char bytes[2];
            if (sample > 1.0) {
                sample = 1.0;
            } else if (sample < -1.0) {
                sample = -1.0;
            }
            pcm = (int16_t)lrint(sample * 32767.0);
            bytes[0] = (unsigned char)(pcm & 0xFF);
            bytes[1] = (unsigned char)((pcm >> 8) & 0xFF);
            fwrite(bytes, sizeof(unsigned char), 2U, stream);
        } else if (format == VERBX_WAV_FORMAT_FLOAT32) {
            float float_sample = (float)sample;
            fwrite(&float_sample, sizeof(float), 1U, stream);
        } else {
            fwrite(&sample, sizeof(double), 1U, stream);
        }
    }

    if (fclose(stream) != 0) {
        set_error(error_message, error_message_size, strerror(errno));
        return -1;
    }
    return 0;
}
