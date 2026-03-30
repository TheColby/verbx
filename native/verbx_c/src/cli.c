#include "verbx_c/cli.h"
#include "verbx_c/render.h"
#include "verbx_c/version.h"
#include "verbx_c/wav_io.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(FILE *stream) {
    fprintf(stream, "%s %s\n", VERBX_C_PROJECT_NAME, VERBX_C_VERSION);
    fprintf(stream, "Native executable foundation for the verbx v0.8 track.\n\n");
    fprintf(stream, "Usage:\n");
    fprintf(stream, "  %s help\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  %s version\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  %s doctor\n", VERBX_C_PROJECT_NAME);
    fprintf(
        stream,
        "  %s render <in.wav> <out.wav> [--rt60 SEC] [--wet X] [--dry X]\n",
        VERBX_C_PROJECT_NAME
    );
    fprintf(
        stream,
        "           [--pre-delay-ms MS] [--damping X] [--tail-threshold-db DB]\n"
    );
    fprintf(
        stream,
        "           [--tail-hold-ms MS] [--out-format pcm16|float32|float64]\n"
    );
    fprintf(stream, "\n");
    fprintf(stream, "Status:\n");
    fprintf(stream, "  mono/stereo WAV render path is implemented.\n");
    fprintf(stream, "  algorithmic core is the first native offline port, not feature parity.\n");
}

static void print_version(void) {
    printf("%s %s\n", VERBX_C_PROJECT_NAME, VERBX_C_VERSION);
}

static void print_doctor(void) {
    printf("%s doctor\n", VERBX_C_PROJECT_NAME);
    printf("version: %s\n", VERBX_C_VERSION);
    printf("status: native-render-foundation\n");
    printf("dsp_core: algorithmic offline core (foundational subset)\n");
    printf("wav_io: mono/stereo PCM + IEEE float WAV\n");
    printf("compiler: ");
#if defined(__clang__)
    printf("clang %s\n", __clang_version__);
#elif defined(__GNUC__)
    printf("gcc %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    printf("msvc %d\n", _MSC_VER);
#else
    printf("unknown\n");
#endif
    printf("c_standard: %ld\n", (long)__STDC_VERSION__);
}

static int parse_double_option(const char *name, const char *value_text, double *out_value) {
    char *end = NULL;
    double value;

    errno = 0;
    value = strtod(value_text, &end);
    if ((errno != 0) || (end == value_text) || ((end != NULL) && (*end != '\0'))) {
        fprintf(stderr, "%s: invalid value for %s: %s\n", VERBX_C_PROJECT_NAME, name, value_text);
        return -1;
    }
    *out_value = value;
    return 0;
}

static int parse_out_format(const char *value_text, verbx_wav_format *out_format) {
    if (strcmp(value_text, "pcm16") == 0) {
        *out_format = VERBX_WAV_FORMAT_PCM16;
        return 0;
    }
    if (strcmp(value_text, "float32") == 0) {
        *out_format = VERBX_WAV_FORMAT_FLOAT32;
        return 0;
    }
    if (strcmp(value_text, "float64") == 0) {
        *out_format = VERBX_WAV_FORMAT_FLOAT64;
        return 0;
    }
    fprintf(
        stderr,
        "%s: unsupported --out-format value '%s' (use pcm16, float32, or float64)\n",
        VERBX_C_PROJECT_NAME,
        value_text
    );
    return -1;
}

static int handle_render(int argc, char **argv) {
    const char *input_path;
    const char *output_path;
    verbx_render_options options = {
        .rt60 = 2.5,
        .wet = 0.8,
        .dry = 0.2,
        .damping = 0.45,
        .pre_delay_ms = 20.0,
        .tail_threshold_db = -120.0,
        .tail_hold_ms = 10.0,
        .out_format = VERBX_WAV_FORMAT_FLOAT32
    };
    verbx_render_report report = {0};
    char error_message[256];
    int index;

    if (argc < 4) {
        fprintf(stderr, "%s render requires INFILE and OUTFILE\n", VERBX_C_PROJECT_NAME);
        return 2;
    }
    input_path = argv[2];
    output_path = argv[3];
    for (index = 4; index < argc; ++index) {
        const char *arg = argv[index];
        if ((strcmp(arg, "--rt60") == 0) && (index + 1 < argc)) {
            if (parse_double_option("--rt60", argv[++index], &options.rt60) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--wet") == 0) && (index + 1 < argc)) {
            if (parse_double_option("--wet", argv[++index], &options.wet) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--dry") == 0) && (index + 1 < argc)) {
            if (parse_double_option("--dry", argv[++index], &options.dry) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--damping") == 0) && (index + 1 < argc)) {
            if (parse_double_option("--damping", argv[++index], &options.damping) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--pre-delay-ms") == 0) && (index + 1 < argc)) {
            if (parse_double_option("--pre-delay-ms", argv[++index], &options.pre_delay_ms) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--tail-threshold-db") == 0) && (index + 1 < argc)) {
            if (
                parse_double_option("--tail-threshold-db", argv[++index], &options.tail_threshold_db) != 0
            ) {
                return 2;
            }
        } else if ((strcmp(arg, "--tail-hold-ms") == 0) && (index + 1 < argc)) {
            if (parse_double_option("--tail-hold-ms", argv[++index], &options.tail_hold_ms) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--out-format") == 0) && (index + 1 < argc)) {
            if (parse_out_format(argv[++index], &options.out_format) != 0) {
                return 2;
            }
        } else {
            fprintf(stderr, "%s: unknown render option '%s'\n", VERBX_C_PROJECT_NAME, arg);
            return 2;
        }
    }

    if (verbx_render_file(input_path, output_path, &options, &report, error_message, sizeof(error_message)) != 0) {
        fprintf(stderr, "%s render failed: %s\n", VERBX_C_PROJECT_NAME, error_message);
        return 1;
    }
    printf(
        "%s render complete\nsample_rate: %u\nchannels: %u\ninput_frames: %zu\noutput_frames: %zu\nout_format: %s\n",
        VERBX_C_PROJECT_NAME,
        report.sample_rate,
        report.channels,
        report.input_frames,
        report.output_frames,
        verbx_wav_format_name(report.out_format)
    );
    return 0;
}

int verbx_c_run(int argc, char **argv) {
    const char *command;

    if (argc <= 1) {
        print_usage(stdout);
        return 0;
    }

    command = argv[1];
    if ((strcmp(command, "help") == 0) || (strcmp(command, "--help") == 0) ||
        (strcmp(command, "-h") == 0)) {
        print_usage(stdout);
        return 0;
    }
    if ((strcmp(command, "version") == 0) || (strcmp(command, "--version") == 0) ||
        (strcmp(command, "-V") == 0)) {
        print_version();
        return 0;
    }
    if (strcmp(command, "doctor") == 0) {
        print_doctor();
        return 0;
    }
    if (strcmp(command, "render") == 0) {
        return handle_render(argc, argv);
    }

    fprintf(stderr, "%s: unknown command '%s'\n\n", VERBX_C_PROJECT_NAME, command);
    print_usage(stderr);
    return 2;
}
