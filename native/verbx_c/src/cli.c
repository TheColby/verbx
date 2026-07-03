#include "verbx_c/cli.h"
#include "verbx_c/render.h"
#include "verbx_c/version.h"
#include "verbx_c/wav_io.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_json_string(FILE *stream, const char *value);

static void print_usage(FILE *stream) {
    fprintf(stream, "%s %s\n", VERBX_C_PROJECT_NAME, VERBX_C_VERSION);
    fprintf(stream, "Native executable foundation for the verbx v0.8 track.\n\n");
    fprintf(stream, "Usage:\n");
    fprintf(stream, "  %s help\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  %s version\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  %s doctor [--json-out report.json]\n", VERBX_C_PROJECT_NAME);
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
        "           [--tail-hold-ms MS] [--tail-metric peak|rms]\n"
    );
    fprintf(
        stream,
        "           [--peak-safe] [--peak-ceiling-db DB]\n"
    );
    fprintf(
        stream,
        "           [--out-format pcm16|float32|float64] [--json-out report.json]\n"
    );
    fprintf(stream, "\n");
    fprintf(stream, "Status:\n");
    fprintf(stream, "  mono/stereo WAV render path is implemented.\n");
    fprintf(stream, "  algorithmic core is the first native offline port, not feature parity.\n");
}

static void print_version(void) {
    printf("%s %s\n", VERBX_C_PROJECT_NAME, VERBX_C_VERSION);
}

static const char *compiler_family(void) {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

static const char *compiler_version(void) {
#if defined(__clang__)
    return __clang_version__;
#elif defined(__GNUC__)
    static char version[64];
    snprintf(version, sizeof(version), "%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    return version;
#elif defined(_MSC_VER)
    static char version[64];
    snprintf(version, sizeof(version), "%d", _MSC_VER);
    return version;
#else
    return "unknown";
#endif
}

static void print_doctor(void) {
    printf("%s doctor\n", VERBX_C_PROJECT_NAME);
    printf("version: %s\n", VERBX_C_VERSION);
    printf("status: native-render-foundation\n");
    printf("dsp_core: algorithmic offline core (foundational subset)\n");
    printf("wav_io: mono/stereo PCM + IEEE float WAV\n");
    printf("process_contract: read -> render -> tail_stop -> write (deterministic)\n");
    printf("error_contract: exit 0=ok, 1=render failure, 2=cli usage error\n");
    printf("compiler: %s %s\n", compiler_family(), compiler_version());
    printf("c_standard: %ld\n", (long)__STDC_VERSION__);
}

static int write_doctor_json_report(const char *json_path) {
    FILE *stream;

    if ((json_path == NULL) || (json_path[0] == '\0')) {
        return 0;
    }
    stream = fopen(json_path, "wb");
    if (stream == NULL) {
        fprintf(stderr, "%s: failed to open --json-out '%s': %s\n", VERBX_C_PROJECT_NAME, json_path, strerror(errno));
        return -1;
    }
    fputs("{\n", stream);
    fputs("  \"schema\": \"native-doctor-report-v1\",\n", stream);
    fprintf(stream, "  \"command\": \"%s doctor\",\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  \"project\": \"%s\",\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  \"version\": \"%s\",\n", VERBX_C_VERSION);
    fputs("  \"status\": \"native-render-foundation\",\n", stream);
    fputs("  \"dsp_core\": \"algorithmic offline core (foundational subset)\",\n", stream);
    fputs("  \"wav_io\": \"mono/stereo PCM + IEEE float WAV\",\n", stream);
    fputs("  \"process_contract\": \"read -> render -> tail_stop -> write (deterministic)\",\n", stream);
    fputs("  \"error_contract\": \"exit 0=ok, 1=render failure, 2=cli usage error\",\n", stream);
    fputs("  \"compiler_family\": ", stream);
    write_json_string(stream, compiler_family());
    fputs(",\n  \"compiler_version\": ", stream);
    write_json_string(stream, compiler_version());
    fprintf(stream, ",\n  \"c_standard\": %ld\n", (long)__STDC_VERSION__);
    fputs("}\n", stream);
    if (fclose(stream) != 0) {
        fprintf(stderr, "%s: failed to write --json-out '%s': %s\n", VERBX_C_PROJECT_NAME, json_path, strerror(errno));
        return -1;
    }
    return 0;
}

static int handle_doctor(int argc, char **argv) {
    const char *json_out_path = NULL;
    int index;

    for (index = 2; index < argc; ++index) {
        const char *arg = argv[index];
        if ((strcmp(arg, "--json-out") == 0) && (index + 1 < argc)) {
            json_out_path = argv[++index];
        } else {
            fprintf(stderr, "%s: unknown doctor option '%s'\n", VERBX_C_PROJECT_NAME, arg);
            return 2;
        }
    }
    print_doctor();
    return write_doctor_json_report(json_out_path) == 0 ? 0 : 1;
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

static int parse_tail_metric(const char *value_text, verbx_tail_metric *out_metric) {
    if (strcmp(value_text, "peak") == 0) {
        *out_metric = VERBX_TAIL_METRIC_PEAK;
        return 0;
    }
    if (strcmp(value_text, "rms") == 0) {
        *out_metric = VERBX_TAIL_METRIC_RMS;
        return 0;
    }
    fprintf(
        stderr,
        "%s: unsupported --tail-metric value '%s' (use peak or rms)\n",
        VERBX_C_PROJECT_NAME,
        value_text
    );
    return -1;
}

static void write_json_string(FILE *stream, const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    fputc('"', stream);
    if (cursor != NULL) {
        while (*cursor != '\0') {
            unsigned char ch = *cursor++;
            if ((ch == '"') || (ch == '\\')) {
                fputc('\\', stream);
                fputc((int)ch, stream);
            } else if (ch == '\n') {
                fputs("\\n", stream);
            } else if (ch == '\r') {
                fputs("\\r", stream);
            } else if (ch == '\t') {
                fputs("\\t", stream);
            } else if (ch < 0x20U) {
                fprintf(stream, "\\u%04x", (unsigned int)ch);
            } else {
                fputc((int)ch, stream);
            }
        }
    }
    fputc('"', stream);
}

static int write_render_json_report(
    const char *json_path,
    const char *input_path,
    const char *output_path,
    const verbx_render_options *options,
    const verbx_render_report *report
) {
    FILE *stream;

    if ((json_path == NULL) || (json_path[0] == '\0')) {
        return 0;
    }
    stream = fopen(json_path, "wb");
    if (stream == NULL) {
        fprintf(stderr, "%s: failed to open --json-out '%s': %s\n", VERBX_C_PROJECT_NAME, json_path, strerror(errno));
        return -1;
    }
    fputs("{\n", stream);
    fputs("  \"schema\": \"native-render-report-v1\",\n", stream);
    fprintf(stream, "  \"command\": \"%s render\",\n", VERBX_C_PROJECT_NAME);
    fputs("  \"status\": ", stream);
    write_json_string(stream, verbx_status_code_name(report->status_code));
    fputs(",\n  \"input_path\": ", stream);
    write_json_string(stream, input_path);
    fputs(",\n  \"output_path\": ", stream);
    write_json_string(stream, output_path);
    fprintf(stream, ",\n  \"sample_rate\": %u,\n", report->sample_rate);
    fprintf(stream, "  \"channels\": %u,\n", report->channels);
    fprintf(stream, "  \"input_frames\": %zu,\n", report->input_frames);
    fprintf(stream, "  \"output_frames\": %zu,\n", report->output_frames);
    fputs("  \"out_format\": ", stream);
    write_json_string(stream, verbx_wav_format_name(report->out_format));
    fputs(",\n  \"tail_metric\": ", stream);
    write_json_string(stream, verbx_tail_metric_name(report->tail_metric));
    fprintf(stream, ",\n  \"rt60\": %.17g,\n", options->rt60);
    fprintf(stream, "  \"wet\": %.17g,\n", options->wet);
    fprintf(stream, "  \"dry\": %.17g,\n", options->dry);
    fprintf(stream, "  \"damping\": %.17g,\n", options->damping);
    fprintf(stream, "  \"pre_delay_ms\": %.17g,\n", options->pre_delay_ms);
    fprintf(stream, "  \"tail_threshold_db\": %.17g,\n", options->tail_threshold_db);
    fprintf(stream, "  \"tail_hold_ms\": %.17g,\n", options->tail_hold_ms);
    fprintf(stream, "  \"peak_safe\": %s,\n", report->peak_safe_applied ? "true" : "false");
    fprintf(stream, "  \"peak_ceiling_db\": %.17g,\n", report->peak_ceiling_db);
    fprintf(stream, "  \"input_peak_abs\": %.17g,\n", report->input_peak_abs);
    fprintf(stream, "  \"output_peak_abs\": %.17g,\n", report->output_peak_abs);
    fprintf(stream, "  \"peak_gain\": %.17g\n", report->peak_gain);
    fputs("}\n", stream);
    if (fclose(stream) != 0) {
        fprintf(stderr, "%s: failed to write --json-out '%s': %s\n", VERBX_C_PROJECT_NAME, json_path, strerror(errno));
        return -1;
    }
    return 0;
}

static int handle_render(int argc, char **argv) {
    const char *input_path;
    const char *output_path;
    const char *json_out_path = NULL;
    verbx_render_options options = {
        .rt60 = 2.5,
        .wet = 0.8,
        .dry = 0.2,
        .damping = 0.45,
        .pre_delay_ms = 20.0,
        .tail_threshold_db = -120.0,
        .tail_hold_ms = 10.0,
        .peak_safe = 0,
        .peak_ceiling_db = -1.0,
        .tail_metric = VERBX_TAIL_METRIC_PEAK,
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
        } else if ((strcmp(arg, "--tail-metric") == 0) && (index + 1 < argc)) {
            if (parse_tail_metric(argv[++index], &options.tail_metric) != 0) {
                return 2;
            }
        } else if (strcmp(arg, "--peak-safe") == 0) {
            options.peak_safe = 1;
        } else if ((strcmp(arg, "--peak-ceiling-db") == 0) && (index + 1 < argc)) {
            if (parse_double_option("--peak-ceiling-db", argv[++index], &options.peak_ceiling_db) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--out-format") == 0) && (index + 1 < argc)) {
            if (parse_out_format(argv[++index], &options.out_format) != 0) {
                return 2;
            }
        } else if ((strcmp(arg, "--json-out") == 0) && (index + 1 < argc)) {
            json_out_path = argv[++index];
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
        "%s render complete\nsample_rate: %u\nchannels: %u\ninput_frames: %zu\noutput_frames: %zu\nout_format: %s\ntail_metric: %s\npeak_safe: %s\npeak_ceiling_db: %.2f\ninput_peak_abs: %.9f\noutput_peak_abs: %.9f\npeak_gain: %.9f\nstatus: %s\n",
        VERBX_C_PROJECT_NAME,
        report.sample_rate,
        report.channels,
        report.input_frames,
        report.output_frames,
        verbx_wav_format_name(report.out_format),
        verbx_tail_metric_name(report.tail_metric),
        report.peak_safe_applied ? "true" : "false",
        report.peak_ceiling_db,
        report.input_peak_abs,
        report.output_peak_abs,
        report.peak_gain,
        verbx_status_code_name(report.status_code)
    );
    if (write_render_json_report(json_out_path, input_path, output_path, &options, &report) != 0) {
        return 1;
    }
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
        return handle_doctor(argc, argv);
    }
    if (strcmp(command, "render") == 0) {
        return handle_render(argc, argv);
    }

    fprintf(stderr, "%s: unknown command '%s'\n\n", VERBX_C_PROJECT_NAME, command);
    print_usage(stderr);
    return 2;
}
