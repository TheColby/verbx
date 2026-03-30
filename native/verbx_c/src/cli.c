#include "verbx_c/cli.h"
#include "verbx_c/version.h"

#include <stdio.h>
#include <string.h>

static void print_usage(FILE *stream) {
    fprintf(stream, "%s %s\n", VERBX_C_PROJECT_NAME, VERBX_C_VERSION);
    fprintf(stream, "Native executable scaffold for the verbx v0.8 track.\n\n");
    fprintf(stream, "Usage:\n");
    fprintf(stream, "  %s help\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  %s version\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  %s doctor\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "  %s render <in.wav> <out.wav> [options]\n", VERBX_C_PROJECT_NAME);
    fprintf(stream, "\n");
    fprintf(stream, "Status:\n");
    fprintf(stream, "  render is intentionally not implemented yet in this scaffold.\n");
}

static void print_version(void) {
    printf("%s %s\n", VERBX_C_PROJECT_NAME, VERBX_C_VERSION);
}

static void print_doctor(void) {
    printf("%s doctor\n", VERBX_C_PROJECT_NAME);
    printf("version: %s\n", VERBX_C_VERSION);
    printf("status: scaffold-only\n");
    printf("dsp_core: not yet ported\n");
    printf("wav_io: not yet implemented\n");
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
        fprintf(
            stderr,
            "%s render is not implemented yet.\n"
            "This v0.8 scaffold establishes the native executable, build path, and CLI shell first.\n",
            VERBX_C_PROJECT_NAME
        );
        return 2;
    }

    fprintf(stderr, "%s: unknown command '%s'\n\n", VERBX_C_PROJECT_NAME, command);
    print_usage(stderr);
    return 2;
}
