#include "verbx_c/plugin_params.h"

#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    verbx_plugin_parameter_id id;
    const char *key;
    const char *label;
    const char *unit;
    verbx_plugin_parameter_kind kind;
    double minimum;
    double maximum;
    double default_value;
} expected_parameter;

static int fail_message(const char *message) {
    fprintf(stderr, "test_plugin_params: %s\n", message);
    return 1;
}

static int require_true(int condition, const char *message) {
    if (!condition) {
        return fail_message(message);
    }

    return 0;
}

static int require_close(double actual, double expected, double tolerance, const char *message) {
    if (fabs(actual - expected) > tolerance) {
        fprintf(
            stderr,
            "test_plugin_params: %s (actual=%0.17g expected=%0.17g tolerance=%0.17g)\n",
            message,
            actual,
            expected,
            tolerance
        );
        return 1;
    }

    return 0;
}

static int require_string_equal(const char *actual, const char *expected, const char *message) {
    if (actual == NULL || expected == NULL || strcmp(actual, expected) != 0) {
        fprintf(
            stderr,
            "test_plugin_params: %s (actual=%s expected=%s)\n",
            message,
            actual != NULL ? actual : "(null)",
            expected != NULL ? expected : "(null)"
        );
        return 1;
    }

    return 0;
}

static int check_parameter_matches(const verbx_plugin_parameter *actual, const expected_parameter *expected) {
    if (require_true(actual != NULL, "manifest entry is NULL") != 0) {
        return 1;
    }
    if (require_true(expected != NULL, "expected manifest entry is NULL") != 0) {
        return 1;
    }
    if (require_true(actual->id == expected->id, "parameter id mismatch") != 0) {
        return 1;
    }
    if (require_true(actual->key != NULL, "parameter key is NULL") != 0) {
        return 1;
    }
    if (require_true(actual->label != NULL, "parameter label is NULL") != 0) {
        return 1;
    }
    if (require_true(actual->unit != NULL, "parameter unit is NULL") != 0) {
        return 1;
    }
    if (require_string_equal(actual->key, expected->key, "parameter key mismatch") != 0) {
        return 1;
    }
    if (require_string_equal(actual->label, expected->label, "parameter label mismatch") != 0) {
        return 1;
    }
    if (require_string_equal(actual->unit, expected->unit, "parameter unit mismatch") != 0) {
        return 1;
    }
    if (require_true(actual->kind == expected->kind, "parameter kind mismatch") != 0) {
        return 1;
    }
    if (require_close(actual->minimum, expected->minimum, 1e-12, "parameter minimum mismatch") != 0) {
        return 1;
    }
    if (require_close(actual->maximum, expected->maximum, 1e-12, "parameter maximum mismatch") != 0) {
        return 1;
    }
    if (require_close(actual->default_value, expected->default_value, 1e-12, "parameter default mismatch") != 0) {
        return 1;
    }
    if (require_true(actual->minimum <= actual->maximum, "parameter minimum exceeds maximum") != 0) {
        return 1;
    }
    if (require_true(actual->default_value >= actual->minimum, "parameter default below minimum") != 0) {
        return 1;
    }
    if (require_true(actual->default_value <= actual->maximum, "parameter default above maximum") != 0) {
        return 1;
    }

    return 0;
}

int main(void) {
    static const expected_parameter expected_parameters[VERBX_PLUGIN_PARAMETER_COUNT] = {
        {VERBX_PLUGIN_PARAM_PRE_DELAY_MS, "pre_delay_ms", "Pre-Delay", "ms", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 1000.0, 18.0},
        {VERBX_PLUGIN_PARAM_ROOM_SIZE, "room_size", "Room Size", "%", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 1.0, 0.72},
        {VERBX_PLUGIN_PARAM_RT60_COARSE, "rt60_coarse", "RT60 Coarse", "s", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 1.0, 0.50},
        {VERBX_PLUGIN_PARAM_RT60_FINE, "rt60_fine", "RT60 Fine", "%", VERBX_PLUGIN_PARAMETER_FLOAT, -1.0, 1.0, 0.0},
        {VERBX_PLUGIN_PARAM_DAMPING, "damping", "Damping", "", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 0.98, 0.41},
        {VERBX_PLUGIN_PARAM_WIDTH, "width", "Width", "", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 2.0, 1.35},
        {VERBX_PLUGIN_PARAM_DIFFUSION, "diffusion", "Diffusion", "", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 1.0, 0.65},
        {VERBX_PLUGIN_PARAM_WET, "wet", "Wet", "", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 1.0, 0.62},
        {VERBX_PLUGIN_PARAM_DRY, "dry", "Dry", "", VERBX_PLUGIN_PARAMETER_FLOAT, 0.0, 1.0, 0.78},
        {VERBX_PLUGIN_PARAM_FREEZE, "freeze", "Freeze", "", VERBX_PLUGIN_PARAMETER_BOOL, 0.0, 1.0, 0.0},
        {VERBX_PLUGIN_PARAM_REVERSE, "reverse", "Reverse", "", VERBX_PLUGIN_PARAMETER_BOOL, 0.0, 1.0, 0.0},
        {VERBX_PLUGIN_PARAM_QUALITY_MODE, "quality_mode", "Quality", "", VERBX_PLUGIN_PARAMETER_CHOICE, 0.0, 3.0, (double)VERBX_PLUGIN_QUALITY_TARGET_192K},
    };
    const verbx_plugin_parameter *parameter;
    double midpoint;
    double fine_up;
    double fine_down;
    size_t index;

    if (require_true(verbx_plugin_parameter_count() == VERBX_PLUGIN_PARAMETER_COUNT, "parameter count mismatch") != 0) {
        return 1;
    }
    if (require_true(VERBX_PLUGIN_PARAMETER_COUNT == 12U, "parameter enum count changed") != 0) {
        return 1;
    }

    for (index = 0U; index < VERBX_PLUGIN_PARAMETER_COUNT; ++index) {
        parameter = verbx_plugin_parameter_at(index);
        if (check_parameter_matches(parameter, &expected_parameters[index]) != 0) {
            return 1;
        }
        if (require_true(verbx_plugin_parameter_by_id(expected_parameters[index].id) == parameter, "lookup by id returned wrong parameter") != 0) {
            return 1;
        }
    }

    if (require_close(verbx_plugin_map_rt60_seconds(0.0, 0.0), 0.01, 1e-12, "rt60 minimum mapping mismatch") != 0) {
        return 1;
    }
    if (require_close(verbx_plugin_map_rt60_seconds(1.0, 0.0), 360.0, 1e-9, "rt60 maximum mapping mismatch") != 0) {
        return 1;
    }

    midpoint = verbx_plugin_map_rt60_seconds(0.5, 0.0);
    if (require_close(midpoint, sqrt(0.01 * 360.0), 1e-12, "rt60 midpoint mapping mismatch") != 0) {
        return 1;
    }

    fine_up = verbx_plugin_map_rt60_seconds(0.5, 1.0);
    if (require_close(fine_up, midpoint * 1.20, 1e-12, "rt60 positive fine trim mismatch") != 0) {
        return 1;
    }

    fine_down = verbx_plugin_map_rt60_seconds(0.5, -1.0);
    if (require_close(fine_down, midpoint / 1.20, 1e-12, "rt60 negative fine trim mismatch") != 0) {
        return 1;
    }

    if (require_close(verbx_plugin_map_rt60_seconds(-1.0, -2.0), 0.01, 1e-12, "rt60 lower clamp mismatch") != 0) {
        return 1;
    }
    if (require_close(verbx_plugin_map_rt60_seconds(2.0, 2.0), 360.0, 1e-9, "rt60 upper clamp mismatch") != 0) {
        return 1;
    }

    if (require_true(verbx_plugin_parameter_by_id((verbx_plugin_parameter_id)9999) == NULL, "invalid id lookup should return NULL") != 0) {
        return 1;
    }
    if (require_true(verbx_plugin_parameter_at(VERBX_PLUGIN_PARAMETER_COUNT) == NULL, "out-of-range index lookup should return NULL") != 0) {
        return 1;
    }

    return 0;
}
