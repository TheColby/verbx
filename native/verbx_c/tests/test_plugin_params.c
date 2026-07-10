#include "verbx_c/plugin_params.h"

#include <assert.h>
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

static void assert_close(double actual, double expected, double tolerance) {
    assert(fabs(actual - expected) <= tolerance);
}

static void assert_parameter_matches(const verbx_plugin_parameter *actual, const expected_parameter *expected) {
    assert(actual != NULL);
    assert(expected != NULL);
    assert(actual->id == expected->id);
    assert(actual->key != NULL);
    assert(actual->label != NULL);
    assert(actual->unit != NULL);
    assert(strcmp(actual->key, expected->key) == 0);
    assert(strcmp(actual->label, expected->label) == 0);
    assert(strcmp(actual->unit, expected->unit) == 0);
    assert(actual->kind == expected->kind);
    assert_close(actual->minimum, expected->minimum, 1e-12);
    assert_close(actual->maximum, expected->maximum, 1e-12);
    assert_close(actual->default_value, expected->default_value, 1e-12);
    assert(actual->minimum <= actual->maximum);
    assert(actual->default_value >= actual->minimum);
    assert(actual->default_value <= actual->maximum);
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

    assert(verbx_plugin_parameter_count() == VERBX_PLUGIN_PARAMETER_COUNT);
    assert(VERBX_PLUGIN_PARAMETER_COUNT == 12U);

    for (index = 0U; index < VERBX_PLUGIN_PARAMETER_COUNT; ++index) {
        parameter = verbx_plugin_parameter_at(index);
        assert_parameter_matches(parameter, &expected_parameters[index]);
        assert(verbx_plugin_parameter_by_id(expected_parameters[index].id) == parameter);
    }

    assert_close(verbx_plugin_map_rt60_seconds(0.0, 0.0), 0.01, 1e-12);
    assert_close(verbx_plugin_map_rt60_seconds(1.0, 0.0), 360.0, 1e-9);

    midpoint = verbx_plugin_map_rt60_seconds(0.5, 0.0);
    assert_close(midpoint, sqrt(0.01 * 360.0), 1e-12);

    fine_up = verbx_plugin_map_rt60_seconds(0.5, 1.0);
    assert_close(fine_up, midpoint * 1.20, 1e-12);

    fine_down = verbx_plugin_map_rt60_seconds(0.5, -1.0);
    assert_close(fine_down, midpoint / 1.20, 1e-12);

    assert_close(verbx_plugin_map_rt60_seconds(-1.0, -2.0), 0.01, 1e-12);
    assert_close(verbx_plugin_map_rt60_seconds(2.0, 2.0), 360.0, 1e-9);

    assert(verbx_plugin_parameter_by_id((verbx_plugin_parameter_id)9999) == NULL);
    assert(verbx_plugin_parameter_at(VERBX_PLUGIN_PARAMETER_COUNT) == NULL);

    return 0;
}
