#include "verbx_c/plugin_params.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static void assert_close(double actual, double expected, double tolerance) {
    assert(fabs(actual - expected) <= tolerance);
}

int main(void) {
    const verbx_plugin_parameter *rt60_coarse;
    const verbx_plugin_parameter *rt60_fine;
    const verbx_plugin_parameter *reverse;
    const verbx_plugin_parameter *quality;
    double midpoint;
    double fine_up;

    assert(verbx_plugin_parameter_count() == VERBX_PLUGIN_PARAMETER_COUNT);
    assert(VERBX_PLUGIN_PARAMETER_COUNT >= 12U);

    rt60_coarse = verbx_plugin_parameter_by_id(VERBX_PLUGIN_PARAM_RT60_COARSE);
    rt60_fine = verbx_plugin_parameter_by_id(VERBX_PLUGIN_PARAM_RT60_FINE);
    reverse = verbx_plugin_parameter_by_id(VERBX_PLUGIN_PARAM_REVERSE);
    quality = verbx_plugin_parameter_by_id(VERBX_PLUGIN_PARAM_QUALITY_MODE);

    assert(rt60_coarse != NULL);
    assert(rt60_fine != NULL);
    assert(reverse != NULL);
    assert(quality != NULL);
    assert(strcmp(rt60_coarse->key, "rt60_coarse") == 0);
    assert(strcmp(rt60_fine->key, "rt60_fine") == 0);
    assert(strcmp(reverse->label, "Reverse") == 0);
    assert(quality->default_value == (double)VERBX_PLUGIN_QUALITY_TARGET_192K);

    assert_close(verbx_plugin_map_rt60_seconds(0.0, 0.0), 0.01, 1e-12);
    assert_close(verbx_plugin_map_rt60_seconds(1.0, 0.0), 360.0, 1e-9);

    midpoint = verbx_plugin_map_rt60_seconds(0.5, 0.0);
    assert_close(midpoint, sqrt(0.01 * 360.0), 1e-12);

    fine_up = verbx_plugin_map_rt60_seconds(0.5, 1.0);
    assert_close(fine_up, midpoint * 1.20, 1e-12);

    assert_close(verbx_plugin_map_rt60_seconds(-1.0, -2.0), 0.01, 1e-12);
    assert_close(verbx_plugin_map_rt60_seconds(2.0, 2.0), 360.0, 1e-9);

    assert(verbx_plugin_parameter_by_id((verbx_plugin_parameter_id)9999) == NULL);
    assert(verbx_plugin_parameter_at(VERBX_PLUGIN_PARAMETER_COUNT) == NULL);

    return 0;
}
