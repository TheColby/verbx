#include "verbx_c/plugin_params.h"

#include <math.h>

static const verbx_plugin_parameter VERBX_PLUGIN_PARAMETERS[VERBX_PLUGIN_PARAMETER_COUNT] = {
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

size_t verbx_plugin_parameter_count(void) {
    return VERBX_PLUGIN_PARAMETER_COUNT;
}

const verbx_plugin_parameter *verbx_plugin_parameter_at(size_t index) {
    if (index >= VERBX_PLUGIN_PARAMETER_COUNT) {
        return 0;
    }

    return &VERBX_PLUGIN_PARAMETERS[index];
}

const verbx_plugin_parameter *verbx_plugin_parameter_by_id(verbx_plugin_parameter_id id) {
    size_t index;

    for (index = 0U; index < VERBX_PLUGIN_PARAMETER_COUNT; ++index) {
        if (VERBX_PLUGIN_PARAMETERS[index].id == id) {
            return &VERBX_PLUGIN_PARAMETERS[index];
        }
    }

    return 0;
}

double verbx_plugin_clamp(double value, double minimum, double maximum) {
    if (value < minimum) {
        return minimum;
    }

    if (value > maximum) {
        return maximum;
    }

    return value;
}

double verbx_plugin_map_rt60_seconds(double normalized_coarse, double bipolar_fine) {
    const double min_rt60 = 0.01;
    const double max_rt60 = 360.0;
    const double fine_max_ratio = 1.20;
    double coarse = verbx_plugin_clamp(normalized_coarse, 0.0, 1.0);
    double fine = verbx_plugin_clamp(bipolar_fine, -1.0, 1.0);
    double coarse_seconds = exp(log(min_rt60) + ((log(max_rt60) - log(min_rt60)) * coarse));
    /* Fine trim is symmetric in log space: +1.0 scales by 1.20, -1.0 scales by 1/1.20. */
    double fine_ratio = exp(log(fine_max_ratio) * fine);

    return verbx_plugin_clamp(coarse_seconds * fine_ratio, min_rt60, max_rt60);
}
