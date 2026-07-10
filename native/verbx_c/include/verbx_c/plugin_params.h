#ifndef VERBX_C_PLUGIN_PARAMS_H
#define VERBX_C_PLUGIN_PARAMS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VERBX_PLUGIN_PARAM_PRE_DELAY_MS = 0,
    VERBX_PLUGIN_PARAM_ROOM_SIZE = 1,
    VERBX_PLUGIN_PARAM_RT60_COARSE = 2,
    VERBX_PLUGIN_PARAM_RT60_FINE = 3,
    VERBX_PLUGIN_PARAM_DAMPING = 4,
    VERBX_PLUGIN_PARAM_WIDTH = 5,
    VERBX_PLUGIN_PARAM_DIFFUSION = 6,
    VERBX_PLUGIN_PARAM_WET = 7,
    VERBX_PLUGIN_PARAM_DRY = 8,
    VERBX_PLUGIN_PARAM_FREEZE = 9,
    VERBX_PLUGIN_PARAM_REVERSE = 10,
    VERBX_PLUGIN_PARAM_QUALITY_MODE = 11,
    VERBX_PLUGIN_PARAMETER_COUNT = 12
} verbx_plugin_parameter_id;

typedef enum {
    VERBX_PLUGIN_PARAMETER_FLOAT = 0,
    VERBX_PLUGIN_PARAMETER_BOOL = 1,
    VERBX_PLUGIN_PARAMETER_CHOICE = 2
} verbx_plugin_parameter_kind;

typedef enum {
    VERBX_PLUGIN_QUALITY_HOST = 0,
    VERBX_PLUGIN_QUALITY_2X = 1,
    VERBX_PLUGIN_QUALITY_4X = 2,
    VERBX_PLUGIN_QUALITY_TARGET_192K = 3
} verbx_plugin_quality_mode;

typedef struct {
    verbx_plugin_parameter_id id;
    const char *key;
    const char *label;
    const char *unit;
    verbx_plugin_parameter_kind kind;
    double minimum;
    double maximum;
    double default_value;
} verbx_plugin_parameter;

size_t verbx_plugin_parameter_count(void);
const verbx_plugin_parameter *verbx_plugin_parameter_at(size_t index);
const verbx_plugin_parameter *verbx_plugin_parameter_by_id(verbx_plugin_parameter_id id);
double verbx_plugin_clamp(double value, double minimum, double maximum);
double verbx_plugin_map_rt60_seconds(double normalized_coarse, double bipolar_fine);

#ifdef __cplusplus
}
#endif

#endif
