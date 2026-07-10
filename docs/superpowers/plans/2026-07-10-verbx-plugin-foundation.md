# VERBX Plug-in Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first testable AUv3/VST3 plug-in foundation: shared native library boundaries, plug-in parameter manifest, RT60 coarse/fine mapping, realtime-safe context API, and guarded JUCE shell scaffolding.

**Architecture:** Start below the DAW by making the native C code reusable as a library and adding a host-facing parameter/realtime boundary. The first realtime API is deterministic and pass-through-safe; it proves parameter mapping, quality-target status, latency reporting, and callback contracts before the full reverb DSP is moved into a stateful realtime core. The JUCE shell is scaffolded behind an opt-in CMake flag so the repo remains buildable without JUCE installed.

**Tech Stack:** C11 native core, CMake/CTest, existing Python pytest native harness, optional JUCE C++17 shell for AU/AUv3/VST3.

---

## Scope

This plan implements the first safe slice of the approved design spec at `docs/superpowers/specs/2026-07-10-verbx-auv3-vst3-plugin-design.md`.

This plan intentionally stops before a polished DAW plug-in binary. It produces independently testable software that future AU/VST work can depend on:

- A reusable `verbx_c_core` library target.
- Stable plug-in parameter manifest.
- Logarithmic RT60 coarse/fine mapping from `0.01s` to `360s`.
- Realtime host context API with default Target 192 kHz / 32-bit-float semantics.
- Visible `freeze` and `reverse` mode parameters.
- Guarded JUCE scaffold that configures cleanly when plug-in building is disabled.

## File Structure

- Create: `native/verbx_c/include/verbx_c/plugin_params.h`
- Create: `native/verbx_c/src/plugin_params.c`
- Create: `native/verbx_c/tests/test_plugin_params.c`
- Create: `native/verbx_c/include/verbx_c/plugin_realtime.h`
- Create: `native/verbx_c/src/plugin_realtime.c`
- Create: `native/verbx_c/tests/test_plugin_realtime.c`
- Create: `native/verbx_plugin/CMakeLists.txt`
- Create: `native/verbx_plugin/README.md`
- Create: `native/verbx_plugin/src/VerbXPluginProcessor.h`
- Create: `native/verbx_plugin/src/VerbXPluginProcessor.cpp`
- Create: `native/verbx_plugin/src/VerbXPluginEditor.h`
- Create: `native/verbx_plugin/src/VerbXPluginEditor.cpp`
- Modify: `native/verbx_c/CMakeLists.txt`
- Modify: `scripts/build_verbx_c.sh`
- Modify: `tests/test_native_scaffold.py`
- Modify: `docs/NATIVE_PARITY.md`
- Modify: `native/verbx_c/README.md`

---

### Task 1: Add Plug-in Parameter Manifest And RT60 Mapping

**Files:**
- Create: `native/verbx_c/include/verbx_c/plugin_params.h`
- Create: `native/verbx_c/src/plugin_params.c`
- Create: `native/verbx_c/tests/test_plugin_params.c`
- Modify: `native/verbx_c/CMakeLists.txt`

- [ ] **Step 1: Write the failing C unit test**

Create `native/verbx_c/tests/test_plugin_params.c`:

```c
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
```

- [ ] **Step 2: Wire the test into CMake before implementation**

Modify `native/verbx_c/CMakeLists.txt` enough to reference the future test target:

```cmake
include(CTest)

if(BUILD_TESTING)
  add_executable(test_plugin_params tests/test_plugin_params.c)
  target_link_libraries(test_plugin_params PRIVATE verbx_c_core)
  add_test(NAME plugin_params COMMAND test_plugin_params)
endif()
```

Run:

```bash
cmake -S native/verbx_c -B build/native/verbx_c-plan
```

Expected: fail because `verbx_c_core` or `verbx_c/plugin_params.h` is not defined yet.

- [ ] **Step 3: Add the public parameter header**

Create `native/verbx_c/include/verbx_c/plugin_params.h`:

```c
#ifndef VERBX_C_PLUGIN_PARAMS_H
#define VERBX_C_PLUGIN_PARAMS_H

#include <stddef.h>

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

#endif
```

- [ ] **Step 4: Add the manifest implementation**

Create `native/verbx_c/src/plugin_params.c`:

```c
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
    double fine_ratio = exp(log(fine_max_ratio) * fine);
    return verbx_plugin_clamp(coarse_seconds * fine_ratio, min_rt60, max_rt60);
}
```

- [ ] **Step 5: Finish the CMake split needed by the test**

Replace `native/verbx_c/CMakeLists.txt` with:

```cmake
cmake_minimum_required(VERSION 3.16)

project(verbx_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(VERBX_C_CORE_SOURCES
  src/audio.c
  src/algo_reverb.c
  src/render.c
  src/wav_io.c
  src/plugin_params.c
)

add_library(verbx_c_core STATIC ${VERBX_C_CORE_SOURCES})
target_include_directories(
  verbx_c_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(verbx_c_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT MSVC)
  target_link_libraries(verbx_c_core PUBLIC m)
endif()

add_executable(
  verbx-c
  src/main.c
  src/cli.c
)

target_link_libraries(verbx-c PRIVATE verbx_c_core)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(verbx-c PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
  add_executable(test_plugin_params tests/test_plugin_params.c)
  target_link_libraries(test_plugin_params PRIVATE verbx_c_core)
  add_test(NAME plugin_params COMMAND test_plugin_params)
endif()

install(TARGETS verbx-c RUNTIME DESTINATION bin)
```

- [ ] **Step 6: Run the test and existing native build**

Run:

```bash
cmake -S native/verbx_c -B build/native/verbx_c-plan
cmake --build build/native/verbx_c-plan
ctest --test-dir build/native/verbx_c-plan --output-on-failure
./build/native/verbx_c-plan/verbx-c version
```

Expected:

```text
100% tests passed
verbx-c 0.8.0-dev
```

- [ ] **Step 7: Commit**

```bash
git add native/verbx_c/CMakeLists.txt native/verbx_c/include/verbx_c/plugin_params.h native/verbx_c/src/plugin_params.c native/verbx_c/tests/test_plugin_params.c
git commit -m "Add plugin parameter manifest"
```

---

### Task 2: Add Realtime Host Context API

**Files:**
- Create: `native/verbx_c/include/verbx_c/plugin_realtime.h`
- Create: `native/verbx_c/src/plugin_realtime.c`
- Create: `native/verbx_c/tests/test_plugin_realtime.c`
- Modify: `native/verbx_c/CMakeLists.txt`

- [ ] **Step 1: Write the failing realtime API test**

Create `native/verbx_c/tests/test_plugin_realtime.c`:

```c
#include "verbx_c/plugin_realtime.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static void assert_close(double actual, double expected, double tolerance) {
    assert(fabs(actual - expected) <= tolerance);
}

int main(void) {
    verbx_plugin_realtime_context context;
    verbx_plugin_realtime_config config;
    verbx_plugin_realtime_params params;
    verbx_plugin_realtime_status status;
    char error[256];
    float in_l[4] = {0.25f, -0.50f, 0.75f, -1.0f};
    float in_r[4] = {-0.25f, 0.50f, -0.75f, 1.0f};
    float out_l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out_r[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float *inputs[2] = {in_l, in_r};
    float *outputs[2] = {out_l, out_r};
    size_t frame;

    memset(&context, 0, sizeof(context));
    memset(&config, 0, sizeof(config));
    memset(&params, 0, sizeof(params));
    memset(&status, 0, sizeof(status));
    memset(error, 0, sizeof(error));

    config.host_sample_rate = 48000U;
    config.max_block_frames = 512U;
    config.channel_count = 2U;
    config.quality_mode = VERBX_PLUGIN_QUALITY_TARGET_192K;

    params.rt60_coarse_normalized = 0.5;
    params.rt60_fine_bipolar = 1.0;
    params.dry = 1.0;
    params.wet = 0.0;
    params.freeze = 1;
    params.reverse = 1;

    assert(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) == 0);
    assert(verbx_plugin_realtime_internal_sample_rate(&context) == 192000U);
    assert(verbx_plugin_realtime_latency_frames(&context) == 0U);

    assert(verbx_plugin_realtime_process(&context, inputs, outputs, 4U, 2U, &params, &status) == 0);
    for (frame = 0U; frame < 4U; ++frame) {
        assert(out_l[frame] == in_l[frame]);
        assert(out_r[frame] == in_r[frame]);
    }
    assert_close(status.effective_rt60_seconds, sqrt(0.01 * 360.0) * 1.20, 1e-12);
    assert(status.freeze_enabled == 1);
    assert(status.reverse_enabled == 1);
    assert(status.quality_mode == VERBX_PLUGIN_QUALITY_TARGET_192K);
    assert(status.host_sample_rate == 48000U);
    assert(status.internal_sample_rate == 192000U);

    verbx_plugin_realtime_release(&context);

    config.channel_count = 0U;
    assert(verbx_plugin_realtime_prepare(&context, &config, error, sizeof(error)) != 0);
    assert(strstr(error, "channel_count") != NULL);

    return 0;
}
```

- [ ] **Step 2: Wire the failing test target**

Add to the existing `if(BUILD_TESTING)` block in `native/verbx_c/CMakeLists.txt`:

```cmake
  add_executable(test_plugin_realtime tests/test_plugin_realtime.c)
  target_link_libraries(test_plugin_realtime PRIVATE verbx_c_core)
  add_test(NAME plugin_realtime COMMAND test_plugin_realtime)
```

Run:

```bash
cmake --build build/native/verbx_c-plan
```

Expected: fail because `plugin_realtime.h` and `plugin_realtime.c` do not exist.

- [ ] **Step 3: Add the realtime API header**

Create `native/verbx_c/include/verbx_c/plugin_realtime.h`:

```c
#ifndef VERBX_C_PLUGIN_REALTIME_H
#define VERBX_C_PLUGIN_REALTIME_H

#include "verbx_c/plugin_params.h"

#include <stddef.h>

typedef struct {
    unsigned int host_sample_rate;
    size_t max_block_frames;
    size_t channel_count;
    verbx_plugin_quality_mode quality_mode;
} verbx_plugin_realtime_config;

typedef struct {
    double pre_delay_ms;
    double room_size;
    double rt60_coarse_normalized;
    double rt60_fine_bipolar;
    double damping;
    double width;
    double diffusion;
    double wet;
    double dry;
    int freeze;
    int reverse;
} verbx_plugin_realtime_params;

typedef struct {
    unsigned int host_sample_rate;
    unsigned int internal_sample_rate;
    size_t latency_frames;
    double effective_rt60_seconds;
    verbx_plugin_quality_mode quality_mode;
    int freeze_enabled;
    int reverse_enabled;
} verbx_plugin_realtime_status;

typedef struct {
    unsigned int host_sample_rate;
    unsigned int internal_sample_rate;
    size_t max_block_frames;
    size_t channel_count;
    size_t latency_frames;
    verbx_plugin_quality_mode quality_mode;
    int prepared;
} verbx_plugin_realtime_context;

int verbx_plugin_realtime_prepare(
    verbx_plugin_realtime_context *context,
    const verbx_plugin_realtime_config *config,
    char *error_message,
    size_t error_message_size
);

int verbx_plugin_realtime_process(
    verbx_plugin_realtime_context *context,
    const float *const *inputs,
    float *const *outputs,
    size_t frames,
    size_t channels,
    const verbx_plugin_realtime_params *params,
    verbx_plugin_realtime_status *status
);

void verbx_plugin_realtime_reset(verbx_plugin_realtime_context *context);
void verbx_plugin_realtime_release(verbx_plugin_realtime_context *context);
size_t verbx_plugin_realtime_latency_frames(const verbx_plugin_realtime_context *context);
unsigned int verbx_plugin_realtime_internal_sample_rate(const verbx_plugin_realtime_context *context);

#endif
```

- [ ] **Step 4: Add the realtime implementation**

Create `native/verbx_c/src/plugin_realtime.c`:

```c
#include "verbx_c/plugin_realtime.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *error_message, size_t error_message_size, const char *message) {
    if ((error_message == 0) || (error_message_size == 0U)) {
        return;
    }
    snprintf(error_message, error_message_size, "%s", message);
}

static unsigned int internal_rate_for_quality(unsigned int host_rate, verbx_plugin_quality_mode quality_mode) {
    if (quality_mode == VERBX_PLUGIN_QUALITY_2X) {
        return host_rate * 2U;
    }
    if (quality_mode == VERBX_PLUGIN_QUALITY_4X) {
        return host_rate * 4U;
    }
    if (quality_mode == VERBX_PLUGIN_QUALITY_TARGET_192K) {
        return host_rate >= 192000U ? host_rate : 192000U;
    }
    return host_rate;
}

int verbx_plugin_realtime_prepare(
    verbx_plugin_realtime_context *context,
    const verbx_plugin_realtime_config *config,
    char *error_message,
    size_t error_message_size
) {
    if ((context == 0) || (config == 0)) {
        set_error(error_message, error_message_size, "invalid realtime prepare arguments");
        return -1;
    }
    if (config->host_sample_rate == 0U) {
        set_error(error_message, error_message_size, "host_sample_rate must be non-zero");
        return -1;
    }
    if (config->max_block_frames == 0U) {
        set_error(error_message, error_message_size, "max_block_frames must be non-zero");
        return -1;
    }
    if (config->channel_count == 0U) {
        set_error(error_message, error_message_size, "channel_count must be non-zero");
        return -1;
    }

    memset(context, 0, sizeof(*context));
    context->host_sample_rate = config->host_sample_rate;
    context->internal_sample_rate = internal_rate_for_quality(config->host_sample_rate, config->quality_mode);
    context->max_block_frames = config->max_block_frames;
    context->channel_count = config->channel_count;
    context->latency_frames = 0U;
    context->quality_mode = config->quality_mode;
    context->prepared = 1;
    return 0;
}

int verbx_plugin_realtime_process(
    verbx_plugin_realtime_context *context,
    const float *const *inputs,
    float *const *outputs,
    size_t frames,
    size_t channels,
    const verbx_plugin_realtime_params *params,
    verbx_plugin_realtime_status *status
) {
    size_t channel;
    size_t frame;

    if ((context == 0) || (context->prepared == 0) || (inputs == 0) || (outputs == 0) || (params == 0)) {
        return -1;
    }
    if ((channels == 0U) || (channels > context->channel_count) || (frames > context->max_block_frames)) {
        return -1;
    }

    for (channel = 0U; channel < channels; ++channel) {
        if ((inputs[channel] == 0) || (outputs[channel] == 0)) {
            return -1;
        }
        for (frame = 0U; frame < frames; ++frame) {
            outputs[channel][frame] = inputs[channel][frame];
        }
    }

    if (status != 0) {
        status->host_sample_rate = context->host_sample_rate;
        status->internal_sample_rate = context->internal_sample_rate;
        status->latency_frames = context->latency_frames;
        status->effective_rt60_seconds = verbx_plugin_map_rt60_seconds(
            params->rt60_coarse_normalized,
            params->rt60_fine_bipolar
        );
        status->quality_mode = context->quality_mode;
        status->freeze_enabled = params->freeze ? 1 : 0;
        status->reverse_enabled = params->reverse ? 1 : 0;
    }

    return 0;
}

void verbx_plugin_realtime_reset(verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return;
    }
}

void verbx_plugin_realtime_release(verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return;
    }
    memset(context, 0, sizeof(*context));
}

size_t verbx_plugin_realtime_latency_frames(const verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return 0U;
    }
    return context->latency_frames;
}

unsigned int verbx_plugin_realtime_internal_sample_rate(const verbx_plugin_realtime_context *context) {
    if (context == 0) {
        return 0U;
    }
    return context->internal_sample_rate;
}
```

- [ ] **Step 5: Add realtime source to the core library**

Modify `native/verbx_c/CMakeLists.txt`:

```cmake
set(VERBX_C_CORE_SOURCES
  src/audio.c
  src/algo_reverb.c
  src/render.c
  src/wav_io.c
  src/plugin_params.c
  src/plugin_realtime.c
)
```

- [ ] **Step 6: Run realtime tests**

Run:

```bash
cmake --build build/native/verbx_c-plan
ctest --test-dir build/native/verbx_c-plan --output-on-failure
```

Expected:

```text
100% tests passed
```

- [ ] **Step 7: Commit**

```bash
git add native/verbx_c/CMakeLists.txt native/verbx_c/include/verbx_c/plugin_realtime.h native/verbx_c/src/plugin_realtime.c native/verbx_c/tests/test_plugin_realtime.c
git commit -m "Add realtime plugin context API"
```

---

### Task 3: Keep Direct Native Builds In Sync

**Files:**
- Modify: `scripts/build_verbx_c.sh`
- Modify: `tests/test_native_scaffold.py`

- [ ] **Step 1: Add the new sources to the pytest build helper**

Modify `_native_sources()` in `tests/test_native_scaffold.py`:

```python
def _native_sources(repo_root: Path) -> list[str]:
    return [
        str(repo_root / "native/verbx_c/src/audio.c"),
        str(repo_root / "native/verbx_c/src/algo_reverb.c"),
        str(repo_root / "native/verbx_c/src/render.c"),
        str(repo_root / "native/verbx_c/src/wav_io.c"),
        str(repo_root / "native/verbx_c/src/plugin_params.c"),
        str(repo_root / "native/verbx_c/src/plugin_realtime.c"),
        str(repo_root / "native/verbx_c/src/main.c"),
        str(repo_root / "native/verbx_c/src/cli.c"),
    ]
```

- [ ] **Step 2: Add the new sources to the shell build script**

Modify the compiler invocation in `scripts/build_verbx_c.sh`:

```bash
"${cc_bin}" "${cflags[@]}" \
  -I "${repo_root}/native/verbx_c/include" \
  "${repo_root}/native/verbx_c/src/audio.c" \
  "${repo_root}/native/verbx_c/src/algo_reverb.c" \
  "${repo_root}/native/verbx_c/src/render.c" \
  "${repo_root}/native/verbx_c/src/wav_io.c" \
  "${repo_root}/native/verbx_c/src/plugin_params.c" \
  "${repo_root}/native/verbx_c/src/plugin_realtime.c" \
  "${repo_root}/native/verbx_c/src/main.c" \
  "${repo_root}/native/verbx_c/src/cli.c" \
  "${ldflags[@]}" \
  -o "${exe}"
```

- [ ] **Step 3: Run focused Python native tests**

Run:

```bash
uv run pytest tests/test_native_scaffold.py::test_native_scaffold_builds_and_reports_version tests/test_native_scaffold.py::test_native_build_script_exposes_ergonomic_flags -q
```

Expected:

```text
2 passed
```

- [ ] **Step 4: Run the native build script**

Run:

```bash
./scripts/build_verbx_c.sh --clean --doctor
```

Expected: build succeeds and doctor prints process/error contract lines.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_verbx_c.sh tests/test_native_scaffold.py
git commit -m "Keep native direct builds aligned with plugin foundation"
```

---

### Task 4: Add Guarded JUCE Plug-in Shell Scaffold

**Files:**
- Create: `native/verbx_plugin/CMakeLists.txt`
- Create: `native/verbx_plugin/README.md`
- Create: `native/verbx_plugin/src/VerbXPluginProcessor.h`
- Create: `native/verbx_plugin/src/VerbXPluginProcessor.cpp`
- Create: `native/verbx_plugin/src/VerbXPluginEditor.h`
- Create: `native/verbx_plugin/src/VerbXPluginEditor.cpp`

- [ ] **Step 1: Add the guarded plug-in CMake project**

Create `native/verbx_plugin/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.22)

project(verbx_plugin LANGUAGES C CXX)

option(VERBX_ENABLE_JUCE_PLUGIN "Build the VERBX AU/AUv3/VST3 plug-in target" OFF)

if(NOT VERBX_ENABLE_JUCE_PLUGIN)
  message(STATUS "VERBX_ENABLE_JUCE_PLUGIN is OFF; JUCE plug-in target is not built.")
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(JUCE CONFIG REQUIRED)

add_subdirectory(../verbx_c verbx_c_core_build)

juce_add_plugin(VERBXPlugin
  COMPANY_NAME "VERBX"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE
  PLUGIN_MANUFACTURER_CODE Vrbx
  PLUGIN_CODE Vrbx
  FORMATS AU AUv3 VST3 Standalone
  PRODUCT_NAME "VERBX"
)

target_sources(VERBXPlugin
  PRIVATE
    src/VerbXPluginProcessor.cpp
    src/VerbXPluginEditor.cpp
)

target_include_directories(VERBXPlugin
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../verbx_c/include
)

target_link_libraries(VERBXPlugin
  PRIVATE
    verbx_c_core
    juce::juce_audio_utils
)

target_compile_definitions(VERBXPlugin
  PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)
```

- [ ] **Step 2: Add shell documentation**

Create `native/verbx_plugin/README.md`:

```markdown
# VERBX Plug-in Scaffold

This directory contains the opt-in AU/AUv3/VST3 shell for the VERBX plug-in track.

Default repository builds do not require JUCE. Configure this project with `VERBX_ENABLE_JUCE_PLUGIN=ON` only on machines where JUCE is available through CMake.

```bash
cmake -S native/verbx_plugin -B build/native/verbx_plugin
cmake -S native/verbx_plugin -B build/native/verbx_plugin-juce -DVERBX_ENABLE_JUCE_PLUGIN=ON
```

The plug-in shell consumes the C foundation in `native/verbx_c`:

- parameter manifest from `verbx_c/plugin_params.h`
- realtime context API from `verbx_c/plugin_realtime.h`
- default Target 192 kHz quality mode
- RT60 coarse/fine mapping from `0.01s` to `360s`
- visible Freeze and Reverse mode parameters
```

- [ ] **Step 3: Add the processor header**

Create `native/verbx_plugin/src/VerbXPluginProcessor.h`:

```cpp
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

extern "C" {
#include "verbx_c/plugin_realtime.h"
}

class VerbXPluginProcessor final : public juce::AudioProcessor {
public:
    VerbXPluginProcessor();
    ~VerbXPluginProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& state();

private:
    juce::AudioProcessorValueTreeState parameters_;
    verbx_plugin_realtime_context realtimeContext_{};

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    verbx_plugin_realtime_params currentRealtimeParams() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VerbXPluginProcessor)
};
```

- [ ] **Step 4: Add the processor implementation**

Create `native/verbx_plugin/src/VerbXPluginProcessor.cpp`:

```cpp
#include "VerbXPluginProcessor.h"
#include "VerbXPluginEditor.h"

#include <vector>

namespace {

juce::String paramId(verbx_plugin_parameter_id id) {
    const auto* parameter = verbx_plugin_parameter_by_id(id);
    return parameter != nullptr ? juce::String(parameter->key) : juce::String();
}

float valueFor(juce::AudioProcessorValueTreeState& state, verbx_plugin_parameter_id id) {
    auto* value = state.getRawParameterValue(paramId(id));
    return value != nullptr ? value->load() : 0.0f;
}

} // namespace

VerbXPluginProcessor::VerbXPluginProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                      .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters_(*this, nullptr, "VERBX", createParameterLayout()) {}

VerbXPluginProcessor::~VerbXPluginProcessor() {
    verbx_plugin_realtime_release(&realtimeContext_);
}

juce::AudioProcessorValueTreeState::ParameterLayout VerbXPluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> layout;
    for (size_t index = 0; index < verbx_plugin_parameter_count(); ++index) {
        const auto* parameter = verbx_plugin_parameter_at(index);
        const juce::String id(parameter->key);
        const juce::String label(parameter->label);
        if (parameter->kind == VERBX_PLUGIN_PARAMETER_BOOL) {
            layout.push_back(std::make_unique<juce::AudioParameterBool>(id, label, parameter->default_value >= 0.5));
        } else if (parameter->kind == VERBX_PLUGIN_PARAMETER_CHOICE) {
            juce::StringArray choices;
            choices.add("Host");
            choices.add("2x");
            choices.add("4x");
            choices.add("Target 192 kHz");
            layout.push_back(std::make_unique<juce::AudioParameterChoice>(id, label, choices, static_cast<int>(parameter->default_value)));
        } else {
            layout.push_back(std::make_unique<juce::AudioParameterFloat>(
                id,
                label,
                juce::NormalisableRange<float>(static_cast<float>(parameter->minimum), static_cast<float>(parameter->maximum)),
                static_cast<float>(parameter->default_value)
            ));
        }
    }
    return {layout.begin(), layout.end()};
}

void VerbXPluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    char error[256] = {};
    verbx_plugin_realtime_config config{};
    config.host_sample_rate = static_cast<unsigned int>(sampleRate);
    config.max_block_frames = static_cast<size_t>(samplesPerBlock > 0 ? samplesPerBlock : 1);
    config.channel_count = static_cast<size_t>(juce::jmax(getTotalNumInputChannels(), getTotalNumOutputChannels()));
    config.quality_mode = VERBX_PLUGIN_QUALITY_TARGET_192K;
    verbx_plugin_realtime_prepare(&realtimeContext_, &config, error, sizeof(error));
}

void VerbXPluginProcessor::releaseResources() {
    verbx_plugin_realtime_release(&realtimeContext_);
}

bool VerbXPluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& input = layouts.getMainInputChannelSet();
    const auto& output = layouts.getMainOutputChannelSet();
    return input == output && (output == juce::AudioChannelSet::mono() || output == juce::AudioChannelSet::stereo());
}

verbx_plugin_realtime_params VerbXPluginProcessor::currentRealtimeParams() const {
    auto& mutableState = const_cast<juce::AudioProcessorValueTreeState&>(parameters_);
    verbx_plugin_realtime_params params{};
    params.pre_delay_ms = valueFor(mutableState, VERBX_PLUGIN_PARAM_PRE_DELAY_MS);
    params.room_size = valueFor(mutableState, VERBX_PLUGIN_PARAM_ROOM_SIZE);
    params.rt60_coarse_normalized = valueFor(mutableState, VERBX_PLUGIN_PARAM_RT60_COARSE);
    params.rt60_fine_bipolar = valueFor(mutableState, VERBX_PLUGIN_PARAM_RT60_FINE);
    params.damping = valueFor(mutableState, VERBX_PLUGIN_PARAM_DAMPING);
    params.width = valueFor(mutableState, VERBX_PLUGIN_PARAM_WIDTH);
    params.diffusion = valueFor(mutableState, VERBX_PLUGIN_PARAM_DIFFUSION);
    params.wet = valueFor(mutableState, VERBX_PLUGIN_PARAM_WET);
    params.dry = valueFor(mutableState, VERBX_PLUGIN_PARAM_DRY);
    params.freeze = valueFor(mutableState, VERBX_PLUGIN_PARAM_FREEZE) >= 0.5f ? 1 : 0;
    params.reverse = valueFor(mutableState, VERBX_PLUGIN_PARAM_REVERSE) >= 0.5f ? 1 : 0;
    return params;
}

void VerbXPluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const auto channels = static_cast<size_t>(buffer.getNumChannels());
    const auto frames = static_cast<size_t>(buffer.getNumSamples());
    const float* inputs[64] = {};
    float* outputs[64] = {};

    for (size_t channel = 0; channel < channels && channel < 64U; ++channel) {
        inputs[channel] = buffer.getReadPointer(static_cast<int>(channel));
        outputs[channel] = buffer.getWritePointer(static_cast<int>(channel));
    }

    auto params = currentRealtimeParams();
    verbx_plugin_realtime_process(&realtimeContext_, inputs, outputs, frames, channels, &params, nullptr);
}

juce::AudioProcessorEditor* VerbXPluginProcessor::createEditor() {
    return new VerbXPluginEditor(*this);
}

bool VerbXPluginProcessor::hasEditor() const { return true; }
const juce::String VerbXPluginProcessor::getName() const { return "VERBX"; }
bool VerbXPluginProcessor::acceptsMidi() const { return false; }
bool VerbXPluginProcessor::producesMidi() const { return false; }
bool VerbXPluginProcessor::isMidiEffect() const { return false; }
double VerbXPluginProcessor::getTailLengthSeconds() const { return 360.0; }
int VerbXPluginProcessor::getNumPrograms() { return 1; }
int VerbXPluginProcessor::getCurrentProgram() { return 0; }
void VerbXPluginProcessor::setCurrentProgram(int index) { juce::ignoreUnused(index); }
const juce::String VerbXPluginProcessor::getProgramName(int index) { juce::ignoreUnused(index); return {}; }
void VerbXPluginProcessor::changeProgramName(int index, const juce::String& newName) { juce::ignoreUnused(index, newName); }

void VerbXPluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto stateXml = parameters_.copyState().createXml();
    copyXmlToBinary(*stateXml, destData);
}

void VerbXPluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    auto stateXml = getXmlFromBinary(data, sizeInBytes);
    if (stateXml != nullptr) {
        parameters_.replaceState(juce::ValueTree::fromXml(*stateXml));
    }
}

juce::AudioProcessorValueTreeState& VerbXPluginProcessor::state() {
    return parameters_;
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new VerbXPluginProcessor();
}
```

- [ ] **Step 5: Add the editor header**

Create `native/verbx_plugin/src/VerbXPluginEditor.h`:

```cpp
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

class VerbXPluginProcessor;

class VerbXPluginEditor final : public juce::AudioProcessorEditor {
public:
    explicit VerbXPluginEditor(VerbXPluginProcessor& processor);
    ~VerbXPluginEditor() override = default;

    void paint(juce::Graphics& graphics) override;
    void resized() override;

private:
    VerbXPluginProcessor& processor_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VerbXPluginEditor)
};
```

- [ ] **Step 6: Add the editor implementation**

Create `native/verbx_plugin/src/VerbXPluginEditor.cpp`:

```cpp
#include "VerbXPluginEditor.h"
#include "VerbXPluginProcessor.h"

VerbXPluginEditor::VerbXPluginEditor(VerbXPluginProcessor& processor)
    : AudioProcessorEditor(&processor), processor_(processor) {
    setResizable(true, true);
    setSize(1560, 920);
}

void VerbXPluginEditor::paint(juce::Graphics& graphics) {
    const auto bounds = getLocalBounds().toFloat();
    graphics.fillAll(juce::Colour::fromRGB(5, 7, 9));
    graphics.setColour(juce::Colour::fromRGB(140, 246, 210));
    graphics.setFont(juce::FontOptions(34.0f, juce::Font::bold));
    graphics.drawText("VERBX", 28, 24, 220, 48, juce::Justification::centredLeft);

    graphics.setColour(juce::Colour::fromRGB(180, 197, 200));
    graphics.setFont(juce::FontOptions(15.0f));
    graphics.drawText("Spatial Decay Theater - Target 192 kHz - RT60 0.01s to 360s",
                      28, 76, static_cast<int>(bounds.getWidth()) - 56, 28,
                      juce::Justification::centredLeft);

    graphics.setColour(juce::Colour::fromRGBA(140, 246, 210, 36));
    graphics.fillRoundedRectangle(bounds.reduced(28, 130), 24.0f);
    graphics.setColour(juce::Colour::fromRGBA(232, 240, 247, 48));
    graphics.drawRoundedRectangle(bounds.reduced(28, 130), 24.0f, 1.0f);

    graphics.setColour(juce::Colour::fromRGB(238, 247, 244));
    graphics.setFont(juce::FontOptions(22.0f, juce::Font::bold));
    graphics.drawText("Full-screen spatial console scaffold",
                      getLocalBounds().reduced(48, 160),
                      juce::Justification::centred);
}

void VerbXPluginEditor::resized() {
    juce::ignoreUnused(processor_);
}
```

- [ ] **Step 7: Verify default scaffold configuration does not require JUCE**

Run:

```bash
cmake -S native/verbx_plugin -B build/native/verbx_plugin-plan
```

Expected output includes:

```text
VERBX_ENABLE_JUCE_PLUGIN is OFF; JUCE plug-in target is not built.
```

- [ ] **Step 8: Commit**

```bash
git add native/verbx_plugin
git commit -m "Add guarded JUCE plugin scaffold"
```

---

### Task 5: Update Native Clamp And Documentation For 0.01s RT60

**Files:**
- Modify: `native/verbx_c/src/algo_reverb.c`
- Modify: `native/verbx_c/README.md`
- Modify: `docs/NATIVE_PARITY.md`
- Modify: `tests/test_native_scaffold.py`

- [ ] **Step 1: Add a regression test for native render accepting 0.01s RT60**

Add this test to `tests/test_native_scaffold.py`:

```python
def test_native_render_accepts_plugin_minimum_rt60(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)
    sr = 48_000
    audio = np.zeros((256, 1), dtype=np.float64)
    audio[0, 0] = 0.5
    infile = tmp_path / "short_rt60_in.wav"
    outfile = tmp_path / "short_rt60_out.wav"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    subprocess.run(
        [
            str(exe),
            "render",
            str(infile),
            str(outfile),
            "--rt60",
            "0.01",
            "--wet",
            "1.0",
            "--dry",
            "0.0",
            "--tail-hold-ms",
            "1",
            "--out-format",
            "float32",
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    rendered, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == sr
    assert rendered.shape[1] == 1
    assert np.all(np.isfinite(rendered))
```

- [ ] **Step 2: Run the failing focused test**

Run:

```bash
uv run pytest tests/test_native_scaffold.py::test_native_render_accepts_plugin_minimum_rt60 -q
```

Expected: fail because the current native core clamps RT60 to `0.1`.

- [ ] **Step 3: Lower the native DSP clamp**

Modify `native/verbx_c/src/algo_reverb.c`:

```c
    if (rt60 < 0.01) {
        rt60 = 0.01;
    }
```

- [ ] **Step 4: Update native docs**

Add this bullet under "What Works Today" in `native/verbx_c/README.md`:

```markdown
- plug-in foundation parameters: RT60 coarse/fine mapping supports `0.01s` to `360s`, with Freeze and Reverse represented as explicit mode parameters
```

Add this row to the matrix in `docs/NATIVE_PARITY.md`:

```markdown
| Plug-in foundation | Not applicable | Parameter manifest, RT60 mapping, realtime context API, guarded JUCE scaffold | Foundation slice | `ctest --test-dir build/native/verbx_c-plan --output-on-failure` |
```

- [ ] **Step 5: Run the native tests**

Run:

```bash
uv run pytest tests/test_native_scaffold.py -q
cmake --build build/native/verbx_c-plan
ctest --test-dir build/native/verbx_c-plan --output-on-failure
```

Expected:

```text
tests/test_native_scaffold.py passes
100% tests passed
```

- [ ] **Step 6: Commit**

```bash
git add native/verbx_c/src/algo_reverb.c native/verbx_c/README.md docs/NATIVE_PARITY.md tests/test_native_scaffold.py
git commit -m "Align native RT60 floor with plugin design"
```

---

### Task 6: Final Verification And Handoff

**Files:**
- Modify only if verification reveals a mismatch in files from earlier tasks.

- [ ] **Step 1: Run the focused verification suite**

Run:

```bash
uv run pytest tests/test_native_scaffold.py -q
cmake -S native/verbx_c -B build/native/verbx_c-plan
cmake --build build/native/verbx_c-plan
ctest --test-dir build/native/verbx_c-plan --output-on-failure
cmake -S native/verbx_plugin -B build/native/verbx_plugin-plan
./scripts/build_verbx_c.sh --clean --doctor
```

Expected:

```text
tests/test_native_scaffold.py passes
100% tests passed
VERBX_ENABLE_JUCE_PLUGIN is OFF; JUCE plug-in target is not built.
doctor prints process_contract and error_contract
```

- [ ] **Step 2: Inspect the final diff**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected: only intended files from this plan are modified or untracked.

- [ ] **Step 3: Commit any verification fixes**

If Step 1 required corrections, commit them:

```bash
git add native/verbx_c native/verbx_plugin scripts/build_verbx_c.sh tests/test_native_scaffold.py docs/NATIVE_PARITY.md
git commit -m "Stabilize plugin foundation verification"
```

If no corrections were needed, do not create an empty commit.

## Self-Review Checklist

- Spec coverage: The plan covers parameter manifest, RT60 coarse/fine `0.01s` to `360s`, default Target 192 kHz semantics, Freeze/Reverse mode parameters, realtime callback boundary, JUCE shell direction, documentation, and tests.
- Scope boundary: Full realtime FDN DSP, oversampling implementation, final full-screen editor, DXF import, and DAW validation are separate follow-up plans after this foundation passes.
- Type consistency: Parameter enum names, struct names, and function names are consistent across headers, implementations, tests, and JUCE scaffold snippets.
- Verification: The plan includes CTest, pytest, native script build, and default plug-in scaffold CMake configuration.
