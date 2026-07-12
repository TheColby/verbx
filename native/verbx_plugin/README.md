# VERBX Plug-in Scaffold

This directory contains the opt-in C++/JUCE shell for the VERBX AU, AUv3,
VST3, and standalone plug-in track.

Default repository builds do not require JUCE. Configure this project with
`VERBX_ENABLE_JUCE_PLUGIN=ON` only on machines where JUCE is available through
CMake.

```bash
cmake -S native/verbx_plugin -B build/native/verbx_plugin
cmake -S native/verbx_plugin -B build/native/verbx_plugin-juce \
  -DVERBX_ENABLE_JUCE_PLUGIN=ON
```

If JUCE is available as a source checkout rather than an installed CMake
package, add `-DVERBX_JUCE_SOURCE_DIR=/path/to/JUCE`.

The C++ shell consumes the realtime-safe C foundation in `native/verbx_c`:

- parameter manifest from `verbx_c/plugin_params.h`
- realtime context API from `verbx_c/plugin_realtime.h`
- default Target 192 kHz / 32-bit-float processing contract
- RT60 coarse/fine mapping from `0.01s` to `360s`
- visible Freeze and Reverse mode parameters
- realtime post-DSP spectrum overlay with a lock-free audio handoff, 8192-point
  Hann FFT, logarithmic frequency grid, smoothed response, and peak trace

The current realtime core is deliberately pass-through-safe while the native
reverb DSP is moved behind the stateful callback boundary.
