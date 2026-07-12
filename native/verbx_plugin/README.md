# VERBX Plug-in Scaffold

This directory contains the opt-in C++/JUCE shell for the VERBX AU, AUv3,
VST3, and standalone plug-in track.

The host-visible plug-in vendor and author is **Colby Leider**. The stable
manufacturer code remains `Clby`, and the macOS bundle identifier is
`com.colbyleider.verbx`.

Fresh macOS configurations default to universal `arm64+x86_64` binaries with
a macOS 12 deployment target, allowing discovery by native Apple Silicon and
Rosetta-hosted DAWs.

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

Release artifacts are written beneath
`build/native/verbx_plugin-juce/VERBXPlugin_artefacts/Release/` as
`Standalone/VERBX.app`, `AU/VERBX.component`, `AUv3/VERBX.appex`, and
`VST3/VERBX.vst3`. The standalone container embeds the true AUv3 extension at
`VERBX.app/Contents/PlugIns/VERBX.appex`; install and sign the containing app
rather than copying the `.appex` into the AUv2 Components directory.

VERBX enables JUCE's AUv3 wrapper for standard macOS CMake generators and
supplies the app-extension entry point plus container embed step itself. This
keeps AUv3 builds available when Xcode's project generator is unavailable while
retaining the same JUCE processor, editor, parameter state, and DSP core.

The C++ shell consumes the realtime-safe C foundation in `native/verbx_c`:

![Compiled VERBX realtime spectrum analyzer](../../docs/assets/verbx_plugin_native_analyzer.jpg)

- parameter manifest from `verbx_c/plugin_params.h`
- realtime context API from `verbx_c/plugin_realtime.h`
- default Target 192 kHz / 32-bit-float processing contract
- RT60 coarse/fine mapping from `0.01s` to `360s`
- visible Freeze and Reverse mode parameters
- realtime post-DSP spectrum overlay with a lock-free audio handoff, 8192-point
  Hann FFT, logarithmic frequency grid, smoothed response, and peak trace

The current realtime core is an allocation-free mono/stereo Schroeder engine.
It implements the complete initial parameter slice, including pre-delay,
room-scaled delay geometry, logarithmic RT60, damping, diffusion, width,
wet/dry mixing, freeze-safe feedback, and a transient-triggered reverse-style
swell. Continuous DSP controls use 20 ms smoothing to prevent zipper noise
during host automation. The reverse mode is a zero-lookahead musical approximation, not offline
time reversal. Quality modes currently report the intended processing target;
production oversampling remains a separate parity slice.
