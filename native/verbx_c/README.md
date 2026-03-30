# verbx-c

`verbx-c` is the native C executable track for `verbx` `v0.8`.

This directory currently contains the first scaffold only:

- native build configuration
- standalone `verbx-c` executable target
- core CLI command dispatch (`help`, `version`, `doctor`)
- a deliberate placeholder for `render`

The Python application in `src/verbx/` remains the released implementation for
`v0.7.x`. The native track is where the executable rewrite starts.

## Build

With a plain C compiler:

```bash
./scripts/build_verbx_c.sh
./build/native/verbx_c/verbx-c version
```

With CMake:

```bash
cmake -S native/verbx_c -B build/native/verbx_c
cmake --build build/native/verbx_c
./build/native/verbx_c/verbx-c doctor
```

## Immediate goals

- stabilize CLI contract and process model
- define deterministic WAV I/O and offline render lifecycle in C
- port the algorithmic render core in small, testable pieces
- keep regression parity with the `v0.7.x` Python renderer during the migration
