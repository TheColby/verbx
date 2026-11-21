# Render benchmark timing policy

The render baseline measures throughput after initialization. The runner executes
all four complete scenario paths once before measuring each scenario again with
fresh input/output files in a separate directory. This warms lazy imports and
Numba compilation for the exact algorithmic and convolution paths under test.
There is one measured run per scenario: no retry, minimum selection, or automatic
budget adjustment is applied.

JSON `scenarios` contains the measured results used for both relative and absolute
budget enforcement. `first_render_scenarios` retains the first-pass pipeline wall
times for startup and JIT triage; `timing_policy` records the warmup policy. These
first-pass times include rendering as well as initialization and are not isolated
compiler timings or whole-process startup timings. Existing disk caches may
already be warm, so a first render is not necessarily a cold compilation.

When comparing historical reports, check this metadata: reports predating this
policy may include first-use initialization in the first scenario's elapsed time.
The policy does not change baseline references or maximum budgets. A slowdown
that persists after warmup still fails the CI performance gate.
