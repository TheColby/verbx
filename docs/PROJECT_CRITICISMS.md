# Project Criticisms (Fresh Audit)

Date: 2026-03-26

1. The CLI is still a monolith doing orchestration, validation, serialization, and UX in one file.
   - Evidence: `src/verbx/cli.py:737`, `src/verbx/cli.py:3984`

2. `render` is a mega-command with too many flags and too much inline mapping logic, increasing regression risk.
   - Evidence: `src/verbx/cli.py:738`, `src/verbx/cli.py:1501`

3. `RenderConfig` is a god object with a very large field surface, creating tight coupling between concerns.
   - Evidence: `src/verbx/config.py:43`, `src/verbx/config.py:231`

4. Parameter semantics are inconsistent across layers (RT60 defaults/limits differ across docs, CLI, and core validation).
   - Evidence: `src/verbx/cli.py:751`, `src/verbx/config.py:53`, `src/verbx/config.py:87`, `docs/SCHEMA_REFERENCE.md:59`, `src/verbx/core/control_targets.py:9`

5. The "stable" Python API is weakly typed (`Any` in key signatures), which undermines API stability for integrators.
   - Evidence: `src/verbx/api.py:48`, `src/verbx/api.py:100`

6. There is no dedicated API test module, so Python API contract drift can go undetected.
   - Evidence: `tests/` contains no `test_api.py`; coverage is concentrated in `tests/test_cli.py`

7. Type-checking is configured as strict while disabling important unknown-type diagnostics; `Any` remains heavy in core CLI paths.
   - Evidence: `pyproject.toml:72`, `pyproject.toml:80`, `src/verbx/cli.py:535`

8. Version/schema metadata is fragmented and stringly-typed (`"0.5"`, `"0.7"` in multiple payloads), increasing migration risk.
   - Evidence: `src/verbx/cli.py:3031`, `src/verbx/cli.py:3413`, `src/verbx/cli.py:3913`, `src/verbx/core/immersive.py:558`

9. Release automation can report success while skipping PyPI/Homebrew publication when auth is unconfigured, making release status ambiguous.
   - Evidence: `.github/workflows/release.yml:66`, `.github/workflows/release.yml:78`, `.github/workflows/release.yml:107`, `.github/workflows/release.yml:119`

10. Documentation and UX surface area is huge and manually synchronized; drift risk is explicitly acknowledged in roadmap.
    - Evidence: `README.md:1`, `docs/EXTREME_COOKBOOK.md:1`, `docs/ROADMAP.md:74`
