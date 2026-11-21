# PyPI Publish Setup

`verbx` release workflow supports two publish auth modes:

- **API token mode** (`PYPI_API_TOKEN` secret)
- **Trusted publishing mode** (`PYPI_TRUSTED_PUBLISHING=true` repo variable + PyPI trusted publisher configuration)

Release policy is explicit:

- `RELEASE_REQUIRE_PYPI=true` (default) means release fails when publish auth is missing.
- `RELEASE_REQUIRE_PYPI=false` allows skip mode.

When required mode is enabled (default), GitHub Release publication is blocked
until the PyPI publishing job succeeds. The separate install-verification job
is not a dependency of GitHub release creation.

## Current Configuration (2026-10-17)

Trusted publishing is registered on PyPI with these exact identifiers:

| Setting | Value |
|---|---|
| Project | `verbx` |
| GitHub owner | `TheColby` |
| Repository | `verbx` |
| Workflow file | `release.yml` |
| GitHub environment | `pypi` |
| `PYPI_TRUSTED_PUBLISHING` | `true` |
| `RELEASE_REQUIRE_PYPI` | `true` |

No PyPI API token is needed. Keep the workflow's `id-token: write` permission.
The v0.9.10 publish job succeeded and its verification job installed the package
from PyPI. That job then failed on the obsolete `verbx --version` invocation;
development now uses `verbx version`. The corrected workflow needs validation
on the next tagged release.

For an existing tag, dispatch the release workflow using the tag as the ref:

```bash
gh workflow run release.yml --repo TheColby/verbx --ref v0.9.10
```

Check whether the version is already on PyPI before retrying publication;
PyPI does not permit replacing uploaded distributions. Do not move an existing
tag or treat `0.9.11.dev0` as a final release.

## Option 1: API token mode

1. Create a PyPI project-scoped token for `verbx`.
2. Configure GitHub secret:

```bash
gh secret set PYPI_API_TOKEN --repo TheColby/verbx
```

3. Re-run release workflow (or push next release tag).

## Option 2: Trusted publishing mode

1. Configure trusted publisher in PyPI for this GitHub repository/workflow.
2. Enable workflow mode:

```bash
gh variable set PYPI_TRUSTED_PUBLISHING --repo TheColby/verbx --body true
```

3. Re-run release workflow (or push next release tag).

## Notes

- `release.yml` resolves mode in this order: token -> trusted -> skip.
- `RELEASE_REQUIRE_PYPI` accepts `true` or `false` only.
- Keep `permissions.id-token: write` enabled for trusted publishing.
