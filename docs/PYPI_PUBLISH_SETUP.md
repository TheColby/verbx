# PyPI Publish Setup

`verbx` release workflow supports two publish auth modes:

- **API token mode** (`PYPI_API_TOKEN` secret)
- **Trusted publishing mode** (`PYPI_TRUSTED_PUBLISHING=true` repo variable + PyPI trusted publisher configuration)

If neither is configured, the release workflow builds artifacts and creates the
GitHub release, then skips PyPI upload with an explicit summary message.

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
- Keep `permissions.id-token: write` enabled for trusted publishing.
