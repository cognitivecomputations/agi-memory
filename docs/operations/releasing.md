# Releasing

The whole release pipeline hangs off one action: pushing a version tag.
Nothing is published manually.

## The user journey this protects

```bash
uv tool install hexis   # gets CLI X.Y.Z
hexis up                # pulls ghcr images tagged X.Y.Z — same commit as the CLI
hexis upgrade           # updates the CLI package, then pulls the new pinned images, then migrates
```

The CLI pins `HEXIS_IMAGE_TAG` to its own package version when driving
`ops/docker-compose.runtime.yml`, so a PyPI release and its images are always
the same code. `latest` means "newest release" and is only the fallback (e.g.
installs from git where no version metadata exists).

## Cutting a release

1. Bump `version` in `pyproject.toml` (e.g. `1.0.6`).
2. Commit it to `main` (via your normal flow).
3. Tag and push:

   ```bash
   git tag v1.0.6
   git push origin v1.0.6
   ```

That single tag push makes `.github/workflows/publish.yml`:

- build + push multi-arch images `1.0.6`, `1.0`, `1`, and move `latest`
  (ghcr.io/quixiai/hexis-brain, -worker, -channels, -ui);
- build the sdist/wheel and publish `hexis==1.0.6` to PyPI — after a guard
  that refuses to publish if the tag and `pyproject.toml` version disagree.

Pushes to `main` between releases publish only the moving `edge` tag plus
immutable `sha-*` tags; they never touch `latest` or any version tag, so
users are never handed untagged tip-of-main code.

## One-time setup (already done? check before repeating)

PyPI publishing uses **trusted publishing** (OIDC) — no API tokens, no keys:
on pypi.org → project `hexis` → *Manage* → *Publishing* → *Add a new
publisher* → GitHub, with owner `QuixiAI`, repository `Hexis`, workflow
`publish.yml`, environment left blank. GitHub's runner then proves its
identity to PyPI per-run; nothing secret is stored anywhere.
