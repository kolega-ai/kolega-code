# Releasing Kolega Code

This project publishes the `kolega-code` package to PyPI and serves the public
installer from `https://kolega.dev/install-kolega-code.sh`. Each published
version should also have a GitHub Release for its matching tag.

## Release process

1. Confirm `pyproject.toml`, `uv.lock`, and package `__version__` values have
   the release version.

2. Run the fast test suite:

   ```bash
   uv run pytest -ra --durations=50 --import-mode=importlib -m "not slow"
   ```

3. Commit the release bump:

   ```bash
   git commit -m "chore: release v0.3.0"
   ```

4. Create and push a matching tag:

   ```bash
   git tag v0.3.0
   git push origin main
   git push origin v0.3.0
   ```

5. Confirm the `Release` GitHub Actions workflow completes. It builds and tests
   the package, publishes to PyPI, then creates the GitHub Release.

6. Verify the release:

   ```bash
   uv tool install --force kolega-code
   kolega-code --version
   gh release view v0.3.0 --repo kolega-ai/kolega-code
   ```

The GitHub Release uses PyPI as the canonical package distribution and keeps
GitHub's automatic source archives as the only release assets.

## First PyPI release setup

The first release can be published from the existing maintainer PyPI user
account, then transferred to the Kolega PyPI organization after PyPI approves
the organization request.

1. In PyPI, create a pending Trusted Publisher under the maintainer user account.

   Use these values:

   - PyPI project name: `kolega-code`
   - GitHub owner: `kolega-ai`
   - GitHub repository: `kolega-code`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`

2. In GitHub, create an environment named `pypi`.

   Require approval from trusted maintainers before deployment. The release
   workflow will pause before publishing to PyPI.

## Installer handoff

The canonical installer source is tracked at:

```text
scripts/install-kolega-code.sh
```

The `kolega.dev` site repo should publish that file verbatim at:

```text
https://kolega.dev/install-kolega-code.sh
```

After updating the site, verify:

```bash
curl -fsSL https://kolega.dev/install-kolega-code.sh | sh
```

## Transfer to the PyPI organization

After the Kolega PyPI organization is approved:

1. Open the PyPI organization page.
2. Go to **Projects**.
3. Transfer the existing `kolega-code` project from the maintainer user account.
4. Confirm maintainers and teams have the expected permissions.
5. Re-check the `kolega-code` project **Publishing** page and confirm the
   Trusted Publisher still points at `kolega-ai/kolega-code` and `release.yml`.

The package name and install commands stay the same after transfer.
