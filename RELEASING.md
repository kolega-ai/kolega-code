# Releasing Kolega Code

This project publishes the `kolega-code` package to PyPI and serves the public
installer from `https://kolega.dev/install-kolega-code.sh`.

## First PyPI release

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

3. Confirm `pyproject.toml` has the release version.

4. Create and push a matching tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

5. Approve the `pypi` environment deployment in GitHub Actions.

6. Verify the release:

   ```bash
   uv tool install --force kolega-code
   kolega-code --version
   ```

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
