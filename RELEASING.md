# Releasing Kolega Code

This project publishes the `kolega-code` package to PyPI and serves the public
installer from `https://kolega.dev/install-kolega-code.sh`. Each published
version should also have a GitHub Release for its matching tag.

## Release process

Release version bumps must happen in a pull request. Do not tag an unmerged
release branch.

1. Create a release branch from the latest `main`:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b chore/release-v0.3.2
   ```

2. Update `pyproject.toml`, `uv.lock`, and package `__version__` values to the
   release version.

   Running `uv lock` advances the `exclude-newer` cutoff recorded in `uv.lock`,
   because `pyproject.toml` sets `exclude-newer = "1 week"` as a relative
   duration. The cutoff timestamp moving forward, and any dependency versions
   released within the new one-week window being pulled in, is expected and
   acceptable. Do not revert these changes.

3. Update `CHANGELOG.md` in the release PR before merging. Do not update the
   changelog after the tag or in a separate follow-up PR.

   - Move relevant entries from `Unreleased` into a new section for the release.
   - Use the release date from the release PR.
   - Keep entries concise and user-facing; GitHub Releases will still provide
     detailed generated notes from the merged pull requests.

4. Run the fast test suite:

   ```bash
   ./run_tests.sh
   ```

5. Commit the release bump and open a pull request against `main`:

   ```bash
   git commit -m "chore: release v0.3.2"
   git push -u origin chore/release-v0.3.2
   ```

   The PR must be reviewed, pass CI, and be merged before tagging.

6. After the PR is merged, update local `main`, then create and push a matching
   tag from the merge commit:

   ```bash
   git checkout main
   git pull origin main
   git tag v0.3.2
   git push origin v0.3.2
   ```

7. Confirm the `Release` GitHub Actions workflow completes. It builds and tests
   the package, publishes to PyPI, then creates the GitHub Release.

8. Verify the release:

   ```bash
   uv tool install --force kolega-code
   kolega-code --version
   gh release view v0.3.2 --repo kolega-ai/kolega-code
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
