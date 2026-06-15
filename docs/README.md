# Kolega Code Documentation

The documentation site for **Kolega Code**, built with [Astro](https://astro.build/)
and [Starlight](https://starlight.astro.build/).

## Local development

```bash
cd docs
npm install
npm run dev
```

The dev server prints a local URL (default `http://localhost:4321/kolega-code`).

## Build

```bash
npm run build      # outputs static site to docs/dist/
npm run preview    # preview the production build locally
```

Starlight reports broken internal links at build time, so a clean `npm run build`
means navigation is intact.

## Project layout

```
docs/
├── astro.config.mjs          # Site config + sidebar
├── src/
│   └── content/
│       └── docs/             # All documentation pages (Markdown / MDX)
└── public/                   # Static assets (favicon, images)
```

To add a page, create a `.md`/`.mdx` file under `src/content/docs/` and add it to
the `sidebar` in `astro.config.mjs`.

## Deployment

Pushes to `main` that touch `docs/**` trigger `.github/workflows/docs.yml`, which
builds the site and publishes it to GitHub Pages.

> **One-time setup:** In the GitHub repository, go to **Settings → Pages** and set
> the **Source** to **GitHub Actions**.

The site is configured for `https://kolega-ai.github.io/kolega-code/`. If you deploy
elsewhere (custom domain, different path), update `site` and `base` in
`astro.config.mjs`.
