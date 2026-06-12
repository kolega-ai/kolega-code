// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  // Update these if the docs are served from a different host/path.
  // For GitHub Pages at https://kolega-ai.github.io/kolega-code/ keep the
  // `base` set to "/kolega-code".
  site: "https://kolega-ai.github.io",
  base: "/kolega-code",
  integrations: [
    starlight({
      title: "Kolega Code",
      description:
        "A local-first, terminal AI coding agent. Plan, build, and ship from your CLI.",
      // The Kolega "</>" mark, alongside the "Kolega Code" wordmark in the nav.
      logo: {
        light: "./src/assets/kolega-light.svg",
        dark: "./src/assets/kolega-dark.svg",
        alt: "Kolega Code",
      },
      // Brand fonts (Geist) + the Kolega Code theme. Order matters: fonts first
      // so the brand stylesheet can reference their family names.
      customCss: [
        "@fontsource-variable/geist/index.css",
        "@fontsource-variable/geist-mono/index.css",
        "./src/styles/brand.css",
      ],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/kolega-ai/kolega-code",
        },
      ],
      // Surface "Edit this page" links back to the repo.
      editLink: {
        baseUrl:
          "https://github.com/kolega-ai/kolega-code/edit/main/docs/",
      },
      lastUpdated: true,
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Introduction", slug: "getting-started/introduction" },
            { label: "Installation", slug: "getting-started/installation" },
            { label: "Quick Start", slug: "getting-started/quick-start" },
          ],
        },
        {
          label: "Configuration",
          items: [
            {
              label: "Providers & Models",
              slug: "configuration/providers-and-models",
            },
            {
              label: "Settings & API Keys",
              slug: "configuration/settings-and-api-keys",
            },
            {
              label: "Environment Variables",
              slug: "configuration/environment-variables",
            },
          ],
        },
        {
          label: "CLI Reference",
          items: [
            { label: "Overview", slug: "cli/overview" },
            { label: "ask", slug: "cli/ask" },
            { label: "sessions", slug: "cli/sessions" },
            { label: "doctor", slug: "cli/doctor" },
          ],
        },
        {
          label: "The Terminal UI",
          items: [
            { label: "Interface Tour", slug: "tui/interface" },
            { label: "Build & Plan Modes", slug: "tui/modes" },
            { label: "Chat Composer", slug: "tui/composer" },
            { label: "Slash Commands", slug: "tui/slash-commands" },
            { label: "Sessions & Resuming", slug: "tui/sessions-and-resume" },
          ],
        },
        {
          label: "Skills",
          items: [{ label: "Agent Skills", slug: "skills" }],
        },
        {
          label: "How It Works",
          items: [
            { label: "Architecture", slug: "concepts/how-it-works" },
            { label: "Agents", slug: "concepts/agents" },
            { label: "Tools", slug: "concepts/tools" },
          ],
        },
      ],
    }),
  ],
});
