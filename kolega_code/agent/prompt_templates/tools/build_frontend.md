Build the frontend defined by the project manifest (.kolega-manifest.yaml).

When to use this tool:
- When you need to compile, bundle, or otherwise build the frontend application
- When you want a consistent build execution that adapts to local or sandbox contexts

Guidance:
- Prefer this tool over manually running build commands in a terminal; it reads the manifest to choose the
  correct command and standardizes execution and output across environments

Returns:
    Build output as markdown (combined stdout/stderr)
