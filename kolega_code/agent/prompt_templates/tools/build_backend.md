Build the backend defined by the project manifest (.kolega-manifest.yaml).

When to use this tool:
- When you need to compile, bundle, or otherwise build the backend for the current workspace
- When verifying that the backend build still succeeds after code changes

Guidance:
- Prefer this tool over manually running build commands in a terminal; it automatically selects the correct
  command from the manifest and works in both local and sandbox environments with standardized output

Returns:
    Build output as markdown (combined stdout/stderr)
