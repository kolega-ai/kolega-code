# Cloud Sandbox Implementation

This directory contains the cloud sandbox implementation for running AI agents in isolated environments using E2B.

## Architecture Overview

The sandbox system provides a pluggable architecture that allows agents to run in cloud-based isolated environments instead of locally. The implementation uses E2B as the sandbox provider but is designed to support other providers in the future.

### Key Components

1. **Base Interfaces** (`base.py`)
   - `SandboxConfig`: Configuration for sandbox creation (git URL, branch, environment variables)
   - `ProjectManifest`: Project configuration for dependencies and setup
   - `SandboxManager`: Abstract base class for sandbox management

2. **E2B Implementation** (`sandbox_e2b.py`)
   - `E2BSandboxManager`: Concrete implementation using E2B sandboxes
   - Handles Git authentication using GitLab group access tokens
   - Manages sandbox lifecycle (create, destroy, commit, push)
   - Mounts workspace files from S3 for agent access

3. **Sandbox-Aware Services**
   - `SandboxFileSystem`: FileSystem implementation that operates within sandbox
   - `SandboxTerminalManager`: Terminal manager for executing commands in sandbox
   - `SandboxBrowserManager`: Browser manager placeholder (not yet implemented)

4. **Utilities** (`utils.py`)
   - Helper functions for common sandbox operations
   - Git status parsing and diff generation
   - Project test execution

## How It Works

1. **Host Job Integration**: When a host application job starts and sandbox mode is enabled:
   - Creates an E2B sandbox with the workspace's Git repository
   - Clones the repository using GitLab group access token
   - Mounts workspace files from S3 as read-only
   - Installs dependencies if a manifest file exists
   - Injects sandbox-aware services into the agent

2. **Agent Execution**: The agent runs with:
   - Filesystem operations redirected to sandbox
   - Terminal commands executed in sandbox
   - Access to uploaded workspace files at `/home/user/workspace/kolega-project-files`
   - All changes isolated from local environment

3. **Git Workflow**: After agent completion:
   - Detects modified files using `git status`
   - Commits changes with descriptive message
   - Pushes changes back to GitLab repository

4. **Workspace Files**: Users can upload files through the UI which are:
   - Stored in S3 at `<bucket>/<workspace_id>/<filename>`
   - Automatically mounted in sandboxes at `/home/user/workspace/kolega-project-files`
   - Available as read-only to prevent accidental modifications
   - Excluded from git tracking via `.gitignore`

## Configuration

### Environment Variables

```bash
# Enable sandbox mode
USE_SANDBOX=true

# Sandbox provider (currently only 'e2b' supported)
SANDBOX_PROVIDER=e2b

# E2B API key from https://e2b.dev
E2B_API_KEY=your_e2b_api_key

# E2B template (must use custom template with s3fs)
E2B_TEMPLATE=your_custom_template_id

# GitLab group access token with read/write permissions
GITLAB_GROUP_ACCESS_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx

# S3 Configuration for workspace files
S3_BUCKET_NAME=your-kolega-files-bucket
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1  # Optional, defaults to us-east-1
```

### E2B Template Requirements

The E2B template must have s3fs installed. See `backend/e2b-template/` for the custom template that includes:
- Base E2B code interpreter
- s3fs for mounting S3 buckets
- Proper permissions setup

Build and deploy the template:
```bash
cd backend/e2b-template
e2b template build
# Update E2B_TEMPLATE env var with the new template ID
```

### Project Manifest

Projects can include a `.kolega-manifest.yaml` file in the repository root:

```yaml
name: my-project
runtime: node:18  # or python:3.11, etc.

# Optional: Commands to install dependencies
install_commands:
  - npm install

# Optional: Commands to run before install
environment_setup:
  - npm config set registry https://registry.npmjs.org/

# Optional: Development server command (runs automatically in background)
dev_server_command: npm run dev

# Optional: Test commands
test_commands:
  - npm test

# Optional: Build command
build_command: npm run build
```

#### Dev Server Auto-Start

When a `dev_server_command` is specified in the manifest, it will automatically start in the background after dependencies are installed. This is useful for:

- React/Vue/Angular development servers
- API servers (Express, FastAPI, etc.)
- Documentation servers
- Any long-running development process

Example configurations:

**React App:**
```yaml
name: my-react-app
runtime: node:18
install_commands:
  - npm install
dev_server_command: npm start -- --port 9001

```

**Python FastAPI:**
```yaml
name: my-api
runtime: python:3.11
install_commands:
  - pip install -r requirements.txt
dev_server_command: uvicorn main:app --reload --host 0.0.0.0 --port 9001

```

**Next.js App:**
```yaml
name: my-nextjs-app
runtime: node:18
install_commands:
  - npm install
dev_server_command: npm run dev -- -p 9001
```

**Minimal Project (no dependencies):**
```yaml
name: simple-static-site
runtime: node:18
# No install_commands needed for a simple static site
dev_server_command: python -m http.server 9001
```

## Usage

The sandbox is automatically used when:
1. `USE_SANDBOX=true` in environment
2. Workspace has a GitLab repository (`gitlab_project_url` is set)
3. Agent job is started by the host application

No code changes are required in agents - they continue using the standard FileSystem and TerminalManager interfaces.

### Accessing Workspace Files

Agents can access uploaded workspace files at:
```python
workspace_files = "/home/user/workspace/kolega-project-files"

# List files
import os
files = os.listdir(workspace_files)

# Read a file
with open(f"{workspace_files}/document.pdf", "rb") as f:
    content = f.read()
```

## Testing

The implementation includes comprehensive tests:
