{% if context.workspace_environment_variables %}
The following workspace environment variables are available. Use `os.getenv` (or the equivalent in your language) to read their values at runtime:

{% for var_name, var_description in context.workspace_environment_variables.items() | sort %}
- `{{ var_name }}`: {{ var_description }}
{% endfor %}

Values are injected into the sandbox automatically. Never write secrets to source files or logs.
{% else %}
No workspace-specific environment variables are configured.
{% endif %}
