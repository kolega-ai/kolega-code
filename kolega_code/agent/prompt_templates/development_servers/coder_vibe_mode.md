The frontend development server is already running on port 9001. If it stops due to code errors, it will 
The backend development server is already running on port 9002.
If they stop due to code errors, they will be restarted automatically.

You never have to start them. Always assume they are available. They restart automatically when you update code or dependencies. NEVER KILL THEM AND RESTART THEM.

The logs for the backend are in backend.log. The logs for the fronted are in frontend.log. These can be long, so prefer reading the end of the file, not the entire file.

When running any program that binds to a port, you must ensure it uses one of the available ports.
The available ports are in the range {available_ports}

**IMPORTANT: Host Resolution for HTTP Requests**
Before making ANY HTTP requests or accessing development servers:
1. Use the `get_host` tool to obtain the correct hostname
2. Pass the port number you need to access
3. Use the returned hostname to construct your URLs

Example workflow:
- Call get_host(9001) to get the hostname  
- Use the result to construct URLs like: http://<hostname>/api/endpoint

This ensures your code works both locally and in cloud sandboxes.
