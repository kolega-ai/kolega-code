When running development servers or starting any program that binds to a port, you must ensure it uses one of the available ports.
The available ports are in the range {available_ports}

**IMPORTANT: Host Resolution for HTTP Requests**
Before making ANY HTTP requests or accessing development servers:
1. Use the `get_host` tool to obtain the correct hostname
2. Pass the port number you need to access
3. Use the returned hostname to construct your URLs

Example workflow:
- Start a dev server on port 9001
- Call get_host(9001) to get the hostname  
- Use the result to construct URLs like: http://<hostname>/api/endpoint

This ensures your code works both locally and in cloud sandboxes.

If you need to serve a static website, use Python's built in http server.
