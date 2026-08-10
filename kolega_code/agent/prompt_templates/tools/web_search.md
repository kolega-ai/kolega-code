Search the web and return a ranked list of results (title, URL, and a short snippet).

Use this to discover relevant pages for a query when you don't already know the URL.
The search backend (DuckDuckGo, Firecrawl, Tavily, or a self-hosted SearXNG instance)
is whatever the user configured in Settings; the default works without an API key. To
read a specific result in depth, follow up with the web_fetch tool on its URL.

Returns:
    A markdown list of results, or a message if no results were found.
