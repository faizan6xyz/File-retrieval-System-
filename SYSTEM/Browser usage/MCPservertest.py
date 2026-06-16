import httpx

def is_mcp_running(port=3000):
    try:
        r = httpx.get(f"http://localhost:{port}/", timeout=2)
        print("✅ MCP server is running")
        return True
    except Exception:
        print("❌ MCP server is NOT running")
        return False

is_mcp_running()