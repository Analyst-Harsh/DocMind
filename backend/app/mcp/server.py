# app/mcp/server.py
from app.mcp import tools  # noqa: F401 -- registers the 4 tools against `mcp`
from app.mcp.instance import mcp


def main() -> None:
    mcp.run(transport="stdio")  # explicit, not relying on the SDK's
    # default -- the transport choice is also this server's trust boundary


if __name__ == "__main__":
    main()
