"""
MCP Server Entry Point

This file can be used to run the MCP server independently.
The server can run in STDIO mode (for Claude Desktop) or HTTP mode (for web clients).

Usage:
    # STDIO mode (default, for Claude Desktop):
    python mcp_server.py
    
    # HTTP mode (for web clients):
    python mcp_server.py --transport http --host 0.0.0.0 --port 8000
"""

import sys
import argparse
from main import mcp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackExpensio MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for HTTP/SSE transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP/SSE transport (default: 8000)"
    )
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        print("Starting MCP server in STDIO mode (for Claude Desktop)...")
        print("All tools are registered and ready.")
        mcp.run()
    elif args.transport == "http":
        print(f"Starting MCP server in HTTP mode on {args.host}:{args.port}...")
        print("All tools are registered and ready.")
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "sse":
        print(f"Starting MCP server in SSE mode on {args.host}:{args.port}...")
        print("All tools are registered and ready.")
        mcp.run(transport="sse", host=args.host, port=args.port)

