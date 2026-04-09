#!/usr/bin/env python3
"""
Start an ngrok tunnel to expose the frontend (NGINX on port 80).

Usage:
    python3 tunnel.py <YOUR_NGROK_AUTHTOKEN>

Or set the environment variable:
    export NGROK_AUTHTOKEN=<your_token>
    python3 tunnel.py
"""

import sys
import time

from pyngrok import conf, ngrok


def main():
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        import os
        token = os.environ.get("NGROK_AUTHTOKEN")

    if not token:
        print("Usage: python3 tunnel.py <NGROK_AUTHTOKEN>")
        print("  Or:  export NGROK_AUTHTOKEN=<token> && python3 tunnel.py")
        sys.exit(1)

    conf.get_default().auth_token = token

    tunnel = ngrok.connect(80, "http")
    print(f"\n{'='*50}")
    print(f"  Frontend available at: {tunnel.public_url}")
    print(f"{'='*50}\n")
    print("Press Ctrl+C to stop the tunnel.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping tunnel...")
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        print("Tunnel stopped.")


if __name__ == "__main__":
    main()
