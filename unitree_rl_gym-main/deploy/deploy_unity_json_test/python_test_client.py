#!/usr/bin/env python3
"""Python client used to verify the standalone Unity JSON test server."""

import argparse
import json
import socket


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9003)
    return parser.parse_args()


def main():
    args = parse_args()
    with socket.create_connection((args.host, args.port), timeout=5.0) as connection:
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        writer = connection.makefile("w", encoding="utf-8", newline="\n")

        print("RX welcome:", reader.readline().strip())
        writer.write(json.dumps({"type": "ping", "seq": 1}) + "\n")
        writer.flush()
        print("TX ping:  {\"type\": \"ping\", \"seq\": 1}")
        print("RX pong:  ", reader.readline().strip())


if __name__ == "__main__":
    main()
