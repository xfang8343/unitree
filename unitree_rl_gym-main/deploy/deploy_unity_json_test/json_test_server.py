#!/usr/bin/env python3
"""Standalone TCP JSON-Lines server for Unity network integration testing."""

import argparse
import json
import socket
import socketserver
from datetime import datetime, timezone


class JsonTestRequestHandler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send_json(self, payload):
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.wfile.write(encoded.encode("utf-8"))
        self.wfile.flush()
        print("TX {}: {}".format(self.peer, encoded.rstrip()), flush=True)

    def handle(self):
        self.peer = "{}:{}".format(*self.client_address)
        print("Unity test client connected: {}".format(self.peer), flush=True)
        self.send_json(
            {
                "type": "welcome",
                "message": "Python JSON test server connected",
                "server_time_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        try:
            for raw_line in self.rfile:
                try:
                    request = json.loads(raw_line.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("JSON message must be an object")
                    print(
                        "RX {}: {}".format(
                            self.peer, json.dumps(request, ensure_ascii=False)
                        ),
                        flush=True,
                    )
                    response = self.make_response(request)
                except Exception as exc:
                    response = {"type": "error", "error": str(exc)}
                self.send_json(response)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            print("Unity test client disconnected: {}".format(self.peer), flush=True)

    @staticmethod
    def make_response(request):
        message_type = request.get("type", "echo")
        sequence = request.get("seq")
        if message_type == "ping":
            return {"type": "pong", "seq": sequence, "message": "Python received Unity ping"}
        if message_type == "echo":
            return {"type": "echo", "seq": sequence, "received": request}
        return {
            "type": "ack",
            "seq": sequence,
            "message": "Python received Unity JSON",
            "received_type": message_type,
        }


class JsonTestServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="listen address")
    parser.add_argument("--port", type=int, default=9003, help="listen port")
    return parser.parse_args()


def main():
    args = parse_args()
    with JsonTestServer((args.host, args.port), JsonTestRequestHandler) as server:
        print("Unity JSON test server listening on {}:{}".format(args.host, args.port), flush=True)
        print("Protocol: one UTF-8 JSON object per line", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping JSON test server.", flush=True)


if __name__ == "__main__":
    main()
