#!/usr/bin/env python3
"""
Simple HTTP server for serving the goalkeeper detection demo
Handles CORS and file serving for local development
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    port = 8000
    
    # Change to the js directory
    js_dir = Path(__file__).parent
    os.chdir(js_dir)
    
    print(f"🌐 Starting HTTP server on http://localhost:{port}")
    print(f"📁 Serving files from: {js_dir}")
    print(f"🎯 Open http://localhost:{port}/example.html to test the demo")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()