"""Movfuscator Visualizer Server.

A minimal HTTP server that provides:
- GET /         : Serve visualizer.html
- POST /translate: Translate assembly source, return JSON metadata
- GET /samples  : List available sample programs
- GET /samples/<name>: Load a sample program's source
"""

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from lexer import Lexer, TokenType
from parser import Parser
from translator import Translator, OutputFormatter, TranslatorConfig
from translator.metadata import metadata_to_dict

VISUALIZER_PATH = SRC_DIR.parent / "visualizer.html"
SAMPLES_DIR = SRC_DIR.parent / "samples" / "in"
LUT_DIR = SRC_DIR.parent / "lut"


class VisualizerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the visualizer server."""

    def log_message(self, format, *args):
        """Override to use cleaner logging format."""
        print(f"[{self.log_date_time_string()}] {args[0]}")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self._serve_file(VISUALIZER_PATH, "text/html")
        elif self.path == "/samples":
            self._serve_samples_list()
        elif self.path.startswith("/samples/"):
            name = self.path[len("/samples/") :]
            self._serve_sample(name)
        else:
            self.send_error(404, f"Not found: {self.path}")

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/translate":
            self._handle_translate()
        else:
            self.send_error(404, f"Not found: {self.path}")

    def _handle_translate(self):
        """Translate assembly source and return JSON metadata."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            source = data.get("source", "")

            if not source.strip():
                self._send_json(400, {"error": "Empty source"})
                return

            # Run the translator with metadata
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            tokens = Lexer.filter_tokens(tokens, [TokenType.COMMENT])
            parser = Parser(tokens)
            program = parser.parse()

            config = TranslatorConfig(
                include_luts=False,
                include_scratch=False,
                lut_path=LUT_DIR,
            )
            translator = Translator(config)
            translated, metadata = translator.translate_with_metadata(program)

            formatter = OutputFormatter(config=config)
            result = metadata_to_dict(metadata, formatter)

            self._send_json(200, result)

        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
        except Exception as e:
            import traceback

            traceback.print_exc()
            self._send_json(400, {"error": str(e)})

    def _serve_samples_list(self):
        """Return list of available sample programs."""
        samples = sorted(f.stem for f in SAMPLES_DIR.glob("*.S"))
        self._send_json(200, {"samples": samples})

    def _serve_sample(self, name):
        """Serve a sample program's source code."""
        # Sanitize name to prevent path traversal
        name = Path(name).name

        # Try both .S and .s extensions
        for ext in (".S", ".s"):
            path = SAMPLES_DIR / f"{name}{ext}"
            if path.exists():
                self._serve_file(path, "text/plain")
                return
        self.send_error(404, f"Sample not found: {name}")

    def _serve_file(self, path: Path, content_type: str):
        """Serve a file with the given content type."""
        if not path.exists():
            self.send_error(404, f"File not found: {path}")
            return

        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, data: dict):
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def main():
    """Start the visualizer server."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    host = "localhost"

    server = HTTPServer((host, port), VisualizerHandler)
    print(f"Movfuscator Visualizer Server")
    print(f"=" * 40)
    print(f"Server running at: http://{host}:{port}")
    print(f"Visualizer:        http://{host}:{port}/")
    print(f"Samples API:       http://{host}:{port}/samples")
    print(f"Translate API:     POST http://{host}:{port}/translate")
    print(f"=" * 40)
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
