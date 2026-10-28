"""Interactive SDK client for 14_serving.py; endpoint/token file are configurable."""

import os

from fastrag.serve import CLI


def main():
    app = CLI(
        backend_url=os.environ.get("FASTRAG_BACKEND_URL", "http://127.0.0.1:8100"),
        token_file=os.environ.get("FASTRAG_CLI_TOKEN_FILE"),
    )
    app()


if __name__ == "__main__":
    main()
