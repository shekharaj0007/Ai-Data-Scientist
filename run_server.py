"""Start the API server with .env loaded before uvicorn imports the app."""
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=True)

import uvicorn

if __name__ == "__main__":
    if not ENV_PATH.exists():
        print(f"WARNING: {ENV_PATH} not found — LLM insights will be disabled.")
    else:
        print(f"Loaded environment from {ENV_PATH.name}")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
