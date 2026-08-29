"""Load the environment selected for a host worker before runtime imports."""

from __future__ import annotations

import os

from dotenv import load_dotenv


SELECTED_ENV_FILE = os.getenv("HEXIS_ENV_FILE") or None
load_dotenv(SELECTED_ENV_FILE)
