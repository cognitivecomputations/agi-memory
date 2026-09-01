from __future__ import annotations

import json
import os
import subprocess
import sys


def test_selected_worker_env_loads_before_rabbitmq_constants(tmp_path):
    """A RabbitMQBridge resolves the selected worker env regardless of import
    order: settings are read live at construction time (core/rabbitmq_bridge.py's
    ``_runtime_value``), not baked into module-level constants at import time."""
    env_file = tmp_path / ".env.worker"
    env_file.write_text(
        "RABBITMQ_MANAGEMENT_PORT=49999\n"
        "RABBITMQ_DEFAULT_USER=host-user\n"
        "RABBITMQ_DEFAULT_PASS=host-password\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HEXIS_ENV_FILE"] = str(env_file)
    for name in (
        "RABBITMQ_MANAGEMENT_URL",
        "RABBITMQ_MANAGEMENT_PORT",
        "RABBITMQ_USER",
        "RABBITMQ_PASSWORD",
        "RABBITMQ_DEFAULT_USER",
        "RABBITMQ_DEFAULT_PASS",
    ):
        env.pop(name, None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; import services.worker_environment; "
                "import core.rabbitmq_bridge as bridge; "
                "b = bridge.RabbitMQBridge(None); "
                "print(json.dumps([b.management_url, b.user, b.password]))"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout) == [
        "http://127.0.0.1:49999",
        "host-user",
        "host-password",
    ]
