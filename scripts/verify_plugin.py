"""Write source-bound local test evidence without making production acceptance claims."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("TIMEPREDICT_TEST_ENCODER", None)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    roots = ["src", "tests", "scripts", "deploy", "examples"]
    commands = {
        "ruff_lint": [sys.executable, "-m", "ruff", "check", *roots],
        "ruff_format": [sys.executable, "-m", "ruff", "format", "--check", *roots],
        "pytest": [sys.executable, "-m", "pytest", "-q", "--tb=short"],
    }
    checks = {}
    for name, command in commands.items():
        run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        checks[name] = {
            "status": "passed" if run.returncode == 0 else "failed",
            "output": run.stdout + run.stderr,
        }
    result = {
        "generated_at": time.time(),
        "source_sha256": digest.hexdigest(),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "checks": checks,
        "evidence_scope": "local protocol/CPU model/API tests; fakeredis and acceptance fixtures",
        "server_acceptance_verified": False,
        "production_publish_allowed": False,
    }
    output = ROOT / "outputs" / "plugin-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(check["status"] == "passed" for check in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
