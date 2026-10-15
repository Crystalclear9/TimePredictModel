"""Check/apply the patch against checksum-pinned upstream files, without GPU imports."""

import hashlib
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "patches/sglang-v0.5.19.json").read_text(encoding="utf-8"))
    checked = {}
    with tempfile.TemporaryDirectory(prefix="timepredict-patch-") as directory:
        target = Path(directory).resolve()
        for name, digest in manifest["upstream_sha256"].items():
            path = (target / name).resolve()
            if not path.is_relative_to(target):
                raise ValueError("upstream path outside verification directory")
            url = (
                f"https://raw.githubusercontent.com/sgl-project/sglang/{manifest['commit']}/{name}"
            )
            body = urllib.request.urlopen(url, timeout=30).read()
            actual = hashlib.sha256(body).hexdigest()
            if actual != digest:
                raise ValueError(f"upstream checksum mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            checked[name] = actual
        patch = str(ROOT / "patches/sglang-v0.5.19.patch")
        subprocess.run(
            ["git", "apply", "--check", patch], cwd=target, check=True, capture_output=True
        )
        subprocess.run(["git", "apply", patch], cwd=target, check=True, capture_output=True)
        for name in checked:
            compile((target / name).read_text(encoding="utf-8"), name, "exec")
    result = {
        "status": "passed",
        "commit": manifest["commit"],
        "upstream_sha256": checked,
        "patch_sha256": hashlib.sha256(Path(patch).read_bytes()).hexdigest(),
        "scope": "exact upstream hashes, git apply check, applied Python syntax; GPU runtime not verified",
    }
    output = ROOT / "outputs/sglang-patch-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
