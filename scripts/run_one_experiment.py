"""Train and evaluate one experiment, suitable for an isolated worker."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(config_name):
    config = ROOT / "configs" / config_name
    for module in ("src.train", "src.evaluate"):
        subprocess.run([sys.executable, "-m", module, "--config", str(config)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main(sys.argv[1])
