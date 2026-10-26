"""Run and evaluate every registered experiment sequentially."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "experiment_busbra_image.yaml",
    "experiment_busi_image.yaml",
    "experiment_combined_image.yaml",
    "experiment_busbra_multimodal.yaml",
    "experiment_synthetic_augmented.yaml",
]


def run(module, config):
    command = [sys.executable, "-m", module, "--config", str(ROOT / "configs" / config)]
    print(f"\n=== {module}: {config} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_experiments.py")], cwd=ROOT, check=True)
    for config in CONFIGS:
        run("src.train", config)
        run("src.evaluate", config)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_experiments.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
