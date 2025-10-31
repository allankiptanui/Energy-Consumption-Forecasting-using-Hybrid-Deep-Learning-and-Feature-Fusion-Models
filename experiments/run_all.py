 
import subprocess
import os
from pathlib import Path
from datetime import datetime
from src.utils.logger import get_logger

# Directories
RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set up logger
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = LOG_DIR / f"run_all_{timestamp}.txt"
logger = get_logger("run_all", logfile=str(logfile))

# Models to train sequentially
MODELS = ["cnn_gru_attn", "lstm", "gru", "arima"]

logger.info(" Starting batch experiment runs...")
for model in MODELS:
    logger.info(f"=== Running model: {model} ===")
    cmd = [
        "python",
        "experiments/run_training_workflow.py",
        "--model", model,
        "--use_ucimlrepo", "True"
    ]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f" Completed {model}")
    except subprocess.CalledProcessError as e:
        logger.error(f"rror running {model}: {e}")

logger.info(" All experiments finished.")
print(f"Results and logs saved under: {RESULTS_DIR.resolve()}")
summary_path = RESULTS_DIR / "outputs_summary.txt"
with open(summary_path, "w") as out:
    for f in (RESULTS_DIR / "metrics").glob("*.json"):
        out.write(f"{f.name}\n")
        with open(f, "r") as jf:
            out.write(jf.read() + "\n\n")
logger.info(f"📄 Combined summary written to {summary_path}")
