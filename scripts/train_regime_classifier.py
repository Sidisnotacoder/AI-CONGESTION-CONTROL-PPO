import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from rl.regime_classifier import train, save

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_CSV = os.path.join(_PROJECT_ROOT, "data", "raw", "network_metrics.csv")
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "regime_classifier.joblib")


def main():
    if not os.path.exists(DATA_CSV):
        raise SystemExit(
            f"{DATA_CSV} not found. Regenerate it first: "
            f"sudo python3 scripts/collect_dataset.py "
            f"(see README.md for the quick smoke-test settings)."
        )

    logger.info("Training regime classifier + direction regressor from %s", DATA_CSV)
    classifier, regressor, held_out_accuracy = train(DATA_CSV)
    logger.info("Held-out classifier accuracy: %.3f", held_out_accuracy)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save(classifier, regressor, MODEL_PATH)
    logger.info("Saved -> %s", MODEL_PATH)


if __name__ == "__main__":
    main()
