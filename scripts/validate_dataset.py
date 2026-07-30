import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.validator import validate_dataset  # noqa: E402

if __name__ == "__main__":
    ann_path = ROOT / "data" / "raw" / "annotations.json"
    report = validate_dataset(ann_path, ROOT / "data" / "raw")
    print(report.summary())
    sys.exit(0 if report.is_valid else 1)
