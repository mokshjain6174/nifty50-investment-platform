import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics import save_artifacts


if __name__ == "__main__":
    summary = save_artifacts()
    print("Artifacts generated")
    print(f"Stocks: {summary['stocks']}")
    print(f"Rows: {summary['rows']}")
    print(f"Date range: {summary['date_start']} to {summary['date_end']}")
