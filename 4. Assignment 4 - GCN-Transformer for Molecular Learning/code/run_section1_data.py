import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.prepare import prepare_and_report

if __name__ == "__main__":
    prepare_and_report()
