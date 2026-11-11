"""Ve Hinh 4.5 va 4.6 bang bo ve chung cua Chuong 4."""
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from bieu_do_chuong_4 import main

if __name__ == "__main__":
    main(("4.5", "4.6"), SCRIPT_DIR)
