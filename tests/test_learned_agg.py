"""
Quick test script for learned aggregation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sb.models.learned_agg import sanity_check

if __name__ == "__main__":
    sanity_check()
