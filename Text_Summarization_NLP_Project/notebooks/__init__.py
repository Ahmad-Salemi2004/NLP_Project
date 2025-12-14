import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("📓 Notebooks module loaded")
print(f"📁 Project root: {project_root}")
