import sys
from pathlib import Path

# Ensure the project root is importable so `import smart_agent` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
