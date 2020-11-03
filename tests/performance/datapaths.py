from pathlib import Path
import os

performance_root_path = Path(os.getenv("RESISTICS_PERF"))
performance_project = performance_root_path / "lemiProject"