"""Put ``scripts/`` on the import path.

The analysis scripts are not a package, but some of them carry logic whose
failure mode is silent (loading the wrong checkpoint, say), which is exactly the
kind of thing that needs a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
