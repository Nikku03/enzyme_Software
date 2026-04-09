from __future__ import annotations

import sys

from enzyme_software.mainline.data.regime_builder import dispatch_regime_builder


if __name__ == "__main__":
    dispatch_regime_builder(sys.argv[1:])
