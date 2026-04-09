from __future__ import annotations

import sys

from enzyme_software.mainline.data.master_builder import dispatch_master_builder


if __name__ == "__main__":
    dispatch_master_builder(sys.argv[1:])
