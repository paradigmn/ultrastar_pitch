 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          version.py
@brief         project version number
@author        paradigm
"""

from datetime import datetime

__version__ = "dev_build_" + \
              str(datetime.date(datetime.now())).replace("-", "_") + \
              "_" + \
              str(datetime.time(datetime.now())).replace(":", "_").replace(".", "_")
