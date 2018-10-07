#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import numpy as np
import sys
import time
for i in range(100):
    for t in range(10):
        print("\r {}".format(t),end="") #主要是\r与end起作用了
        sys.stdout.flush()
        time.sleep(0.01)
    print("\n")
    print(i)

