# -*- coding: utf-8 -*-
"""
Pytest configuration for SpillSense (SIH-26143).
Forces headless Agg backend for matplotlib across all tests and threads to prevent
Windows Tkinter Tcl_AsyncDelete crashes (0x80000003).
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
