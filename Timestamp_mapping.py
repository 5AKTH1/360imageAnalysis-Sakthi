#!/usr/bin/env python3
"""
Root entry point for Timestamp_mapping.py.
Redirects to PIPELINE.Timestamp_mapping.
"""
import os
import sys

# Ensure PIPELINE folder is in python path
pipeline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PIPELINE")
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

from Timestamp_mapping import main

if __name__ == "__main__":
    main()
