#!/usr/bin/env python3
"""
Batch script to apply memory pool pattern transformations.
Processes the remaining test files.
"""

import re
import sys

# File patterns for remaining files
REMAINING_FILES = {
    'syr2': {'types': 'TA, TX, TY', 'arrays': ['A', 'Aref', 'x', 'y'], 'device': ['dA', 'dx', 'dy'], 'sizes': ['A', 'x', 'y'], 'alloc_params': 'size_A, size_x, size_y'},
    'trmv': {'types': 'TA, TX', 'arrays': ['A', 'x', 'xref'], 'device': ['dA', 'dx'], 'sizes': ['A', 'x'], 'alloc_params': 'size_A, size_x'},
    'trsv': {'types': 'TA, TX', 'arrays': ['A', 'x', 'xref'], 'device': ['dA', 'dx'], 'sizes': ['A', 'x'], 'alloc_params': 'size_A, size_x'},
}

print(f"Remaining files to process: {len(REMAINING_FILES)}")
for name, info in REMAINING_FILES.items():
    print(f"  - test_{name}_device.cc: {info['arrays']}")

# Instructions
print("""
To apply the memory pool pattern to these files:

1. Read each test_*_device.cc file
2. Add memory pool structure before the test function
3. Replace manual memory allocation with pool->allocate()
4. Replace manual deallocation with comment
5. Update device specifics to use pool->queue, pool->dA, etc.

See test_hemv_device.cc, test_her_device.cc, test_her2_device.cc for reference patterns.
""")
