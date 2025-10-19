#!/usr/bin/env python3
"""
Script to apply memory pool patterns to remaining device test files.
This helps track progress on which files have been completed.
"""

completed_files = [
    "test_herk_device.cc",
    "test_her2k_device.cc",
    "test_symm_device.cc",
    "test_syrk_device.cc",
    "test_syr2k_device.cc",
]

remaining_files = {
    # Level 3 BLAS - 2 files with A,B,Bref pattern
    "trmm": "test_trmm_device.cc",
    "trsm": "test_trsm_device.cc",

    # Auxiliary - 2 files with A,B,Bref pattern
    "geadd": "test_geadd_device.cc",
    "tzadd": "test_tzadd_device.cc",

    # Scalar - 2 files (special handling needed)
    "rotg": "test_rotg_device.cc",
    "rotmg": "test_rotmg_device.cc",

    # Batch - 9 files (special handling for batch arrays)
    "batch_gemm": "test_batch_gemm_device.cc",
    "batch_hemm": "test_batch_hemm_device.cc",
    "batch_her2k": "test_batch_her2k_device.cc",
    "batch_herk": "test_batch_herk_device.cc",
    "batch_symm": "test_batch_symm_device.cc",
    "batch_syr2k": "test_batch_syr2k_device.cc",
    "batch_syrk": "test_batch_syrk_device.cc",
    "batch_trmm": "test_batch_trmm_device.cc",
    "batch_trsm": "test_batch_trsm_device.cc",
}

print(f"Completed: {len(completed_files)} files")
print(f"Remaining: {len(remaining_files)} files")
print(f"Total: {len(completed_files) + len(remaining_files)} files")
