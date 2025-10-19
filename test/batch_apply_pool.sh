#!/bin/bash
# Batch script to apply memory pool pattern to remaining files

# List of remaining files to process
files=(
    "test_syr_device.cc"
    "test_syr2_device.cc"
    "test_trmv_device.cc"
    "test_trsv_device.cc"
    "test_hemm_device.cc"
    "test_herk_device.cc"
    "test_her2k_device.cc"
    "test_symm_device.cc"
    "test_syrk_device.cc"
    "test_syr2k_device.cc"
    "test_trmm_device.cc"
    "test_trsm_device.cc"
    "test_geadd_device.cc"
    "test_tzadd_device.cc"
)

echo "Files to process: ${#files[@]}"
for file in "${files[@]}"; do
    echo "- $file"
done
