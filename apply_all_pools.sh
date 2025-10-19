#!/bin/bash
# Script to apply memory pool pattern to all remaining device test files

cd /home/morita/workspace/blaspp/test

# List of files that still need to be modified
FILES=(
    "test_nrm2_device.cc"
    "test_iamax_device.cc"
    "test_rot_device.cc"
    "test_rotg_device.cc"
    "test_rotm_device.cc"
    "test_rotmg_device.cc"
    "test_gemm_device.cc"
    "test_gemv_device.cc"
    "test_ger_device.cc"
    "test_hemm_device.cc"
    "test_hemv_device.cc"
    "test_her_device.cc"
    "test_her2_device.cc"
    "test_herk_device.cc"
    "test_her2k_device.cc"
    "test_symm_device.cc"
    "test_symv_device.cc"
    "test_syr_device.cc"
    "test_syr2_device.cc"
    "test_syrk_device.cc"
    "test_syr2k_device.cc"
    "test_trmm_device.cc"
    "test_trmv_device.cc"
    "test_trsm_device.cc"
    "test_trsv_device.cc"
    "test_geadd_device.cc"
    "test_tzadd_device.cc"
    "test_batch_gemm_device.cc"
    "test_batch_hemm_device.cc"
    "test_batch_her2k_device.cc"
    "test_batch_herk_device.cc"
    "test_batch_symm_device.cc"
    "test_batch_syr2k_device.cc"
    "test_batch_syrk_device.cc"
    "test_batch_trmm_device.cc"
    "test_batch_trsm_device.cc"
)

echo "Files to process: ${#FILES[@]}"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
    else
        echo "WARNING: $file not found"
    fi
done
