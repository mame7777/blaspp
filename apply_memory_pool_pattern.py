#!/usr/bin/env python3
"""
Script to apply memory pool pattern to BLAS++ device test files.
This script analyzes each test file and applies the memory pool transformation.
"""

import re
import sys
from pathlib import Path

def analyze_memory_allocations(content):
    """Analyze a file to extract memory allocations."""
    # Find all new[] allocations in the setup section
    host_vars = []
    device_vars = []

    # Look for pattern: Type* var = new Type[size];
    host_pattern = r'(\w+(?:<[^>]+>)?)\*\s+(\w+)\s*=\s*new\s+\1\[\s*size_(\w+)\s*\]'
    for match in re.finditer(host_pattern, content):
        type_name, var_name, size_var = match.groups()
        host_vars.append((var_name, type_name, size_var))

    # Look for device_malloc calls
    device_pattern = r'(\w+)\s*=\s*blas::device_malloc<([^>]+)>\(\s*size_(\w+)'
    for match in re.finditer(device_pattern, content):
        var_name, type_name, size_var = match.groups()
        device_vars.append((var_name, type_name, size_var))

    # Look for optional device result pointers
    result_dev_pattern = r'result_ptr\s*=\s*blas::device_malloc<([^>]+)>\(\s*1\s*,'
    has_result_dev = bool(re.search(result_dev_pattern, content))

    return host_vars, device_vars, has_result_dev

def get_template_params(content):
    """Extract template parameters from function signature."""
    match = re.search(r'template\s*<([^>]+)>\s*\nvoid\s+test_(\w+)_device_work', content)
    if match:
        params = [p.strip().split()[-1] for p in match.group(1).split(',')]
        test_name = match.group(2)
        return params, test_name
    return [], None

def main():
    test_dir = Path('/home/morita/workspace/blaspp/test')

    # List of files to process (excluding test_axpy_device.cc which is already done)
    files_to_process = [
        'test_dot_device.cc',
        'test_nrm2_device.cc',
        'test_iamax_device.cc',
        'test_rot_device.cc',
        'test_rotg_device.cc',
        'test_rotm_device.cc',
        'test_rotmg_device.cc',
        # Add more as needed
    ]

    for filename in files_to_process:
        filepath = test_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename}: file not found")
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        template_params, test_name = get_template_params(content)
        host_vars, device_vars, has_result_dev = analyze_memory_allocations(content)

        print(f"\n{filename}:")
        print(f"  Template params: {template_params}")
        print(f"  Test name: {test_name}")
        print(f"  Host vars: {host_vars}")
        print(f"  Device vars: {device_vars}")
        print(f"  Has result_dev: {has_result_dev}")

if __name__ == '__main__':
    main()
