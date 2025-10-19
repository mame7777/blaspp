#!/usr/bin/env python3
"""
Script to apply memory pool pattern to device test files.
This script reads each test file, identifies memory allocations,
and generates the appropriate memory pool structure.
"""

import re
import sys
from pathlib import Path

# Template for memory pool structure (to be customized per file)
MEMORY_POOL_TEMPLATE = """
// -----------------------------------------------------------------------------
// Memory pool structure for device memory management
template <{template_params}>
struct {pool_name} {{
    // Host memory
{host_members}
    // Device memory
{device_members}
    // Memory sizes
{size_members}
    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate({alloc_params}) {{
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {{
            free();
            need_realloc = true;
        }}

        // Check if we need to allocate larger memory
        if ({first_ptr} == nullptr{size_checks}) {{
            if ({first_ptr} != nullptr) {{
                free();
            }}
            need_realloc = true;
        }}

        // Allocate new memory if needed
        if (need_realloc || {first_ptr} == nullptr) {{
            // Use larger size to avoid frequent reallocation
{size_updates}
            device_id = dev;

{host_allocs}
            if (queue != nullptr) {{
                delete queue;
            }}
            queue = new blas::Queue(device_id);

{device_allocs}
        }}
{extra_allocs}
    }}

    // Free all memory
    void free() {{
        if ({first_ptr} != nullptr) {{
{host_frees}
{host_nulls}
        }}

        if ({first_dptr} != nullptr && queue != nullptr) {{
{device_frees}
{device_nulls}
        }}

{extra_frees}
        if (queue != nullptr) {{
            delete queue;
            queue = nullptr;
        }}

{size_resets}
        device_id = -1;
    }}

    ~{pool_name}() {{
        free();
    }}
}};

// Static memory pools for each data type
{static_pools}
"""

def analyze_file(filepath):
    """Analyze a test file to extract memory allocation patterns."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the test function name
    match = re.search(r'void (test_\w+_device_work)\s*\(', content)
    if not match:
        return None

    func_name = match.group(1)
    test_name = func_name.replace('_device_work', '').replace('test_', '')

    # Find template parameters
    match = re.search(r'template\s*<(.+?)>\s*void\s+' + re.escape(func_name), content)
    if not match:
        return None

    template_params = match.group(1).strip()

    # Find host memory allocations (new Type[])
    host_allocs = re.findall(r'(\w+)\*\s+(\w+)\s*=\s*new\s+\1\[\s*(\w+)\s*\];', content)

    # Find device memory allocations (device_malloc)
    device_allocs = re.findall(r'(\w+)\*\s+(\w+);\s*\n\s*\2\s*=\s*blas::device_malloc<(\w+)>\(\s*(\w+)', content)

    return {
        'func_name': func_name,
        'test_name': test_name,
        'template_params': template_params,
        'host_allocs': host_allocs,
        'device_allocs': device_allocs,
        'content': content
    }

def main():
    test_dir = Path('/home/morita/workspace/blaspp/test')

    # List of files to process
    todo_files = [
        'test_nrm2_device.cc',
        'test_iamax_device.cc',
        'test_rot_device.cc',
        'test_rotg_device.cc',
        'test_rotm_device.cc',
        'test_rotmg_device.cc',
    ]

    for filename in todo_files:
        filepath = test_dir / filename
        print(f"Analyzing {filename}...")

        info = analyze_file(filepath)
        if info:
            print(f"  Function: {info['func_name']}")
            print(f"  Template: {info['template_params']}")
            print(f"  Host allocs: {info['host_allocs']}")
            print(f"  Device allocs: {info['device_allocs']}")
        else:
            print(f"  Could not analyze {filename}")

if __name__ == '__main__':
    main()
