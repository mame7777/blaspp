#!/usr/bin/env python3
"""
Script to apply memory pool pattern to BLAS device test files.
"""
import re
import sys

def get_memory_pool_struct_2vec(name, ta="TA", tx="TX", ty="TY"):
    """Generate memory pool struct for matrix + 2 vectors (A, x, y/yref)"""
    return f"""// -----------------------------------------------------------------------------
// Memory pool structure for device memory management
template <typename {ta}, typename {tx}, typename {ty}>
struct {name} {{
    // Host memory
    {ta}* A = nullptr;
    {tx}* x = nullptr;
    {ty}* y = nullptr;
    {ty}* yref = nullptr;

    // Device memory
    {ta}* dA = nullptr;
    {tx}* dx = nullptr;
    {ty}* dy = nullptr;

    // Memory sizes
    size_t size_A = 0;
    size_t size_x = 0;
    size_t size_y = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sA, size_t sx, size_t sy, int64_t dev) {{
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {{
            free();
            need_realloc = true;
        }}

        // Check if we need to allocate larger memory
        if (A == nullptr || size_A < sA || size_x < sx || size_y < sy) {{
            if (A != nullptr) {{
                free();
            }}
            need_realloc = true;
        }}

        // Allocate new memory if needed
        if (need_realloc || A == nullptr) {{
            // Use larger size to avoid frequent reallocation
            size_A = (sA > size_A) ? sA : size_A;
            size_x = (sx > size_x) ? sx : size_x;
            size_y = (sy > size_y) ? sy : size_y;
            device_id = dev;

            A = new {ta}[size_A];
            x = new {tx}[size_x];
            y = new {ty}[size_y];
            yref = new {ty}[size_y];

            if (queue != nullptr) {{
                delete queue;
            }}
            queue = new blas::Queue(device_id);

            dA = blas::device_malloc<{ta}>(size_A, *queue);
            dx = blas::device_malloc<{tx}>(size_x, *queue);
            dy = blas::device_malloc<{ty}>(size_y, *queue);
        }}
    }}

    // Free all memory
    void free() {{
        if (A != nullptr) {{
            delete[] A;
            delete[] x;
            delete[] y;
            delete[] yref;
            A = nullptr;
            x = nullptr;
            y = nullptr;
            yref = nullptr;
        }}

        if (dA != nullptr && queue != nullptr) {{
            blas::device_free(dA, *queue);
            blas::device_free(dx, *queue);
            blas::device_free(dy, *queue);
            dA = nullptr;
            dx = nullptr;
            dy = nullptr;
        }}

        if (queue != nullptr) {{
            delete queue;
            queue = nullptr;
        }}

        size_A = 0;
        size_x = 0;
        size_y = 0;
        device_id = -1;
    }}

    ~{name}() {{
        free();
    }}
}};

// Static memory pools for each data type
static {name}<float, float, float> pool_s;
static {name}<double, double, double> pool_d;
static {name}<std::complex<float>, std::complex<float>, std::complex<float>> pool_c;
static {name}<std::complex<double>, std::complex<double>, std::complex<double>> pool_z;

"""

def get_memory_pool_struct_mat_vec(name, ta="TA", tx="TX"):
    """Generate memory pool struct for matrix + vector (A/Aref, x)"""
    return f"""// -----------------------------------------------------------------------------
// Memory pool structure for device memory management
template <typename {ta}, typename {tx}>
struct {name} {{
    // Host memory
    {ta}* A = nullptr;
    {ta}* Aref = nullptr;
    {tx}* x = nullptr;

    // Device memory
    {ta}* dA = nullptr;
    {tx}* dx = nullptr;

    // Memory sizes
    size_t size_A = 0;
    size_t size_x = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sA, size_t sx, int64_t dev) {{
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {{
            free();
            need_realloc = true;
        }}

        // Check if we need to allocate larger memory
        if (A == nullptr || size_A < sA || size_x < sx) {{
            if (A != nullptr) {{
                free();
            }}
            need_realloc = true;
        }}

        // Allocate new memory if needed
        if (need_realloc || A == nullptr) {{
            // Use larger size to avoid frequent reallocation
            size_A = (sA > size_A) ? sA : size_A;
            size_x = (sx > size_x) ? sx : size_x;
            device_id = dev;

            A = new {ta}[size_A];
            Aref = new {ta}[size_A];
            x = new {tx}[size_x];

            if (queue != nullptr) {{
                delete queue;
            }}
            queue = new blas::Queue(device_id);

            dA = blas::device_malloc<{ta}>(size_A, *queue);
            dx = blas::device_malloc<{tx}>(size_x, *queue);
        }}
    }}

    // Free all memory
    void free() {{
        if (A != nullptr) {{
            delete[] A;
            delete[] Aref;
            delete[] x;
            A = nullptr;
            Aref = nullptr;
            x = nullptr;
        }}

        if (dA != nullptr && queue != nullptr) {{
            blas::device_free(dA, *queue);
            blas::device_free(dx, *queue);
            dA = nullptr;
            dx = nullptr;
        }}

        if (queue != nullptr) {{
            delete queue;
            queue = nullptr;
        }}

        size_A = 0;
        size_x = 0;
        device_id = -1;
    }}

    ~{name}() {{
        free();
    }}
}};

// Static memory pools for each data type
static {name}<float, float> pool_s;
static {name}<double, double> pool_d;
static {name}<std::complex<float>, std::complex<float>> pool_c;
static {name}<std::complex<double>, std::complex<double>> pool_z;

"""

print("Script loaded. Ready to process files.")
