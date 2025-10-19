// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
// Memory pool for rotmg_device test
template <typename T>
struct RotmgDeviceMemoryPool {
    // Host memory
    T* d1 = nullptr;
    T* d1_ref = nullptr;
    T* d2 = nullptr;
    T* d2_ref = nullptr;
    T* x1 = nullptr;
    T* x1_ref = nullptr;
    T* y1 = nullptr;
    T* y1_ref = nullptr;
    T* ps = nullptr;
    T* ps_ref = nullptr;

    // Device memory
    T* d_d1 = nullptr;
    T* d_d2 = nullptr;
    T* d_x1 = nullptr;
    T* d_y1 = nullptr;
    T* d_ps = nullptr;

    // Memory sizes
    size_t size_vec = 0;  // for d1, d2, x1, y1
    size_t size_ps = 0;   // for ps (5*n)

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    void allocate(size_t n_vec, size_t n_ps, int64_t dev) {
        // Check if we can reuse existing allocation
        if (d1 != nullptr && size_vec >= n_vec && size_ps >= n_ps && device_id == dev) {
            return;  // Reuse existing allocation
        }

        // Free old allocation if exists
        free();

        // Store new sizes
        size_vec = n_vec;
        size_ps = n_ps;
        device_id = dev;

        // Create queue if needed
        if (queue == nullptr) {
            queue = new blas::Queue(device_id);
        }

        // Allocate host memory
        d1 = new T[size_vec];
        d1_ref = new T[size_vec];
        d2 = new T[size_vec];
        d2_ref = new T[size_vec];
        x1 = new T[size_vec];
        x1_ref = new T[size_vec];
        y1 = new T[size_vec];
        y1_ref = new T[size_vec];
        ps = new T[size_ps];
        ps_ref = new T[size_ps];

        // Allocate device memory
        d_d1 = blas::device_malloc<T>(size_vec, *queue);
        d_d2 = blas::device_malloc<T>(size_vec, *queue);
        d_x1 = blas::device_malloc<T>(size_vec, *queue);
        d_y1 = blas::device_malloc<T>(size_vec, *queue);
        d_ps = blas::device_malloc<T>(size_ps, *queue);
    }

    void free() {
        // Free host memory
        delete[] d1; d1 = nullptr;
        delete[] d1_ref; d1_ref = nullptr;
        delete[] d2; d2 = nullptr;
        delete[] d2_ref; d2_ref = nullptr;
        delete[] x1; x1 = nullptr;
        delete[] x1_ref; x1_ref = nullptr;
        delete[] y1; y1 = nullptr;
        delete[] y1_ref; y1_ref = nullptr;
        delete[] ps; ps = nullptr;
        delete[] ps_ref; ps_ref = nullptr;

        // Free device memory
        if (queue != nullptr) {
            if (d_d1 != nullptr) blas::device_free(d_d1, *queue);
            if (d_d2 != nullptr) blas::device_free(d_d2, *queue);
            if (d_x1 != nullptr) blas::device_free(d_x1, *queue);
            if (d_y1 != nullptr) blas::device_free(d_y1, *queue);
            if (d_ps != nullptr) blas::device_free(d_ps, *queue);

            d_d1 = nullptr;
            d_d2 = nullptr;
            d_x1 = nullptr;
            d_y1 = nullptr;
            d_ps = nullptr;
        }

        size_vec = 0;
        size_ps = 0;
    }

    ~RotmgDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        delete[] d1; d1 = nullptr;
        delete[] d1_ref; d1_ref = nullptr;
        delete[] d2; d2 = nullptr;
        delete[] d2_ref; d2_ref = nullptr;
        delete[] x1; x1 = nullptr;
        delete[] x1_ref; x1_ref = nullptr;
        delete[] y1; y1 = nullptr;
        delete[] y1_ref; y1_ref = nullptr;
        delete[] ps; ps = nullptr;
        delete[] ps_ref; ps_ref = nullptr;
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools (only float and double, rotmg doesn't support complex)
static RotmgDeviceMemoryPool<float> pool_s;
static RotmgDeviceMemoryPool<double> pool_d;

// -----------------------------------------------------------------------------
template <typename T>
void test_rotmg_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs;
    using std::real;
    using std::imag;
    using real_t   = blas::real_type< T >;

    // Constants
    const real_t epsilon = std::numeric_limits< real_t >::epsilon();

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t device  = params.device();
    double tol      = params.tol() * epsilon;

    // mark non-standard output values
    params.ref_time();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // Get the appropriate memory pool
    RotmgDeviceMemoryPool<T>& pool =
        std::is_same<T, float>::value ?
            reinterpret_cast<RotmgDeviceMemoryPool<T>&>(pool_s) :
            reinterpret_cast<RotmgDeviceMemoryPool<T>&>(pool_d);

    // Allocate from pool
    pool.allocate(n, 5*n, device);

    // Get pointers from pool
    T* d1 = pool.d1;
    T* d1_ref = pool.d1_ref;
    T* d2 = pool.d2;
    T* d2_ref = pool.d2_ref;
    T* x1 = pool.x1;
    T* x1_ref = pool.x1_ref;
    T* y1 = pool.y1;
    T* y1_ref = pool.y1_ref;
    T* ps = pool.ps;
    T* ps_ref = pool.ps_ref;

    T* d_d1 = pool.d_d1;
    T* d_d2 = pool.d_d2;
    T* d_x1 = pool.d_x1;
    T* d_y1 = pool.d_y1;
    T* d_ps = pool.d_ps;

    blas::Queue& queue = *pool.queue;

    int64_t idist = 3;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, n, d1 );
    lapack_larnv( idist, iseed, n, d2 );
    lapack_larnv( idist, iseed, n, x1 );
    lapack_larnv( idist, iseed, n, y1 );
    lapack_larnv( idist, iseed, 5*n, ps );

    device_memcpy( d_d1, d1, n, queue );
    device_memcpy( d_d2, d2, n, queue );
    device_memcpy( d_x1, x1, n, queue );
    device_memcpy( d_y1, y1, n, queue );
    device_memcpy( d_ps, ps, 5*n, queue );

    #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode(queue.handle(), CUBLAS_POINTER_MODE_DEVICE);
    #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
    #endif

    cblas_copy( n, d1, 1, d1_ref, 1 );
    cblas_copy( n, d2, 1, d2_ref, 1 );
    cblas_copy( n, x1, 1, x1_ref, 1 );
    cblas_copy( n, y1, 1, y1_ref, 1 );
    cblas_copy( 5*n, ps, 1, ps_ref, 1 );

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    for (int64_t i = 0; i < n; ++i) {
        blas::rotmg( d_d1 + i, d_d2 + i, d_x1 + i, d_y1 + i, d_ps + 5*i, queue );
    }
    queue.sync();
    time = get_wtime() - time;
    params.time() = time * 1000;  // msec

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (int64_t i = 0; i < n; ++i) {
            cblas_rotmg( &d1_ref[i], &d2_ref[i], &x1_ref[i], y1_ref[i], &ps_ref[5*i] );
        }
        time = get_wtime() - time;
        params.ref_time() = time * 1000;  // msec

        device_memcpy( d1, d_d1, n, queue );
        device_memcpy( d2, d_d2, n, queue );
        device_memcpy( x1, d_x1, n, queue );
        device_memcpy( ps, d_ps, 5*n, queue );
        queue.sync();

        // get max error of all outputs
        cblas_axpy(   n, -1.0, &d1[0], 1, &d1_ref[0], 1 );
        cblas_axpy(   n, -1.0, &d2[0], 1, &d2_ref[0], 1 );
        cblas_axpy(   n, -1.0, &x1[0], 1, &x1_ref[0], 1 );
        cblas_axpy( 5*n, -1.0, &ps[0], 1, &ps_ref[0], 1 );

        int64_t id1 = cblas_iamax(   n, &d1_ref[0], 1 );
        int64_t id2 = cblas_iamax(   n, &d2_ref[0], 1 );
        int64_t ix1 = cblas_iamax(   n, &x1_ref[0], 1 );
        int64_t ips = cblas_iamax( 5*n, &ps_ref[0], 1 );

        real_t error = blas::max(
            abs( d1_ref[ id1 ] ),
            abs( d2_ref[ id2 ] ),
            abs( x1_ref[ ix1 ] ),
            abs( ps_ref[ ips ] )
        );

        // error is normally 0, but allow for some rounding just in case.
        params.error() = error;
        params.okay() = (error < tol);
    }

    // Memory is retained in pool for reuse
}

// -----------------------------------------------------------------------------
void test_rotmg_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_rotmg_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_rotmg_device_work< double >( params, run );
            break;

        // modified Givens not available for complex

        default:
            throw std::exception();
            break;
    }
}
