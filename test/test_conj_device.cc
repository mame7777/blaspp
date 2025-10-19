// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "blas/device.hh"

// -----------------------------------------------------------------------------
// Memory pool structure for device memory management
template <typename scalar_t>
struct ConjDeviceMemoryPool {
    // Host memory
    scalar_t* x = nullptr;
    scalar_t* y = nullptr;
    scalar_t* yref = nullptr;

    // Device memory
    scalar_t* dx = nullptr;
    scalar_t* dy = nullptr;

    // Memory sizes
    size_t size_x = 0;
    size_t size_y = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sx, size_t sy, int64_t dev) {
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {
            free();
            need_realloc = true;
        }

        // Check if we need to allocate larger memory
        if (x == nullptr || size_x < sx || size_y < sy) {
            if (x != nullptr) {
                free();
            }
            need_realloc = true;
        }

        // Allocate new memory if needed
        if (need_realloc || x == nullptr) {
            // Use larger size to avoid frequent reallocation
            size_x = (sx > size_x) ? sx : size_x;
            size_y = (sy > size_y) ? sy : size_y;
            device_id = dev;

            x = new scalar_t[size_x];
            y = new scalar_t[size_y];
            yref = new scalar_t[size_y];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dx = blas::device_malloc<scalar_t>(size_x, *queue);
            dy = blas::device_malloc<scalar_t>(size_y, *queue);
        }
    }

    // Free all memory
    void free() {
        if (x != nullptr) {
            delete[] x;
            delete[] y;
            delete[] yref;
            x = nullptr;
            y = nullptr;
            yref = nullptr;
        }

        if (dx != nullptr && queue != nullptr) {
            blas::device_free(dx, *queue);
            blas::device_free(dy, *queue);
            dx = nullptr;
            dy = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_x = 0;
        size_y = 0;
        device_id = -1;
    }

    ~ConjDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (x != nullptr) {
            delete[] x;
            delete[] y;
            delete[] yref;
            x = nullptr;
            y = nullptr;
            yref = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static ConjDeviceMemoryPool<float> pool_s;
static ConjDeviceMemoryPool<double> pool_d;
static ConjDeviceMemoryPool<std::complex<float>> pool_c;
static ConjDeviceMemoryPool<std::complex<double>> pool_z;

//------------------------------------------------------------------------------
template <typename scalar_t>
void cpu_conj(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t*       y, int64_t incy )
{
    using blas::conj;

    int64_t ix = (incx > 0 ? 0 : (1 - n) * incx);
    int64_t iy = (incy > 0 ? 0 : (1 - n) * incy);

    for (int i = 0; i < n; ++i) {
        y[i * incy + iy] = conj( x[i * incx + ix] );
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_conj_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs, std::real, std::imag;
    using blas::max;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t device  = params.device();
    int64_t verbose = params.verbose();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // Get appropriate memory pool
    ConjDeviceMemoryPool<scalar_t>* pool = nullptr;
    if (std::is_same<scalar_t, float>::value) {
        pool = reinterpret_cast<ConjDeviceMemoryPool<scalar_t>*>(&pool_s);
    } else if (std::is_same<scalar_t, double>::value) {
        pool = reinterpret_cast<ConjDeviceMemoryPool<scalar_t>*>(&pool_d);
    } else if (std::is_same<scalar_t, std::complex<float>>::value) {
        pool = reinterpret_cast<ConjDeviceMemoryPool<scalar_t>*>(&pool_c);
    } else if (std::is_same<scalar_t, std::complex<double>>::value) {
        pool = reinterpret_cast<ConjDeviceMemoryPool<scalar_t>*>(&pool_z);
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    size_t size_y = max( (n - 1) * abs( incy ) + 1, 0 );

    // Allocate or reuse memory from pool
    pool->allocate(size_x, size_y, device);

    scalar_t* x = pool->x;
    scalar_t* y = pool->y;
    scalar_t* yref = pool->yref;

    // device specifics
    blas::Queue& queue = *(pool->queue);
    scalar_t* dx = pool->dx;
    scalar_t* dy = pool->dy;

    queue.sync();

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );

    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    blas::device_copy_vector( n, y, abs( incy ), dy, abs( incy ), queue );
    queue.sync();

    // test error exits
    assert_throw( blas::conj( -1, x, incx, y, incy, queue ), blas::Error );
    assert_throw( blas::conj(  n, x,    0, y, incy, queue ), blas::Error );
    assert_throw( blas::conj(  n, x, incx, y,    0, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ),
                llong( n ), llong( incy ), llong( size_y ) );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    blas::conj( n, dx, incx, dy, incy, queue );
    queue.sync();

    blas::device_copy_vector( n, dx, abs( incx ), x, abs( incx ), queue );
    blas::device_copy_vector( n, dy, abs( incy ), y, abs( incy ), queue );
    queue.sync();

    if (verbose >= 2) {
        printf( "y    = " ); print_vector( n, y, incy );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        cpu_conj( n, x, incx, yref, incy );

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // error = ||yref - y||
        cblas_axpy( n, -1.0, y, incy, yref, incy );
        real_t error = cblas_nrm2( n, yref, abs( incy ) );
        params.error() = error;

        // result is expected to be identical since only sign changes
        params.okay() = (error == 0);
    }

    // Memory is managed by the pool and will be reused or freed automatically
    // No explicit deletion needed here
}

//------------------------------------------------------------------------------
void test_conj_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_conj_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_conj_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_conj_device_work< std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_conj_device_work< std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
