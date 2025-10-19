// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
// Memory pool structure for device memory management
template <typename TX, typename TY>
struct SwapDeviceMemoryPool {
    // Host memory
    TX* x = nullptr;
    TX* xref = nullptr;
    TY* y = nullptr;
    TY* yref = nullptr;

    // Device memory
    TX* dx = nullptr;
    TY* dy = nullptr;

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

            x = new TX[size_x];
            xref = new TX[size_x];
            y = new TY[size_y];
            yref = new TY[size_y];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dx = blas::device_malloc<TX>(size_x, *queue);
            dy = blas::device_malloc<TY>(size_y, *queue);
        }
    }

    // Free all memory
    void free() {
        if (x != nullptr) {
            delete[] x;
            delete[] xref;
            delete[] y;
            delete[] yref;
            x = nullptr;
            xref = nullptr;
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

    ~SwapDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (x != nullptr) {
            delete[] x;
            delete[] xref;
            delete[] y;
            delete[] yref;
            x = nullptr;
            xref = nullptr;
            y = nullptr;
            yref = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static SwapDeviceMemoryPool<float, float> pool_s;
static SwapDeviceMemoryPool<double, double> pool_d;
static SwapDeviceMemoryPool<std::complex<float>, std::complex<float>> pool_c;
static SwapDeviceMemoryPool<std::complex<double>, std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
template <typename TX, typename TY>
void test_swap_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs;
    using blas::max;
    using scalar_t = blas::scalar_type< TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t device  = params.device();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // Get appropriate memory pool
    SwapDeviceMemoryPool<TX, TY>* pool = nullptr;
    if (std::is_same<TX, float>::value && std::is_same<TY, float>::value) {
        pool = reinterpret_cast<SwapDeviceMemoryPool<TX, TY>*>(&pool_s);
    } else if (std::is_same<TX, double>::value && std::is_same<TY, double>::value) {
        pool = reinterpret_cast<SwapDeviceMemoryPool<TX, TY>*>(&pool_d);
    } else if (std::is_same<TX, std::complex<float>>::value && std::is_same<TY, std::complex<float>>::value) {
        pool = reinterpret_cast<SwapDeviceMemoryPool<TX, TY>*>(&pool_c);
    } else if (std::is_same<TX, std::complex<double>>::value && std::is_same<TY, std::complex<double>>::value) {
        pool = reinterpret_cast<SwapDeviceMemoryPool<TX, TY>*>(&pool_z);
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    size_t size_y = max( (n - 1) * abs( incy ) + 1, 0 );

    // Allocate or reuse memory from pool
    pool->allocate(size_x, size_y, device);

    TX* x = pool->x;
    TX* xref = pool->xref;
    TY* y = pool->y;
    TY* yref = pool->yref;

    // device specifics
    blas::Queue& queue = *(pool->queue);
    TX* dx = pool->dx;
    TY* dy = pool->dy;

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, x, incx, xref, incx );
    cblas_copy( n, y, incy, yref, incy );

    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    blas::device_copy_vector( n, y, abs( incy ), dy, abs( incy ), queue );
    queue.sync();

    // test error exits
    assert_throw( blas::swap( -1, dx, incx, dy, incy, queue ), blas::Error );
    assert_throw( blas::swap(  n, dx,    0, dy, incy, queue ), blas::Error );
    assert_throw( blas::swap(  n, dx, incx, dy,    0, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ),
                llong( n ), llong( incy ), llong( size_y ) );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::swap( n, dx, incx, dy, incy, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::swap( n );
    double gbyte = blas::Gbyte< scalar_t >::swap( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
    params.gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

    blas::device_copy_vector( n, dx, abs( incx ), x, abs( incx ), queue );
    blas::device_copy_vector( n, dy, abs( incy ), y, abs( incy ), queue );
    queue.sync();

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_swap( n, xref, incx, yref, incy );
        time = get_wtime() - time;
        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
        params.ref_gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

        // error = ||xref - x|| + ||yref - y||
        cblas_axpy( n, -1.0, x, incx, xref, incx );
        cblas_axpy( n, -1.0, y, incy, yref, incy );
        real_t error = cblas_nrm2( n, xref, std::abs(incx) )
                     + cblas_nrm2( n, yref, std::abs(incy) );
        params.error() = error;

        // swap must be exact!
        params.okay() = (error == 0);
    }

    // Memory is managed by the pool and will be reused or freed automatically
    // No explicit deletion needed here
}

// -----------------------------------------------------------------------------
void test_swap_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_swap_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_swap_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_swap_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_swap_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
