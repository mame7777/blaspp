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
template <typename scalar_t>
struct ScalDeviceMemoryPool {
    // Host memory
    scalar_t* x = nullptr;
    scalar_t* xref = nullptr;

    // Device memory
    scalar_t* dx = nullptr;

    // Memory sizes
    size_t size_x = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sx, int64_t dev) {
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {
            free();
            need_realloc = true;
        }

        // Check if we need to allocate larger memory
        if (x == nullptr || size_x < sx) {
            if (x != nullptr) {
                free();
            }
            need_realloc = true;
        }

        // Allocate new memory if needed
        if (need_realloc || x == nullptr) {
            // Use larger size to avoid frequent reallocation
            size_x = (sx > size_x) ? sx : size_x;
            device_id = dev;

            x = new scalar_t[size_x];
            xref = new scalar_t[size_x];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dx = blas::device_malloc<scalar_t>(size_x, *queue);
        }
    }

    // Free all memory
    void free() {
        if (x != nullptr) {
            delete[] x;
            delete[] xref;
            x = nullptr;
            xref = nullptr;
        }

        if (dx != nullptr && queue != nullptr) {
            blas::device_free(dx, *queue);
            dx = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_x = 0;
        device_id = -1;
    }

    ~ScalDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (x != nullptr) {
            delete[] x;
            delete[] xref;
            x = nullptr;
            xref = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static ScalDeviceMemoryPool<float> pool_s;
static ScalDeviceMemoryPool<double> pool_d;
static ScalDeviceMemoryPool<std::complex<float>> pool_c;
static ScalDeviceMemoryPool<std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_scal_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::max;
    using std::abs, std::real, std::imag;
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    scalar_t alpha  = params.alpha.get<scalar_t>();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
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
    params.ref_time.width( 13 );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // Get appropriate memory pool
    ScalDeviceMemoryPool<scalar_t>* pool = nullptr;
    if (std::is_same<scalar_t, float>::value) {
        pool = reinterpret_cast<ScalDeviceMemoryPool<scalar_t>*>(&pool_s);
    } else if (std::is_same<scalar_t, double>::value) {
        pool = reinterpret_cast<ScalDeviceMemoryPool<scalar_t>*>(&pool_d);
    } else if (std::is_same<scalar_t, std::complex<float>>::value) {
        pool = reinterpret_cast<ScalDeviceMemoryPool<scalar_t>*>(&pool_c);
    } else if (std::is_same<scalar_t, std::complex<double>>::value) {
        pool = reinterpret_cast<ScalDeviceMemoryPool<scalar_t>*>(&pool_z);
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );

    // Allocate or reuse memory from pool
    pool->allocate(size_x, device);

    scalar_t* x = pool->x;
    scalar_t* xref = pool->xref;

    // device specifics
    blas::Queue& queue = *(pool->queue);
    scalar_t* dx = pool->dx;

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    queue.sync();

    // test error exits
    assert_throw( blas::scal( -1, alpha, dx, incx, queue ), blas::Error );
    assert_throw( blas::scal(  n, alpha, dx,    0, queue ), blas::Error );
    assert_throw( blas::scal(  n, alpha, dx,   -1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::scal( n, alpha, dx, incx, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::scal( n );
    double gbyte = blas::Gbyte< scalar_t >::scal( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_copy_vector( n, dx, abs( incx ), x, abs( incx ), queue );
    queue.sync();

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_scal( n, alpha, xref, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
        }

        // maximum component-wise forward error:
        // | fl(xi) - xi | / | xi |
        real_t error = 0;
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            error = max( error, abs( (xref[ix] - x[ix]) / xref[ix] ));
            ix += incx;
        }
        params.error() = error;

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex_v<scalar_t>) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.okay() = (error < u);
    }

    // Memory is managed by the pool and will be reused or freed automatically
    // No explicit deletion needed here
}

// -----------------------------------------------------------------------------
void test_scal_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_scal_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_scal_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_scal_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_scal_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
