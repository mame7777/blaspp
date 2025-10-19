// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
// Memory pool structure for device memory management
template <typename TX>
struct RotmDeviceMemoryPool {
    // Host memory
    TX* x = nullptr;
    TX* xref = nullptr;
    TX* y = nullptr;
    TX* yref = nullptr;

    // Device memory
    TX* dx = nullptr;
    TX* dy = nullptr;
    TX* dp = nullptr;

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
            y = new TX[size_y];
            yref = new TX[size_y];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dx = blas::device_malloc<TX>(size_x, *queue);
            dy = blas::device_malloc<TX>(size_y, *queue);
            dp = blas::device_malloc<TX>(5, *queue);
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
            blas::device_free(dp, *queue);
            dx = nullptr;
            dy = nullptr;
            dp = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_x = 0;
        size_y = 0;
        device_id = -1;
    }

    ~RotmDeviceMemoryPool() {
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
static RotmDeviceMemoryPool<float> pool_s;
static RotmDeviceMemoryPool<double> pool_d;

// -----------------------------------------------------------------------------
template <typename TX>
void test_rotm_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs;
    using blas::max;
    using real_t   = blas::real_type< TX >;

    // get & mark input values
    int64_t n          = params.dim.n();
    int64_t incx       = params.incx();
    int64_t incy       = params.incy();
    int64_t device     = params.device();
    int64_t verbose    = params.verbose();

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
    RotmDeviceMemoryPool<TX>* pool = nullptr;
    if (std::is_same<TX, float>::value) {
        pool = reinterpret_cast<RotmDeviceMemoryPool<TX>*>(&pool_s);
    } else if (std::is_same<TX, double>::value) {
        pool = reinterpret_cast<RotmDeviceMemoryPool<TX>*>(&pool_d);
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    size_t size_y = max( (n - 1) * abs( incy ) + 1, 0 );

    // Allocate or reuse memory from pool
    pool->allocate(size_x, size_y, device);

    TX* x = pool->x;
    TX* xref = pool->xref;
    TX* y = pool->y;
    TX* yref = pool->yref;

    // device specifics
    blas::Queue& queue = *(pool->queue);
    TX* dx = pool->dx;
    TX* dy = pool->dy;
    TX* dp = pool->dp;

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, x, incx, xref, incx );
    cblas_copy( n, y, incy, yref, incy );

    // compute random rotation
    TX d[4];
    TX p[5];
    lapack_larnv( idist, iseed, 4, d );
    blas::rotmg( &d[0], &d[1], &d[2], d[3], p );

    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );
    real_t Anorm = sqrt( Xnorm*Xnorm + Ynorm*Ynorm ); // || [x y] ||_F

    blas::device_copy_vector( n, x, std::abs(incx), dx, std::abs(incx), queue );
    blas::device_copy_vector( n, y, std::abs(incy), dy, std::abs(incy), queue );
    blas::device_copy_vector( 5, p, 1, dp, 1, queue );
    queue.sync();

    #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode(queue.handle(), CUBLAS_POINTER_MODE_DEVICE);
    #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
    #endif

    // test error exits
    assert_throw( blas::rotm( -1, x, incx, y, incy, p ), blas::Error );
    assert_throw( blas::rotm(  n, x,    0, y, incy, p ), blas::Error );
    assert_throw( blas::rotm(  n, x, incx, y,    0, p ), blas::Error );

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
    blas::rotm( n, dx, incx, dy, incy, dp, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< TX >::dot( n );
    double gbyte = blas::Gbyte< TX >::dot( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_copy_vector(n, dx, std::abs(incx), x, std::abs(incx), queue);
    blas::device_copy_vector(n, dy, std::abs(incy), y, std::abs(incy), queue);
    queue.sync();

    if (verbose >= 1) {
        printf( "x2   = " ); print_vector( n, x, incx );
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_rotm( n, xref, incx, yref, incy, p );  // todo
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 1) {
            printf( "xref = " ); print_vector( n, xref, incx );
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // check error compared to reference
        // C = [x y] * R + C0, for n x 2 matrix C and 2 x 2 rotation R
        // alpha=1, beta=0, C0norm=0
        TX* C    = new TX[ 2*n ];
        TX* Cref = new TX[ 2*n ];
        blas::copy( n, x,    incx, &C[0],    1 );
        blas::copy( n, y,    incy, &C[n],    1 );
        blas::copy( n, xref, incx, &Cref[0], 1 );
        blas::copy( n, yref, incy, &Cref[n], 1 );
        real_t Rnorm = sqrt(2);  // ||R||_F  // todo
        real_t error;
        bool okay;
        check_gemm( n, 2, 2, TX(1), TX(0), Anorm, Rnorm, real_t(0),
                    Cref, n, C, n, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;

        delete[] C;
        delete[] Cref;
    }

    // Memory is managed by the pool and will be reused or freed automatically
    // No explicit deletion needed here
}

// -----------------------------------------------------------------------------
void test_rotm_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_rotm_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_rotm_device_work< double >( params, run );
            break;

        // modified Givens not available for complex

        default:
            throw std::exception();
            break;
    }
}
