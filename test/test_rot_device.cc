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
template <typename TX, typename TY>
struct RotDeviceMemoryPool {
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

    ~RotDeviceMemoryPool() {
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
static RotDeviceMemoryPool<float, float> pool_s;
static RotDeviceMemoryPool<double, double> pool_d;
static RotDeviceMemoryPool<std::complex<float>, std::complex<float>> pool_c;
static RotDeviceMemoryPool<std::complex<double>, std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
// TX is data [x, y]
// TS is for sine, which can be real (zdrot) or complex (zrot)
// cosine is always real
template <typename TX, typename TS>
void test_rot_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs, std::real, std::imag;
    using blas::conj, blas::max;
    using real_t = blas::real_type< TX >;

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

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // Get appropriate memory pool
    RotDeviceMemoryPool<TX, TX>* pool = nullptr;
    if (std::is_same<TX, float>::value) {
        pool = reinterpret_cast<RotDeviceMemoryPool<TX, TX>*>(&pool_s);
    } else if (std::is_same<TX, double>::value) {
        pool = reinterpret_cast<RotDeviceMemoryPool<TX, TX>*>(&pool_d);
    } else if (std::is_same<TX, std::complex<float>>::value) {
        pool = reinterpret_cast<RotDeviceMemoryPool<TX, TX>*>(&pool_c);
    } else if (std::is_same<TX, std::complex<double>>::value) {
        pool = reinterpret_cast<RotDeviceMemoryPool<TX, TX>*>(&pool_z);
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

    // When TX is complex, TS can be real or complex.
    TS s, data[ 2 ];
    real_t c;  // real

    // device specifics
    blas::Queue& queue = *(pool->queue);
    TX* dx = pool->dx;
    TX* dy = pool->dy;

    int64_t idist = 2;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, x, incx, xref, incx );
    cblas_copy( n, y, incy, yref, incy );

    // Compute [c, s] to eliminate data[1].
    lapack_larnv( idist, iseed, 2, data );
    blas::rotg( &data[0], &data[0], &c, &s );

    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );
    real_t Anorm = sqrt( Xnorm*Xnorm + Ynorm*Ynorm ); // || [x y] ||_F

    blas::device_copy_vector( n, x, std::abs(incx), dx, std::abs(incx), queue );
    blas::device_copy_vector( n, y, std::abs(incy), dy, std::abs(incy), queue );
    queue.sync();

    // test error exits
    assert_throw( blas::rot( -1, x, incx, y, incy, c, s ), blas::Error );
    assert_throw( blas::rot(  n, x,    0, y, incy, c, s ), blas::Error );
    assert_throw( blas::rot(  n, x, incx, y,    0, c, s ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "s = %.4f + %.4fi, c = %.4f, s^2 + c^2 = %.4f\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                real( s ), imag( s ), c, real( s*conj(s) ) + c*c,
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
    blas::rot( n, dx, incx, dy, incy, c, s, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< TX >::dot( n );
    double gbyte = blas::Gbyte< TX >::dot( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
    params.gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

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
        cblas_rot( n, xref, incx, yref, incy, c, s );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
        params.ref_gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

        if (verbose >= 1) {
            printf( "xref = " ); print_vector( n, xref, incx );
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
        params.ref_gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

        // check error compared to reference
        // C = [x y] * R for n x 2 matrix C and 2 x 2 rotation R
        // alpha=1, beta=0, C0norm=0
        TX* C    = new TX[ 2*n ];
        TX* Cref = new TX[ 2*n ];
        blas::copy( n, x,    incx, &C[0],    1 );
        blas::copy( n, y,    incy, &C[n],    1 );
        blas::copy( n, xref, incx, &Cref[0], 1 );
        blas::copy( n, yref, incy, &Cref[n], 1 );
        real_t Rnorm = sqrt(2);  // ||R||_F
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
void test_rot_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_rot_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_rot_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_rot_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_rot_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
        // todo: real sine
        // todo: complex sine

        default:
            throw std::exception();
            break;
    }
}
