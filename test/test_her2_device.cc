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
template <typename TA, typename TX, typename TY>
struct Her2DeviceMemoryPool {
    // Host memory
    TA* A = nullptr;
    TA* Aref = nullptr;
    TX* x = nullptr;
    TY* y = nullptr;

    // Device memory
    TA* dA = nullptr;
    TX* dx = nullptr;
    TY* dy = nullptr;

    // Memory sizes
    size_t size_A = 0;
    size_t size_x = 0;
    size_t size_y = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sA, size_t sx, size_t sy, int64_t dev) {
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {
            free();
            need_realloc = true;
        }

        // Check if we need to allocate larger memory
        if (A == nullptr || size_A < sA || size_x < sx || size_y < sy) {
            if (A != nullptr) {
                free();
            }
            need_realloc = true;
        }

        // Allocate new memory if needed
        if (need_realloc || A == nullptr) {
            // Use larger size to avoid frequent reallocation
            size_A = (sA > size_A) ? sA : size_A;
            size_x = (sx > size_x) ? sx : size_x;
            size_y = (sy > size_y) ? sy : size_y;
            device_id = dev;

            A = new TA[size_A];
            Aref = new TA[size_A];
            x = new TX[size_x];
            y = new TY[size_y];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dA = blas::device_malloc<TA>(size_A, *queue);
            dx = blas::device_malloc<TX>(size_x, *queue);
            dy = blas::device_malloc<TY>(size_y, *queue);
        }
    }

    // Free all memory
    void free() {
        if (A != nullptr) {
            delete[] A;
            delete[] Aref;
            delete[] x;
            delete[] y;
            A = nullptr;
            Aref = nullptr;
            x = nullptr;
            y = nullptr;
        }

        if (dA != nullptr && queue != nullptr) {
            blas::device_free(dA, *queue);
            blas::device_free(dx, *queue);
            blas::device_free(dy, *queue);
            dA = nullptr;
            dx = nullptr;
            dy = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_A = 0;
        size_x = 0;
        size_y = 0;
        device_id = -1;
    }

    ~Her2DeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (A != nullptr) {
            delete[] A;
            delete[] Aref;
            delete[] x;
            delete[] y;
            A = nullptr;
            Aref = nullptr;
            x = nullptr;
            y = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static Her2DeviceMemoryPool<float, float, float> pool_s;
static Her2DeviceMemoryPool<double, double, double> pool_d;
static Her2DeviceMemoryPool<std::complex<float>, std::complex<float>, std::complex<float>> pool_c;
static Her2DeviceMemoryPool<std::complex<double>, std::complex<double>, std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
template <typename TA, typename TX, typename TY>
void test_her2_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs, std::real, std::imag;
    using blas::Uplo, blas::Layout, blas::max;
    using scalar_t = blas::scalar_type< TA, TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha.get<scalar_t>();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t device  = params.device();
    int64_t align   = params.align();
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
    Her2DeviceMemoryPool<TA, TX, TY>* pool = nullptr;
    if (std::is_same<TA, float>::value && std::is_same<TX, float>::value && std::is_same<TY, float>::value) {
        pool = reinterpret_cast<Her2DeviceMemoryPool<TA, TX, TY>*>(&pool_s);
    } else if (std::is_same<TA, double>::value && std::is_same<TX, double>::value && std::is_same<TY, double>::value) {
        pool = reinterpret_cast<Her2DeviceMemoryPool<TA, TX, TY>*>(&pool_d);
    } else if (std::is_same<TA, std::complex<float>>::value && std::is_same<TX, std::complex<float>>::value && std::is_same<TY, std::complex<float>>::value) {
        pool = reinterpret_cast<Her2DeviceMemoryPool<TA, TX, TY>*>(&pool_c);
    } else if (std::is_same<TA, std::complex<double>>::value && std::is_same<TX, std::complex<double>>::value && std::is_same<TY, std::complex<double>>::value) {
        pool = reinterpret_cast<Her2DeviceMemoryPool<TA, TX, TY>*>(&pool_z);
    }

    // setup
    int64_t lda = max( roundup( n, align ), 1 );
    size_t size_A = size_t(lda)*n;
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    size_t size_y = max( (n - 1) * abs( incy ) + 1, 0 );

    // Allocate or reuse memory from pool
    pool->allocate(size_A, size_x, size_y, device);

    TA* A = pool->A;
    TA* Aref = pool->Aref;
    TX* x = pool->x;
    TY* y = pool->y;

    // device specifics
    blas::Queue& queue = *(pool->queue);
    TA* dA = pool->dA;
    TX* dx = pool->dx;
    TY* dy = pool->dy;

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    lapack_lacpy( "g", n, n, A, lda, Aref, lda );

    blas::device_copy_matrix( n, n, A, lda, dA, lda, queue );
    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    blas::device_copy_vector( n, y, abs( incy ), dy, abs( incy ), queue );
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lanhe( "f", to_c_string( uplo ), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );

    // test error exits
    assert_throw( blas::her2( Layout(0), uplo,     n, alpha, dx, incx, dy, incy, dA, lda, queue ), blas::Error );
    assert_throw( blas::her2( layout,    Uplo(0),  n, alpha, dx, incx, dy, incy, dA, lda, queue ), blas::Error );
    assert_throw( blas::her2( layout,    uplo,    -1, alpha, dx, incx, dy, incy, dA, lda, queue ), blas::Error );
    assert_throw( blas::her2( layout,    uplo,     n, alpha, dx,    0, dy, incy, dA, lda, queue ), blas::Error );
    assert_throw( blas::her2( layout,    uplo,     n, alpha, dx, incx, dy,    0, dA, lda, queue ), blas::Error );
    assert_throw( blas::her2( layout,    uplo,     n, alpha, dx, incx, dy, incy, dA, n-1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm,
                llong( n ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "A = " ); print_matrix( n, n, A, lda );
        printf( "x = " ); print_vector( n, x, incx );
        printf( "y = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::her2( layout, uplo, n, alpha, dx, incx, dy, incy, dA, lda, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::her2( n );
    double gbyte = blas::Gbyte< scalar_t >::her2( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
    params.gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

    blas::device_copy_matrix( n, n, dA, lda, A, lda, queue );
    queue.sync();

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, A, lda );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_her2( cblas_layout_const(layout), cblas_uplo_const(uplo),
                    n, alpha, x, incx, y, incy, Aref, lda );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
        params.ref_gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( n, n, Aref, lda );
        }

        // check error compared to reference
        // beta = 1
        real_t error;
        bool okay;
        check_herk( uplo, n, 2, alpha, real_t(1), Xnorm, Ynorm, Anorm,
                    Aref, lda, A, lda, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    // Memory is managed by the pool and will be reused or freed automatically
    // No explicit deletion needed here
}

// -----------------------------------------------------------------------------
void test_her2_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_her2_device_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_her2_device_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_her2_device_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_her2_device_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
