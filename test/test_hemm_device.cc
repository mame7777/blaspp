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
template <typename TA, typename TB, typename TC>
struct HemmDeviceMemoryPool {
    // Host memory
    TA* A = nullptr;
    TB* B = nullptr;
    TC* C = nullptr;
    TC* Cref = nullptr;

    // Device memory
    TA* dA = nullptr;
    TB* dB = nullptr;
    TC* dC = nullptr;

    // Memory sizes
    size_t size_A = 0;
    size_t size_B = 0;
    size_t size_C = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sA, size_t sB, size_t sC, int64_t dev) {
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {
            free();
            need_realloc = true;
        }

        // Check if we need to allocate larger memory
        if (A == nullptr || size_A < sA || size_B < sB || size_C < sC) {
            if (A != nullptr) {
                free();
            }
            need_realloc = true;
        }

        // Allocate new memory if needed
        if (need_realloc || A == nullptr) {
            // Use larger size to avoid frequent reallocation
            size_A = (sA > size_A) ? sA : size_A;
            size_B = (sB > size_B) ? sB : size_B;
            size_C = (sC > size_C) ? sC : size_C;
            device_id = dev;

            A = new TA[size_A];
            B = new TB[size_B];
            C = new TC[size_C];
            Cref = new TC[size_C];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dA = blas::device_malloc<TA>(size_A, *queue);
            dB = blas::device_malloc<TB>(size_B, *queue);
            dC = blas::device_malloc<TC>(size_C, *queue);
        }
    }

    // Free all memory
    void free() {
        if (A != nullptr) {
            delete[] A;
            delete[] B;
            delete[] C;
            delete[] Cref;
            A = nullptr;
            B = nullptr;
            C = nullptr;
            Cref = nullptr;
        }

        if (dA != nullptr && queue != nullptr) {
            blas::device_free(dA, *queue);
            blas::device_free(dB, *queue);
            blas::device_free(dC, *queue);
            dA = nullptr;
            dB = nullptr;
            dC = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_A = 0;
        size_B = 0;
        size_C = 0;
        device_id = -1;
    }

        ~HemmDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (A != nullptr) {
            delete[] A;
            delete[] B;
            delete[] C;
            delete[] Cref;
            A = nullptr;
            B = nullptr;
            C = nullptr;
            Cref = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static HemmDeviceMemoryPool<float, float, float> pool_s;
static HemmDeviceMemoryPool<double, double, double> pool_d;
static HemmDeviceMemoryPool<std::complex<float>, std::complex<float>, std::complex<float>> pool_c;
static HemmDeviceMemoryPool<std::complex<double>, std::complex<double>, std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
void test_hemm_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo, blas::Side, blas::Layout, blas::max;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha.get<scalar_t>();
    scalar_t beta   = params.beta.get<scalar_t>();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t device  = params.device();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // Get appropriate memory pool
    HemmDeviceMemoryPool<TA, TB, TC>* pool = nullptr;
    if (std::is_same<TA, float>::value && std::is_same<TB, float>::value && std::is_same<TC, float>::value) {
        pool = reinterpret_cast<HemmDeviceMemoryPool<TA, TB, TC>*>(&pool_s);
    } else if (std::is_same<TA, double>::value && std::is_same<TB, double>::value && std::is_same<TC, double>::value) {
        pool = reinterpret_cast<HemmDeviceMemoryPool<TA, TB, TC>*>(&pool_d);
    } else if (std::is_same<TA, std::complex<float>>::value && std::is_same<TB, std::complex<float>>::value && std::is_same<TC, std::complex<float>>::value) {
        pool = reinterpret_cast<HemmDeviceMemoryPool<TA, TB, TC>*>(&pool_c);
    } else if (std::is_same<TA, std::complex<double>>::value && std::is_same<TB, std::complex<double>>::value && std::is_same<TC, std::complex<double>>::value) {
        pool = reinterpret_cast<HemmDeviceMemoryPool<TA, TB, TC>*>(&pool_z);
    }

    // setup
    int64_t An = (side == Side::Left ? m : n);
    int64_t Cm = m;
    int64_t Cn = n;
    if (layout == Layout::RowMajor)
        std::swap( Cm, Cn );
    int64_t lda = max( roundup( An, align ), 1 );
    int64_t ldb = max( roundup( Cm, align ), 1 );
    int64_t ldc = max( roundup( Cm, align ), 1 );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Cn;
    size_t size_C = size_t(ldc)*Cn;

    // Allocate or reuse memory from pool
    pool->allocate(size_A, size_B, size_C, device);

    TA* A = pool->A;
    TB* B = pool->B;
    TC* C = pool->C;
    TC* Cref = pool->Cref;

    // device specifics
    blas::Queue& queue = *(pool->queue);
    TA* dA = pool->dA;
    TB* dB = pool->dB;
    TC* dC = pool->dC;

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", Cm, Cn, C, ldc, Cref, ldc );

    blas::device_copy_matrix( An, An, A, lda, dA, lda, queue );
    blas::device_copy_matrix( Cm, Cn, B, ldb, dB, ldb, queue );
    blas::device_copy_matrix( Cm, Cn, C, ldc, dC, ldc, queue );
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lansy( "f", to_c_string( uplo ), An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Cm, Cn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

    // test error exits
    assert_throw( blas::hemm( Layout(0), side,     uplo,     m,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::hemm( layout,    Side(0),  uplo,     m,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::hemm( layout,    side,     Uplo(0),  m,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::hemm( layout,    side,     uplo,    -1,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::hemm( layout,    side,     uplo,     m, -1, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::hemm( layout, Side::Left,  uplo,     m,  n, alpha, dA, m-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::hemm( layout, Side::Right, uplo,     m,  n, alpha, dA, n-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::hemm( Layout::ColMajor, side, uplo,  m,  n, alpha, dA, lda, dB, m-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::hemm( Layout::RowMajor, side, uplo,  m,  n, alpha, dA, lda, dB, n-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::hemm( Layout::ColMajor, side, uplo,  m,  n, alpha, dA, lda, dB, ldb, beta, dC, m-1, queue ), blas::Error );
    assert_throw( blas::hemm( Layout::RowMajor, side, uplo,  m,  n, alpha, dA, lda, dB, ldb, beta, dC, n-1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "side %c, uplo %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B  m=%5lld,  n=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C  m=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                to_char( side ), to_char( uplo ),
                llong( An ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( m ), llong( n ), llong( ldb ), llong( size_B ), Bnorm,
                llong( m ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( An, An, A, lda );
        printf( "B = "    ); print_matrix( Cm, Cn, B, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::hemm( layout, side, uplo, m, n,
                alpha, dA, lda, dB, ldb, beta, dC, ldc, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::hemm( side, m, n );
    params.time()   = time;
    params.gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
    blas::device_copy_matrix(Cm, Cn, dC, ldc, C, ldc, queue);
    queue.sync();

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_hemm( cblas_layout_const(layout),
                    cblas_side_const(side),
                    cblas_uplo_const(uplo),
                    m, n, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, An, alpha, beta, Anorm, Bnorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    // Memory is managed by the pool and will be reused or freed automatically
    // No explicit deletion needed here
}

// -----------------------------------------------------------------------------
void test_hemm_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_hemm_device_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_hemm_device_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_hemm_device_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hemm_device_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
