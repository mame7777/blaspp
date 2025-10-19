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
struct Syr2kDeviceMemoryPool {
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

        ~Syr2kDeviceMemoryPool() {
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
static Syr2kDeviceMemoryPool<float, float, float> pool_s;
static Syr2kDeviceMemoryPool<double, double, double> pool_d;
static Syr2kDeviceMemoryPool<std::complex<float>, std::complex<float>, std::complex<float>> pool_c;
static Syr2kDeviceMemoryPool<std::complex<double>, std::complex<double>, std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
void test_syr2k_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo, blas::Op, blas::Layout, blas::max;
    using scalar_t = blas::scalar_type< TA, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha.get<scalar_t>();
    scalar_t beta   = params.beta.get<scalar_t>();
    int64_t n       = params.dim.n();
    int64_t k       = params.dim.k();
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

    // setup
    int64_t Am = (trans == Op::NoTrans ? n : k);
    int64_t An = (trans == Op::NoTrans ? k : n);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda = max( roundup( Am, align ), 1 );
    int64_t ldb = max( roundup( Am, align ), 1 );
    int64_t ldc = max( roundup(  n, align ), 1 );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*An;
    size_t size_C = size_t(ldc)*n;

    // Get appropriate memory pool based on scalar type
    Syr2kDeviceMemoryPool<TA, TB, TC>* pool;
    if (std::is_same<TA, float>::value) {
        pool = reinterpret_cast<Syr2kDeviceMemoryPool<TA, TB, TC>*>(&pool_s);
    }
    else if (std::is_same<TA, double>::value) {
        pool = reinterpret_cast<Syr2kDeviceMemoryPool<TA, TB, TC>*>(&pool_d);
    }
    else if (std::is_same<TA, std::complex<float>>::value) {
        pool = reinterpret_cast<Syr2kDeviceMemoryPool<TA, TB, TC>*>(&pool_c);
    }
    else {
        pool = reinterpret_cast<Syr2kDeviceMemoryPool<TA, TB, TC>*>(&pool_z);
    }

    // Allocate memory from pool
    pool->allocate(size_A, size_B, size_C, device);

    // Use pointers from pool
    TA* A = pool->A;
    TB* B = pool->B;
    TC* C = pool->C;
    TC* Cref = pool->Cref;
    TA* dA = pool->dA;
    TB* dB = pool->dB;
    TC* dC = pool->dC;
    blas::Queue& queue = *(pool->queue);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", n, n, C, ldc, Cref, ldc );

    blas::device_copy_matrix( Am, An, A, lda, dA, lda, queue );
    blas::device_copy_matrix( Am, An, B, ldb, dB, ldb, queue );
    blas::device_copy_matrix( n, n, C, ldc, dC, ldc, queue );
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Am, An, B, ldb, work );
    real_t Cnorm = lapack_lansy( "f", to_c_string( uplo ), n, C, ldc, work );

    // test error exits
    assert_throw( blas::syr2k( Layout(0), uplo,    trans,  n,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( layout,    Uplo(0), trans,  n,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( layout,    uplo,    Op(0),  n,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( layout,    uplo,    trans, -1,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( layout,    uplo,    trans,  n, -1, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::NoTrans,   n, k, alpha, dA, n-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::Trans,     n, k, alpha, dA, k-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::ConjTrans, n, k, alpha, dA, k-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::NoTrans,   n, k, alpha, dA, k-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::Trans,     n, k, alpha, dA, n-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::ConjTrans, n, k, alpha, dA, n-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::NoTrans,   n, k, alpha, dA, lda, dB, n-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::Trans,     n, k, alpha, dA, lda, dB, k-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::ConjTrans, n, k, alpha, dA, lda, dB, k-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::NoTrans,   n, k, alpha, dA, lda, dB, k-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::Trans,     n, k, alpha, dA, lda, dB, n-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::ConjTrans, n, k, alpha, dA, lda, dB, n-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::syr2k( layout,    uplo,    trans,  n,  k, alpha, dA, lda, dB, ldb, beta, dC, n-1, queue ), blas::Error );

    if (blas::is_complex_v<scalar_t>) {
        // complex syr2k doesn't allow ConjTrans, only Trans
        assert_throw( blas::syr2k( layout, uplo, Op::ConjTrans, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    }

    if (verbose >= 1) {
        printf( "\n"
                "uplo %c, trans %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bn=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C  n=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                to_char( uplo ), to_char( trans ),
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Am ), llong( An ), llong( ldb ), llong( size_B ), Bnorm,
                llong( n ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Am, An, B, ldb );
        printf( "C = "    ); print_matrix(  n,  n, C, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::syr2k( layout, uplo, trans, n, k,
                 alpha, dA, lda, dB, ldb, beta, dC, ldc, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::syr2k( n, k );
    params.time()   = time;
    params.gflops() = gflop / time;
    blas::device_copy_matrix(n, n, dC, ldc, C, ldc, queue);
    queue.sync();

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( n, n, C, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_syr2k( cblas_layout_const(layout),
                     cblas_uplo_const(uplo),
                     cblas_trans_const(trans),
                     n, k, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( n, n, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_herk( uplo, n, 2*k, alpha, beta, Anorm, Bnorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }
}

// -----------------------------------------------------------------------------
void test_syr2k_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_syr2k_device_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_syr2k_device_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_syr2k_device_work< std::complex<float>, std::complex<float>,
                             std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_syr2k_device_work< std::complex<double>, std::complex<double>,
                             std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
