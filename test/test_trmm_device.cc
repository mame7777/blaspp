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
template <typename TA, typename TB>
struct TrmmDeviceMemoryPool {
    // Host memory
    TA* A = nullptr;
    TB* B = nullptr;
    TB* Bref = nullptr;

    // Device memory
    TA* dA = nullptr;
    TB* dB = nullptr;

    // Memory sizes
    size_t size_A = 0;
    size_t size_B = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sA, size_t sB, int64_t dev) {
        bool need_realloc = false;

        // Check if device changed
        if (device_id != dev && queue != nullptr) {
            free();
            need_realloc = true;
        }

        // Check if we need to allocate larger memory
        if (A == nullptr || size_A < sA || size_B < sB) {
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
            device_id = dev;

            A = new TA[size_A];
            B = new TB[size_B];
            Bref = new TB[size_B];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dA = blas::device_malloc<TA>(size_A, *queue);
            dB = blas::device_malloc<TB>(size_B, *queue);
        }
    }

    // Free all memory
    void free() {
        if (A != nullptr) {
            delete[] A;
            delete[] B;
            delete[] Bref;
            A = nullptr;
            B = nullptr;
            Bref = nullptr;
        }

        if (dA != nullptr && queue != nullptr) {
            blas::device_free(dA, *queue);
            blas::device_free(dB, *queue);
            dA = nullptr;
            dB = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_A = 0;
        size_B = 0;
        device_id = -1;
    }

        ~TrmmDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (A != nullptr) {
            delete[] A;
            delete[] B;
            delete[] Bref;
            A = nullptr;
            B = nullptr;
            Bref = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static TrmmDeviceMemoryPool<float, float> pool_s;
static TrmmDeviceMemoryPool<double, double> pool_d;
static TrmmDeviceMemoryPool<std::complex<float>, std::complex<float>> pool_c;
static TrmmDeviceMemoryPool<std::complex<double>, std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
template <typename TA, typename TB>
void test_trmm_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Uplo, blas::Side, blas::Op, blas::Layout, blas::Diag, blas::max;
    using scalar_t = blas::scalar_type< TA, TB >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    blas::Op trans  = params.trans();
    blas::Diag diag = params.diag();
    scalar_t alpha  = params.alpha.get<scalar_t>();
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

    // ----------
    // setup
    int64_t Am = (side == Side::Left ? m : n);
    int64_t Bm = m;
    int64_t Bn = n;
    if (layout == Layout::RowMajor)
        std::swap( Bm, Bn );
    int64_t lda = max( roundup( Am, align ), 1 );
    int64_t ldb = max( roundup( Bm, align ), 1 );
    size_t size_A = size_t(lda)*Am;
    size_t size_B = size_t(ldb)*Bn;

    // Get appropriate memory pool based on scalar type
    TrmmDeviceMemoryPool<TA, TB>* pool;
    if (std::is_same<TA, float>::value) {
        pool = reinterpret_cast<TrmmDeviceMemoryPool<TA, TB>*>(&pool_s);
    }
    else if (std::is_same<TA, double>::value) {
        pool = reinterpret_cast<TrmmDeviceMemoryPool<TA, TB>*>(&pool_d);
    }
    else if (std::is_same<TA, std::complex<float>>::value) {
        pool = reinterpret_cast<TrmmDeviceMemoryPool<TA, TB>*>(&pool_c);
    }
    else {
        pool = reinterpret_cast<TrmmDeviceMemoryPool<TA, TB>*>(&pool_z);
    }

    // Allocate memory from pool
    pool->allocate(size_A, size_B, device);

    // Use pointers from pool
    TA* A = pool->A;
    TB* B = pool->B;
    TB* Bref = pool->Bref;
    TA* dA = pool->dA;
    TB* dB = pool->dB;
    blas::Queue& queue = *(pool->queue);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );  // TODO: generate
    lapack_larnv( idist, iseed, size_B, B );  // TODO
    lapack_lacpy( "g", Bm, Bn, B, ldb, Bref, ldb );

    blas::device_copy_matrix( Am, Am, A, lda, dA, lda, queue );
    blas::device_copy_matrix( Bm, Bn, B, ldb, dB, ldb, queue );
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lantr( "f", to_c_string( uplo ), to_c_string( diag ),
                                 Am, Am, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );

    // test error exits
    assert_throw( blas::trmm( Layout(0), side,    uplo,    trans, diag,     m,  n, alpha, dA, lda, dB, ldb, queue ), blas::Error );
    assert_throw( blas::trmm( layout,    Side(0), uplo,    trans, diag,     m,  n, alpha, dA, lda, dB, ldb, queue ), blas::Error );
    assert_throw( blas::trmm( layout,    side,    Uplo(0), trans, diag,     m,  n, alpha, dA, lda, dB, ldb, queue ), blas::Error );
    assert_throw( blas::trmm( layout,    side,    uplo,    Op(0), diag,     m,  n, alpha, dA, lda, dB, ldb, queue ), blas::Error );
    assert_throw( blas::trmm( layout,    side,    uplo,    trans, Diag(0),  m,  n, alpha, dA, lda, dB, ldb, queue ), blas::Error );
    assert_throw( blas::trmm( layout,    side,    uplo,    trans, diag,    -1,  n, alpha, dA, lda, dB, ldb, queue ), blas::Error );
    assert_throw( blas::trmm( layout,    side,    uplo,    trans, diag,     m, -1, alpha, dA, lda, dB, ldb, queue ), blas::Error );

    assert_throw( blas::trmm( layout, Side::Left,  uplo,   trans, diag,     m,  n, alpha, dA, m-1, dB, ldb, queue ), blas::Error );
    assert_throw( blas::trmm( layout, Side::Right, uplo,   trans, diag,     m,  n, alpha, dA, n-1, dB, ldb, queue ), blas::Error );

    assert_throw( blas::trmm( Layout::ColMajor, side, uplo, trans, diag,    m,  n, alpha, dA, lda, dB, m-1, queue ), blas::Error );
    assert_throw( blas::trmm( Layout::RowMajor, side, uplo, trans, diag,    m,  n, alpha, dA, lda, dB, n-1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, Am=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm=%.2e\n",
                llong( Am ), llong( Am ), llong( lda ), llong( size_A ), Anorm,
                llong( Bm ), llong( Bn ), llong( ldb ), llong( size_B ), Bnorm );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( Am, Am, A, lda );
        printf( "B = " ); print_matrix( Bm, Bn, B, ldb );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::trmm( layout, side, uplo, trans, diag, m, n, alpha, dA, lda, dB, ldb, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::trmm( side, m, n );
    params.time()   = time;
    params.gflops() = gflop / time;
    blas::device_copy_matrix(Bm, Bn, dB, ldb, B, ldb, queue);
    queue.sync();

    if (verbose >= 2) {
        printf( "X = " ); print_matrix( Bm, Bn, B, ldb );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_trmm( cblas_layout_const(layout),
                    cblas_side_const(side),
                    cblas_uplo_const(uplo),
                    cblas_trans_const(trans),
                    cblas_diag_const(diag),
                    m, n, alpha, A, lda, Bref, ldb );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Xref = " ); print_matrix( Bm, Bn, Bref, ldb );
        }

        // check error compared to reference
        // Am is reduction dimension
        // beta = 0, Cnorm = 0 (initial).
        real_t error;
        bool okay;
        check_gemm( Bm, Bn, Am, alpha, scalar_t(0), Anorm, Bnorm, real_t(0),
                    Bref, ldb, B, ldb, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }
}

// -----------------------------------------------------------------------------
void test_trmm_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trmm_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_trmm_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_trmm_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trmm_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
