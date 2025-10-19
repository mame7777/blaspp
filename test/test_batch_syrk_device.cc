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
// Memory pool structure for batch device memory management
template <typename TA, typename TC>
struct BatchSyrkDeviceMemoryPool {
    using scalar_t = blas::scalar_type<TA, TC>;
    using real_t = blas::real_type<scalar_t>;

    // Host memory
    TA* A = nullptr;
    TC* C = nullptr;
    TC* Cref = nullptr;

    // Device memory
    TA* dA = nullptr;
    TC* dC = nullptr;

    // Norm arrays
    real_t* Anorm = nullptr;
    real_t* Cnorm = nullptr;

    // Memory sizes
    size_t size_A = 0;
    size_t size_C = 0;
    size_t batch_size = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes and batch count
    void allocate(size_t sA, size_t sC, size_t batch, int64_t dev) {
        // Free existing memory if sizes changed or different device
        if (queue != nullptr &&
            (size_A != sA || size_C != sC ||
             batch_size != batch || device_id != dev)) {
            free();
        }

        // Allocate if needed
        if (A == nullptr || size_A != sA || batch_size != batch) {
            size_A = sA;
            size_C = sC;
            batch_size = batch;
            device_id = dev;

            // Allocate host memory
            A = new TA[batch * size_A];
            C = new TC[batch * size_C];
            Cref = new TC[batch * size_C];

            // Allocate norm arrays
            Anorm = new real_t[batch];
            Cnorm = new real_t[batch];

            // Create queue if needed
            if (queue == nullptr) {
                queue = new blas::Queue(device_id);
            }

            // Allocate device memory
            dA = blas::device_malloc<TA>(batch * size_A, *queue);
            dC = blas::device_malloc<TC>(batch * size_C, *queue);
            queue->sync();
        }
    }

    // Free all memory
    void free() {
        if (A != nullptr) {
            delete[] A;
            delete[] C;
            delete[] Cref;
            delete[] Anorm;
            delete[] Cnorm;
            A = nullptr;
            C = nullptr;
            Cref = nullptr;
            Anorm = nullptr;
            Cnorm = nullptr;
        }

        if (queue != nullptr && dA != nullptr) {
            blas::device_free(dA, *queue);
            blas::device_free(dC, *queue);
            dA = nullptr;
            dC = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_A = 0;
        size_C = 0;
        batch_size = 0;
        device_id = -1;
    }

        ~BatchSyrkDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (A != nullptr) {
            delete[] A;
            delete[] C;
            delete[] Cref;
            delete[] Anorm;
            delete[] Cnorm;
            A = nullptr;
            C = nullptr;
            Cref = nullptr;
            Anorm = nullptr;
            Cnorm = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static BatchSyrkDeviceMemoryPool<float, float> pool_float;
static BatchSyrkDeviceMemoryPool<double, double> pool_double;
static BatchSyrkDeviceMemoryPool<std::complex<float>, std::complex<float>> pool_complex_float;
static BatchSyrkDeviceMemoryPool<std::complex<double>, std::complex<double>> pool_complex_double;

// Helper function to get the appropriate pool
template <typename TA, typename TC>
BatchSyrkDeviceMemoryPool<TA, TC>& get_pool() {
    if constexpr (std::is_same_v<TA, float>) {
        return reinterpret_cast<BatchSyrkDeviceMemoryPool<TA, TC>&>(pool_float);
    }
    else if constexpr (std::is_same_v<TA, double>) {
        return reinterpret_cast<BatchSyrkDeviceMemoryPool<TA, TC>&>(pool_double);
    }
    else if constexpr (std::is_same_v<TA, std::complex<float>>) {
        return reinterpret_cast<BatchSyrkDeviceMemoryPool<TA, TC>&>(pool_complex_float);
    }
    else if constexpr (std::is_same_v<TA, std::complex<double>>) {
        return reinterpret_cast<BatchSyrkDeviceMemoryPool<TA, TC>&>(pool_complex_double);
    }
}

// -----------------------------------------------------------------------------
template <typename TA, typename TC>
void test_batch_syrk_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Op, blas::Layout, blas::max;
    using scalar_t = blas::scalar_type< TA, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans_     = params.trans();
    blas::Uplo uplo_    = params.uplo();
    scalar_t alpha_     = params.alpha.get<scalar_t>();
    scalar_t beta_      = params.beta.get<scalar_t>();
    int64_t n_          = params.dim.n();
    int64_t k_          = params.dim.k();
    size_t  batch       = params.batch();
    int64_t device      = params.device();
    int64_t align       = params.align();
    int64_t verbose     = params.verbose();

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
    int64_t Am = (trans_ == Op::NoTrans ? n_ : k_);
    int64_t An = (trans_ == Op::NoTrans ? k_ : n_);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda_ = max( roundup( Am, align ), 1 );
    int64_t ldc_ = max( roundup( n_, align ), 1 );
    size_t size_A = size_t(lda_)*An;
    size_t size_C = size_t(ldc_)*n_;

    // Get memory pool and allocate
    auto& pool = get_pool<TA, TC>();
    pool.allocate(size_A, size_C, batch, device);

    // Use pool memory
    TA* A = pool.A;
    TC* C = pool.C;
    TC* Cref = pool.Cref;
    TA* dA = pool.dA;
    TC* dC = pool.dC;
    blas::Queue& queue = *pool.queue;

    // pointer arrays
    std::vector<TA*>    Aarray( batch );
    std::vector<TC*>    Carray( batch );
    std::vector<TC*> Crefarray( batch );
    std::vector<TA*>   dAarray( batch );
    std::vector<TC*>   dCarray( batch );

    for (size_t i = 0; i < batch; ++i) {
         Aarray[i]   =  A   + i * size_A;
         Carray[i]   =  C   + i * size_C;
        Crefarray[i] = Cref + i * size_C;
        dAarray[i]   = dA   + i * size_A;
        dCarray[i]   = dC   + i * size_C;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<blas::Uplo> uplo(1, uplo_);
    std::vector<blas::Op>   trans(1, trans_);
    std::vector<int64_t>    n(1, n_);
    std::vector<int64_t>    k(1, k_);
    std::vector<int64_t>    lda(1, lda_);
    std::vector<int64_t>    ldc(1, ldc_);
    std::vector<scalar_t>   alpha(1, alpha_);
    std::vector<scalar_t>   beta(1, beta_);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );
    lapack_larnv( idist, iseed, batch * size_C, C );
    lapack_lacpy( "g", n_, batch * n_, C, ldc_, Cref, ldc_ );

    blas::device_copy_matrix(Am, batch * An, A, lda_, dA, lda_, queue);
    blas::device_copy_matrix(n_, batch * n_, C, ldc_, dC, ldc_, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t* Anorm = pool.Anorm;
    real_t* Cnorm = pool.Cnorm;

    for (size_t s = 0; s < batch; ++s) {
        Anorm[s] = lapack_lange( "f", Am, An, Aarray[s], lda_, work );
        Cnorm[s] = lapack_lansy( "f", to_c_string( uplo_ ), n_, Carray[s], ldc_, work );
    }

    // decide error checking mode
    info.resize( 0 );

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::batch::syrk( layout, uplo, trans, n, k, alpha, dAarray, lda, beta, dCarray, ldc,
                       batch, info, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = batch * blas::Gflop< scalar_t >::syrk( n_, k_ );
    params.time()   = time;
    params.gflops() = gflop / time;
    blas::device_copy_matrix(n_, batch * n_, dC, ldc_, C, ldc_, queue);
    queue.sync();

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (size_t s = 0; s < batch; ++s) {
            cblas_syrk( cblas_layout_const(layout),
                        cblas_uplo_const(uplo_),
                        cblas_trans_const(trans_),
                        n_, k_, alpha_, Aarray[s], lda_, beta_, Crefarray[s], ldc_ );
        }
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        // check error compared to reference
        real_t err, error = 0;
        bool ok, okay = true;
        for (size_t s = 0; s < batch; ++s) {
            check_herk( uplo_, n_, k_, alpha_, beta_, Anorm[s], Anorm[s], Cnorm[s],
                        Crefarray[s], ldc_, Carray[s], ldc_, verbose, &err, &ok );

            error = std::max( error, err );
            okay &= ok;
        }

        params.error() = error;
        params.okay() = okay;
    }

    // Memory is managed by the pool, no need to free here
}

// -----------------------------------------------------------------------------
void test_batch_syrk_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_batch_syrk_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_batch_syrk_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_batch_syrk_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_batch_syrk_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
