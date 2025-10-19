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
template <typename Tx>
struct IamaxDeviceMemoryPool {
    // Host memory
    Tx* x = nullptr;

    // Device memory
    Tx* dx = nullptr;
    int64_t* result_dev = nullptr;

    // Memory sizes
    size_t size_x = 0;

    // Queue
    blas::Queue* queue = nullptr;
    int64_t device_id = -1;

    // Allocate memory for given sizes
    void allocate(size_t sx, int64_t dev, bool need_result_dev) {
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

            x = new Tx[size_x];

            if (queue != nullptr) {
                delete queue;
            }
            queue = new blas::Queue(device_id);

            dx = blas::device_malloc<Tx>(size_x, *queue);
            if (need_result_dev) {
                result_dev = blas::device_malloc<int64_t>(1, *queue);
            }
        }
        else if (need_result_dev && result_dev == nullptr) {
            result_dev = blas::device_malloc<int64_t>(1, *queue);
        }
    }

    // Free all memory
    void free() {
        if (x != nullptr) {
            delete[] x;
            x = nullptr;
        }

        if (dx != nullptr && queue != nullptr) {
            blas::device_free(dx, *queue);
            dx = nullptr;
        }

        if (result_dev != nullptr && queue != nullptr) {
            blas::device_free(result_dev, *queue);
            result_dev = nullptr;
        }

        if (queue != nullptr) {
            delete queue;
            queue = nullptr;
        }

        size_x = 0;
        device_id = -1;
    }

    ~IamaxDeviceMemoryPool() {
        // Only free host memory; device memory cleanup is skipped
        // to avoid errors when CUDA context is already destroyed at program exit
        if (x != nullptr) {
            delete[] x;
            x = nullptr;
        }
        // DO NOT call blas::device_free()
        // DO NOT delete queue
    }
};

// Static memory pools for each data type
static IamaxDeviceMemoryPool<float> pool_s;
static IamaxDeviceMemoryPool<double> pool_d;
static IamaxDeviceMemoryPool<std::complex<float>> pool_c;
static IamaxDeviceMemoryPool<std::complex<double>> pool_z;

// -----------------------------------------------------------------------------
template <typename T>
void test_iamax_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas;
    using scalar_t = blas::scalar_type< T >;
    using real_t   = blas::real_type< scalar_t >;
    using std::abs;
    using blas::max;

    // get & mark input values
    char mode = params.pointer_mode();
    int64_t n = params.dim.n();
    int64_t incx = params.incx();
    int64_t device = params.device();
    int64_t verbose = params.verbose();

    int64_t result_host;
    int64_t* result_ptr = &result_host;

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
    IamaxDeviceMemoryPool<T>* pool = nullptr;
    if (std::is_same<T, float>::value) {
        pool = reinterpret_cast<IamaxDeviceMemoryPool<T>*>(&pool_s);
    } else if (std::is_same<T, double>::value) {
        pool = reinterpret_cast<IamaxDeviceMemoryPool<T>*>(&pool_d);
    } else if (std::is_same<T, std::complex<float>>::value) {
        pool = reinterpret_cast<IamaxDeviceMemoryPool<T>*>(&pool_c);
    } else if (std::is_same<T, std::complex<double>>::value) {
        pool = reinterpret_cast<IamaxDeviceMemoryPool<T>*>(&pool_z);
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );

    // Allocate or reuse memory from pool
    pool->allocate(size_x, device, mode == 'd');

    T* x = pool->x;

    // device specifics
    blas::Queue& queue = *(pool->queue);
    T* dx = pool->dx;

    int64_t idist = 1;
    int iseed[ 4 ] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );

    blas::device_copy_vector( n, x, std::abs(incx), dx, std::abs(incx), queue );
    queue.sync();

    if (mode == 'd') {
        result_ptr = pool->result_dev;
        #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode( queue.handle(), CUBLAS_POINTER_MODE_DEVICE );
        #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
        #endif
    }

    // test error exits
    assert_throw( blas::iamax( -1, x, incx, result_ptr, queue ), blas::Error );
    assert_throw( blas::iamax(  n, x,    0, result_ptr, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::iamax( n, dx, incx, result_ptr, queue );
    queue.sync();
    time = get_wtime() - time;

    if (mode == 'd') {
        device_memcpy( &result_host, result_ptr, 1, queue );
    }

    double gflop = blas::Gflop< T >::iamax( n );
    double gbyte = blas::Gbyte< T >::iamax( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
    params.gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

    blas::device_copy_vector( n, dx, std::abs(incx), x, std::abs(incx), queue );
    queue.sync();

    if (verbose >= 1) {
        printf( "result = %5lld\n", llong( result_host ) );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        int64_t ref = cblas_iamax( n, x, incx );
        if (n == 0)
            ref -= 1;
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = (gflop > 0 ? gflop / time : testsweeper::no_data_flag);
        params.ref_gbytes() = (gbyte > 0 ? gbyte / time : testsweeper::no_data_flag);

        if (verbose >= 1) {
            printf( "ref    = %5lld\n", llong( ref ) );
        }

        // check error compared to reference
        real_t error = abs( ref - result_host );
        params.error() = error;

        // iamax must be exact!
        params.okay() = (error == 0);
    }

    // Memory is managed by the pool and will be reused or freed automatically
    // No explicit deletion needed here
}

// -----------------------------------------------------------------------------
void test_iamax_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_iamax_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_iamax_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_iamax_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_iamax_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
