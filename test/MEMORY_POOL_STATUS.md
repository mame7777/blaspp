# Memory Pool Pattern Application Status

## Overview
This document tracks the application of the memory pool pattern to BLAS++ device test files.

## Completed Files (5/18)

### Level 2 BLAS
1. ✅ `test_hemv_device.cc` - HemvDeviceMemoryPool (A, x, y, yref)
2. ✅ `test_her_device.cc` - HerDeviceMemoryPool (A, Aref, x)
3. ✅ `test_her2_device.cc` - Her2DeviceMemoryPool (A, Aref, x, y)
4. ✅ `test_symv_device.cc` - SymvDeviceMemoryPool (A, x, y, yref)
5. ✅ `test_syr_device.cc` - SyrDeviceMemoryPool (A, Aref, x)

## Remaining Files (13/18)

### Level 2 BLAS (3 files)
6. ⏳ `test_syr2_device.cc` - Needs: Syr2DeviceMemoryPool (A, Aref, x, y) - Same as Her2
7. ⏳ `test_trmv_device.cc` - Needs: TrmvDeviceMemoryPool (A, x, xref)
8. ⏳ `test_trsv_device.cc` - Needs: TrsvDeviceMemoryPool (A, x, xref)

### Level 3 BLAS (8 files)
9. ⏳ `test_hemm_device.cc` - Needs: HemmDeviceMemoryPool (A, B, C, Cref) - Same as Gemm
10. ⏳ `test_herk_device.cc` - Needs: HerkDeviceMemoryPool (A, C, Cref)
11. ⏳ `test_her2k_device.cc` - Needs: Her2kDeviceMemoryPool (A, B, C, Cref)
12. ⏳ `test_symm_device.cc` - Needs: SymmDeviceMemoryPool (A, B, C, Cref) - Same as Gemm
13. ⏳ `test_syrk_device.cc` - Needs: SyrkDeviceMemoryPool (A, C, Cref)
14. ⏳ `test_syr2k_device.cc` - Needs: Syr2kDeviceMemoryPool (A, B, C, Cref)
15. ⏳ `test_trmm_device.cc` - Needs: TrmmDeviceMemoryPool (A, B, C, Cref)
16. ⏳ `test_trsm_device.cc` - Needs: TrsmDeviceMemoryPool (A, B, C, Cref)

### Auxiliary Operations (2 files)
17. ⏳ `test_geadd_device.cc` - Needs: GeaddDeviceMemoryPool (A, B, Bref)
18. ⏳ `test_tzadd_device.cc` - Needs: TzaddDeviceMemoryPool (A, B, Bref)

## Memory Pool Patterns

### Pattern 1: Matrix + 2 Vectors (with yref)
Used by: hemv, symv
- Host: A, x, y, yref
- Device: dA, dx, dy
- Sizes: size_A, size_x, size_y
- Allocate: `pool->allocate(size_A, size_x, size_y, device)`

### Pattern 2: Matrix + Vector (with Aref)
Used by: her, syr
- Host: A, Aref, x
- Device: dA, dx
- Sizes: size_A, size_x
- Allocate: `pool->allocate(size_A, size_x, device)`

### Pattern 3: Matrix + 2 Vectors (with Aref)
Used by: her2, syr2
- Host: A, Aref, x, y
- Device: dA, dx, dy
- Sizes: size_A, size_x, size_y
- Allocate: `pool->allocate(size_A, size_x, size_y, device)`

### Pattern 4: Matrix + Vector (with xref)
Used by: trmv, trsv
- Host: A, x, xref
- Device: dA, dx
- Sizes: size_A, size_x
- Allocate: `pool->allocate(size_A, size_x, device)`

### Pattern 5: Three Matrices (with Cref)
Used by: hemm, her2k, symm, syr2k, trmm, trsm
- Host: A, B, C, Cref
- Device: dA, dB, dC
- Sizes: size_A, size_B, size_C
- Allocate: `pool->allocate(size_A, size_B, size_C, device)`

### Pattern 6: Two Matrices (with Cref)
Used by: herk, syrk
- Host: A, C, Cref
- Device: dA, dC
- Sizes: size_A, size_C
- Allocate: `pool->allocate(size_A, size_C, device)`

### Pattern 7: Two Matrices (with Bref)
Used by: geadd, tzadd
- Host: A, B, Bref
- Device: dA, dB
- Sizes: size_A, size_B
- Allocate: `pool->allocate(size_A, size_B, device)`

## Implementation Steps for Each File

For each remaining file:

1. **Add Memory Pool Structure** (before function):
   ```cpp
   template <typename T1, typename T2, ...>
   struct XxxDeviceMemoryPool {
       // Host memory pointers
       // Device memory pointers
       // Size members
       // Queue and device_id
       // allocate() method
       // free() method
       // Destructor
   };

   // Static pool instances for each data type
   static XxxDeviceMemoryPool<...> pool_s;
   static XxxDeviceMemoryPool<...> pool_d;
   static XxxDeviceMemoryPool<...> pool_c;
   static XxxDeviceMemoryPool<...> pool_z;
   ```

2. **Get Memory Pool** (replace allocation section):
   ```cpp
   // Get appropriate memory pool
   XxxDeviceMemoryPool<...>* pool = nullptr;
   if (std::is_same<TA, float>::value && ...) {
       pool = reinterpret_cast<XxxDeviceMemoryPool<...>*>(&pool_s);
   } else if ...
   ```

3. **Use Pool Allocation**:
   ```cpp
   // Allocate or reuse memory from pool
   pool->allocate(size_A, size_B, ..., device);

   TA* A = pool->A;
   TB* B = pool->B;
   ...

   blas::Queue& queue = *(pool->queue);
   TA* dA = pool->dA;
   ...
   ```

4. **Remove Manual Deallocation** (replace with comment):
   ```cpp
   // Memory is managed by the pool and will be reused or freed automatically
   // No explicit deletion needed here
   ```

## Verification

After applying the memory pool pattern:
```bash
cd /home/morita/workspace/blaspp
make -j4
```

All modified files should compile without errors.
