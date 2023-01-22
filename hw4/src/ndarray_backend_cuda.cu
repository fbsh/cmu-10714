#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdarg.h>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define BLOCK_SIZE 16

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};


struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t m, size_t p) {
  /**
   * Utility function to get cuda dimensions for 2D call
   * output is the m X p matrix
   * block size is 16x16
   * grid size is ceil(m/16) x ceil(p/16)
   * 
   * Args:
   * m: number of rows
   * p: number of columns
   *
   */
  CudaDims dim;
  size_t num_blocks_x = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t num_blocks_y = (p + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim.block = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim.grid = dim3(num_blocks_x, num_blocks_y, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t calculate_offset(CudaVec shape, CudaVec strides, size_t offset, size_t count) {
  /**
   * Calculate the offset into a contiguous array given the shape, strides, count, and offset
   * into the array.
   */
  size_t res = 0;
  for (int i = 0; i < shape.size; i++) {
    size_t idx = 1;
    for (int j = i+1; j < shape.size; j++) {
      idx *= shape.data[j];
    }
    if (i < shape.size - 1 || shape.size == 1) {
      size_t old_idx = idx;
      idx = count / idx;
      count = count - old_idx * idx;
    } else {
      idx = count % shape.data[shape.size-1];
    }
    res += strides.data[i] * idx;
  }
  return res + offset;
}



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < size; i += stride) {
    size_t idx = calculate_offset(shape, strides, offset, i);
    out[i] = a[idx];
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the ewise setitem opeation. Set items in a (non-compact) array
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of a array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < size; i += stride) {
    size_t idx = calculate_offset(shape, strides, offset, i);
    out[idx] = a[i];
  }
  /// END YOUR SOLUTION
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the scalar setitem opeation. Set items in a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: value to set
   *   out: CUDA point to out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < size; i += stride) {
    size_t idx = calculate_offset(shape, strides, offset, i);
    out[idx] = val;
  }
  /// END YOUR SOLUTION
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
#define EWISE_OP(name, op) \
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op b[gid]; \
} \
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define EWISE_UFUNC(name, func) \
__global__ void Ewise##name##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = func(a[gid]); \
} \
void Ewise##name(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

#define EWISE_BFUNC(name, func) \
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = func(a[gid], b[gid]); \
} \
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define SCALAR_OP(name, op) \
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op val; \
} \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define SCALAR_BFUNC(name, func) \
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = func(a[gid], val); \
} \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

EWISE_OP(Mul, *);
EWISE_OP(Div, /);
EWISE_OP(Eq, ==);
EWISE_OP(Ge, >=);
EWISE_BFUNC(Maximum, [](scalar_t a, scalar_t b) { return (a > b) ? a : b; });
EWISE_UFUNC(Log, log);
EWISE_UFUNC(Exp, exp);
EWISE_UFUNC(Tanh, tanh);
SCALAR_OP(Mul, *);
SCALAR_OP(Div, /);
SCALAR_OP(Eq, ==);
SCALAR_OP(Ge, >=);
SCALAR_BFUNC(Power, pow);
SCALAR_BFUNC(Maximum, [](scalar_t a, scalar_t b) { return (a > b) ? a : b; });
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__device__ scalar_t load_tile_a(const scalar_t* a, size_t row_dim, size_t col_dim, size_t i) {
  size_t which_row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t which_col = i * blockDim.y + threadIdx.y;
  if (which_row < row_dim && which_col < col_dim) {
    return a[which_row * col_dim + which_col];
  } else {
    return 0.0F;
  }
}

__device__ scalar_t load_tile_b(const scalar_t* b, size_t row_dim, size_t col_dim, size_t i) {
  size_t which_row = i * blockDim.x + threadIdx.x;
  size_t which_col = blockIdx.y * blockDim.y + threadIdx.y;
  if (which_row < row_dim && which_col < col_dim) {
    return b[which_row * col_dim + which_col];
  } else {
    return 0.0F;
  }
}

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t m, size_t n, size_t p) {
  /**
   * Matrix multiplication kernel. This kernel use shared memory tiling to improve
   * performance.
   *  - a: m x n matrix
   *  - b: n x p matrix
   *  - out: m x p matrix
   */
  size_t out_col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t out_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_t out_res = 0.0f;

  __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

  size_t nIter = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for(int i = 0; i < nIter; i++)
  {
      // load data from global memory to shared memory
      tileA[threadIdx.x][threadIdx.y] = load_tile_a(a, m, n, i);
      tileB[threadIdx.y][threadIdx.x] = load_tile_b(b, n, p, i);

      // sync to wait for all threads in one block to finish loading datas
      __syncthreads();

      // sub-matrix multiply
      for (size_t l = 0; l < BLOCK_SIZE; l++)
      {
          out_res += tileA[threadIdx.x][l] * tileB[threadIdx.y][l];
      }

      // sync to wait for all threads in one block to finish compute
      __syncthreads();
  }

  // store results into global memory
  if (out_row_idx < m && out_col_idx < p)
  {
      out[out_row_idx * p + out_col_idx] = out_res;
  }

}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaTwoDim(M, P);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);

  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  /**
   * Perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: input array
   *   out: output array
   *   size: size of input array
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t group = size / reduce_size;
  if (gid < group) {
    size_t base = gid * reduce_size;
    scalar_t max = a[base];
    for (size_t i = 1; i < reduce_size; i++) {
      max = max > a[base + i] ? max : a[base + i];
    }
    out[gid] = max;
  }
  /// END YOUR SOLUTION
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, reduce_size);
  /// END YOUR SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  /**
   * Perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: input array
   *   out: output array
   *   size: size of input array
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t group = size / reduce_size;
  if (gid < group) {
    size_t base = gid * reduce_size;
    scalar_t sum = 0;
    for (size_t i = 0; i < reduce_size; i++) {
      sum += a[base + i];
    }
    out[gid] = sum;
  }
  /// END YOUR SOLUTION
}


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, reduce_size);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Matrix
////////////////////////////////////////////////////////////////////////////////

/**
 * A triple that represents an entry of sparse matrix.
 * This is used to represent the sparse matrix in COO format.
 */
struct CudaSparseEntry {
  int row;
  int col;
  scalar_t val;
  CudaSparseEntry(int row, int col, scalar_t val) : row(row), col(col), val(val) {}
};

/**
 * A structure that represents a sparse matrix in COO format.
 */
struct CudaSparseMatrix{
  CudaSparseMatrix(int32_t m, int32_t n, int32_t nnz) : m(m), n(n), nnz(nnz) {
    cudaError_t err = cudaMalloc((void**)&data, nnz * sizeof(CudaSparseEntry));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  }

  ~CudaSparseMatrix() { cudaFree(data); }

  CudaSparseEntry* data;
  int32_t m;
  int32_t n;
  int32_t nnz;
};

__global__ void ToSparseKernel(scalar_t* a, CudaSparseEntry* out, int32_t m, int32_t n, int32_t nnz) {
  // size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t cur = 0;
  for (int32_t i = 0; i < m; i++) {
    for (int32_t j = 0; j < n; j++) {
      if (a[i * n + j]  - 0.0F > FLT_EPSILON) {
        out[cur].row = i;
        out[cur].col = j;
        out[cur].val = a[i * n + j];
        cur++;
      }
    }
  }
  // // print sparse matrix
  // std::cout << "sparse matrix:" << std::endl;
  // for (int32_t i = 0; i < nnz; i++) {
  //   std::cout << out[i].row << " " << out[i].col << " " << out[i].val << std::endl;
  // }
}

void ToSparse(const CudaArray& a, CudaSparseMatrix* out, int32_t m, int32_t n, int32_t nnz) {
  /**
   * Convert a dense matrix to a sparse matrix in COO format.
   * The input a is a dense matrix, which has the size of (m, n).
   */
  CudaDims dim = CudaOneDim(1);
  ToSparseKernel<<<dim.grid, dim.block>>>(a.ptr, out->data, m, n, nnz);
}

__global__ void ToDenseKernel(CudaSparseEntry* a, scalar_t* out, int32_t n, int32_t nnz) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nnz) {
    out[a[gid].row * n + a[gid].col] = a[gid].val;
  }
}

void ToDense(const CudaSparseMatrix& a, CudaArray* out) {
  /**
   * Convert a sparse matrix in COO format to a dense matrix.
   * The output out is a dense matrix, which has the size of (m, n).
   */
  Fill(out, 0.0f);
  CudaDims dim = CudaOneDim(a.nnz);
  ToDenseKernel<<<dim.grid, dim.block>>>(a.data, out->ptr, a.n, a.nnz);
}

__global__ void SparseTransposeKernel(CudaSparseEntry* a, CudaSparseEntry* out, int32_t nnz) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nnz) {
    out[gid].row = a[gid].col;
    out[gid].col = a[gid].row;
    out[gid].val = a[gid].val;
  }
}

void SparseTranspose(const CudaSparseMatrix& a, CudaSparseMatrix* out) {
  /**
   * Convert a sparse matrix in COO format to a dense matrix.
   * The output out is a dense matrix, which has the size of (m, n).
   */
  CudaDims dim = CudaOneDim(a.nnz);
  SparseTransposeKernel<<<dim.grid, dim.block>>>(a.data, out->data, a.nnz);
}

__global__ void SparseDenseMatMulKernel(const CudaSparseEntry* a,
                                        const scalar_t* b,
                                        scalar_t* out, 
                                        int32_t m, int32_t n, int32_t p, 
                                        int32_t nnz_a) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nnz_a) {
    int32_t row = a[gid].row;
    int32_t col = a[gid].col;
    scalar_t val = a[gid].val;
    for (int32_t i = 0; i < p; i++) {
      atomicAdd(&out[row * p + i], val * b[col * p + i]);
    }
  }
  
}

void SparseDenseMatMul(const CudaSparseMatrix& a, const CudaArray& b, CudaArray* out, int32_t m, int32_t n, int32_t p) {
  /**
   * Multiply two sparse matrices in COO format.
   * The output out is a dense matrix, which has the size of (m, p).
   * The input a is a sparse matrix in COO format, which has the size of (m, n).
   * The input b is a dense matrix, which has the size of (n, p).
   */
  Fill(out, 0.0f);
  CudaDims dim = CudaOneDim(a.nnz);
  SparseDenseMatMulKernel<<<dim.grid, dim.block>>>(a.data, b.ptr, out->ptr, m, n, p, a.nnz);
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Matrix End
////////////////////////////////////////////////////////////////////////////////

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // TODO: add SparseMatrix Class
  py::class_<CudaSparseMatrix>(m, "SparseMatrix")
      .def(py::init<int32_t, int32_t, int32_t>(), py::return_value_policy::take_ownership)
      .def_readonly("m", &CudaSparseMatrix::m)
      .def_readonly("n", &CudaSparseMatrix::n)
      .def_readonly("nnz", &CudaSparseMatrix::nnz);
  
  // TODO: add ToDense() and ToSparse() functions
  m.def("to_dense", ToDense);
  m.def("to_sparse", ToSparse);
  m.def("sparse_transpose", SparseTranspose);
  m.def("spmm", SparseDenseMatMul);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
