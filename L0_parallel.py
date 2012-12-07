# Import Libraries
import numpy as np
import cv2
import time

# Import User Libraries
import L0_helpers

# Import PyCUDA and initialize device
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.autoinit

# Import scikits.cuda for CUDA FFT capabilities
import scikits.cuda.fft as cu_fft

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

# Kernel to solve for hp, vp
hv_kernel_source = \
"""
#define SOS_RGB(pixel) (pixel.x * pixel.x + pixel.y * pixel.y + pixel.z * pixel.z)
#define SUB_PIXELS(a, b) (make_float3(a.x - b.x, a.y - b.y, a.z - b.z))

__global__ void hv_kernel(float3* h, float3* v, float3* S, int Nx, int Ny, float threshold)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < Nx && y < Ny) {
    // find relevant pixels
    float3 pc = S[x + y * Nx];
    float3 px = S[((x + 1) % Nx) + y * Nx];
    float3 py = S[x + ((y + 1) % Ny) * Nx];

    // compute dxSp and dySp
    float3 dx = SUB_PIXELS(px, pc);
    float3 dy = SUB_PIXELS(py, pc);

    // compute minimum energy E = dxSp^2 + dySp^2
    float delta = SOS_RGB(dx) + SOS_RGB(dy);

    // compute piecewise solution for h,v: find where E <= _lambda/beta
    h[x + y * Nx] = delta > threshold ? dx : make_float3(0.0, 0.0, 0.0);
    v[x + y * Nx] = delta > threshold ? dy : make_float3(0.0, 0.0, 0.0);
  }
}
"""

Sa_kernel_source = \
"""
#include <cuComplex.h>

#define ADD_PIXELS(a, b) (make_float3(a.x + b.x, a.y + b.y, a.z + b.z))
#define SUB_PIXELS(a, b) (make_float3(a.x - b.x, a.y - b.y, a.z - b.z))

__global__ void Sa_kernel(cuFloatComplex* R, cuFloatComplex* G, cuFloatComplex* B, float3* h, float3* v, int Nx, int Ny)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < Nx && y < Ny) {
    // find relevant pixels
    float3 ph = h[x + y * Nx];
    float3 dhx = h[((Nx + x - 1) % Nx) + y * Nx];
    float3 pv = v[x + y * Nx];
    float3 dvy = v[x + ((Ny + y - 1) % Ny) * Nx];

    // compute dxhp and dyvp
    float3 dx = SUB_PIXELS(dhx, ph);
    float3 dy = SUB_PIXELS(dvy, pv);

    // compute sum and split into channels
    float3 sum = ADD_PIXELS(dx, dy);
    R[x + y * Nx] = make_cuFloatComplex(sum.x, 0);
    G[x + y * Nx] = make_cuFloatComplex(sum.y, 0);
    B[x + y * Nx] = make_cuFloatComplex(sum.z, 0);
  }
}
"""

Sb_kernel_source = \
"""
#include <cuComplex.h>

__global__ void Sb_kernel(cuFloatComplex* FS, cuFloatComplex* FI, cuFloatComplex* FD, cuFloatComplex* d, float beta, int Nx, int Ny)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < Nx && y < Ny) {
    FS[x + y * Nx] = cuCdivf(cuCaddf(FI[x + y * Nx],
                                     cuCmulf(make_cuFloatComplex(beta, 0.0),
                                             FD[x + y * Nx])),
                             d[x + y * Nx]);
  }
}
"""

d_kernel_source = \
"""
#include <cuComplex.h>

__global__ void d_kernel(cuFloatComplex* d, cuFloatComplex* m, float beta, int Nx, int Ny)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < Nx && y < Ny) {
    d[x + y * Nx] = cuCaddf(make_cuFloatComplex(1.0, 0.0),
                            cuCmulf(make_cuFloatComplex(beta, 0.0),
                                    m[x + y * Nx]));
  }
}
"""

merge_r_kernel_source = \
"""
#include <cuComplex.h>

__global__ void merge_r_kernel(float3* S, cuFloatComplex* R, cuFloatComplex* G, cuFloatComplex* B, int Nx, int Ny)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < Nx && y < Ny) {
    S[x + y * Nx] = make_float3(cuCrealf(R[x + y * Nx]),
                                cuCrealf(G[x + y * Nx]),
                                cuCrealf(B[x + y * Nx]));
  }
}
"""

# Image File Path
image_file = "images/flowers_5.jpg"

# L0 minimization parameters
kappa = 2.0;
_lambda = 2e-2;

if __name__ == '__main__':
  # Read image I
  image = cv2.imread(image_file)

  # Total time start
  s_total = time.time()

  # Validate image format
  N, M, D = image.shape
  assert D == 3, "Error: input must be 3-channel RGB image"
  print "Processing %d x %d RGB image" % (M, N)

  ### Compile and initialize CUDA kernels and FFT plans
  hv_kernel = cuda_compile(hv_kernel_source, "hv_kernel")
  Sa_kernel = cuda_compile(Sa_kernel_source, "Sa_kernel")
  Sb_kernel = cuda_compile(Sb_kernel_source, "Sb_kernel")
  d_kernel = cuda_compile(d_kernel_source, "d_kernel")
  merge_r_kernel = cuda_compile(merge_r_kernel_source, "merge_r_kernel")
  plan = cu_fft.Plan((N, M), np.complex64, np.complex64)

  # Initialize S with I and normalize RGB values
  S = np.float32(image) / 256

  # Initialize buffers on host
  channel3r = np.float32(np.zeros((N, M, D)))
  channel1c = np.complex64(np.zeros((N, M)))
  FS = np.complex64(np.zeros((N, M, D)))

  # Compute image OTF
  size_2D = [N, M]
  fx = np.int32([[1, -1]])
  fy = np.int32([[1], [-1]])
  otfFx = L0_helpers.psf2otf(fx, size_2D)
  otfFy = L0_helpers.psf2otf(fy, size_2D)

  # Compute MTF
  MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
  MTF_d = gpu.to_gpu(np.array(MTF, np.complex64))

  ### Allocate memory on GPU

  # I, S images
  FS_d = gpu.to_gpu(FS)             # 3-channel complex FS
  S_d = gpu.to_gpu(S)               # 3-channel real Sp

  FIR_d = gpu.to_gpu(channel1c)     # 1-channel complex FI
  FIG_d = gpu.to_gpu(channel1c)
  FIB_d = gpu.to_gpu(channel1c)

  # h, v estimators for dxSp, dySp
  h_d = gpu.to_gpu(channel3r)       # 3-channel real hp
  v_d = gpu.to_gpu(channel3r)       # 3-channel real vp

  # normalizing denominator
  d_d = gpu.to_gpu(channel1c)       # 1-channel complex d

  # input / output for CUDA FFT
  FFToR_d = gpu.to_gpu(channel1c)   # 1-channel complex FFT
  FFToG_d = gpu.to_gpu(channel1c)
  FFToB_d = gpu.to_gpu(channel1c)

  # initially load starting image I for CUDA FFT
  FFTiR_d = gpu.to_gpu(np.array(S[:,:,0], dtype=np.complex64))
  FFTiG_d = gpu.to_gpu(np.array(S[:,:,1], dtype=np.complex64))
  FFTiB_d = gpu.to_gpu(np.array(S[:,:,2], dtype=np.complex64))

  ### Compute Fourier transform of original image 
  cu_fft.fft(FFTiR_d, FIR_d, plan)
  cu_fft.fft(FFTiG_d, FIG_d, plan)
  cu_fft.fft(FFTiB_d, FIB_d, plan)

  ## CUDA kernel settings
  Nx, Ny = np.int32(M), np.int32(N)
  x_tpb = 32
  y_tpb = 16
  x_blocks = int(np.ceil(Nx * 1.0/x_tpb))
  y_blocks = int(np.ceil(Ny * 1.0/y_tpb))
  blocksize = (x_tpb, y_tpb, 1)
  gridsize  = (x_blocks, y_blocks)

  ### Iteration settings
  beta_max = 1e5
  beta = 2 * _lambda
  iteration = 0

  ### Iterate until desired convergence in similarity
  while beta < beta_max:

    print "ITERATION %i" % iteration

    ### Step 1: estimate (h, v) subproblem

    # subproblem 1 start time
    print "-subproblem 1: estimate (h,v)"
    s_time = time.time()

    # compute piecewise solution for hp, vp
    threshold = np.float32(_lambda / beta)
    hv_kernel(h_d, v_d, S_d, Nx, Ny, threshold, block=blocksize, grid=gridsize)

    # subproblem 1 end time
    e_time = time.time()
    print "--time: %f (s)" % (e_time - s_time)

    ### Step 2: estimate S subproblem

    # subproblem 2 start time
    print "-subproblem 2: estimate S + 1"
    s_time = time.time()

    # find S delta in original domain
    Sa_kernel(FFTiR_d, FFTiG_d, FFTiB_d, h_d, v_d, Nx, Ny, block=blocksize, grid=gridsize)

    # find S delta in Fourier domain in each color channel
    cu_fft.fft(FFTiR_d, FFToR_d, plan)
    cu_fft.fft(FFTiG_d, FFToG_d, plan)
    cu_fft.fft(FFTiB_d, FFToB_d, plan)

    # solve for normalizing denominator
    d_kernel(d_d, MTF_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)

    # solve for S + 1 in Fourier domain in each color channel
    Sb_kernel(FFTiR_d, FIR_d, FFToR_d, d_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)
    Sb_kernel(FFTiG_d, FIG_d, FFToG_d, d_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)
    Sb_kernel(FFTiB_d, FIB_d, FFToB_d, d_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)

    # inverse FFT to compute S + 1 in each color channel
    cu_fft.ifft(FFTiR_d, FFToR_d, plan, scale=True)
    cu_fft.ifft(FFTiG_d, FFToG_d, plan, scale=True)
    cu_fft.ifft(FFTiB_d, FFToB_d, plan, scale=True)

    # merge real components of 3 complex color channels
    merge_r_kernel(S_d, FFToR_d, FFToG_d, FFToB_d, Nx, Ny, block=blocksize, grid=gridsize)

    # subproblem 2 end time
    e_time = time.time()
    print "--time: %f (s)" % (e_time - s_time)

    # update beta for next iteration
    beta *= kappa
    iteration += 1

    print ""

  # Clean up handle to FFT plan
  del plan

  # Rescale final image output
  final = S_d.get() * 256

  # Total end time
  e_total = time.time()
  print "Parallel Time: %f" % (e_total - s_total)

  # Write image output to file
  cv2.imwrite("out_parallel.png", final)

