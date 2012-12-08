# Import Libraries
import numpy as np
import cv2
import argparse
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

mtf_kernel_source = \
"""
#include <cuComplex.h>

__global__ void mtf_kernel(cuFloatComplex* z, cuFloatComplex* a, cuFloatComplex* b, int Nx, int Ny)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int index = x + y * Nx;

  if (x < Nx && y < Ny) {
    cuFloatComplex pa = a[index];
    cuFloatComplex pb = b[index];

    z[index] = make_cuFloatComplex(pa.x * pa.x + pa.y * pa.y +
                                   pb.x * pb.x + pb.y * pb.y,
                                   0);
  }
}
"""

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
image_r = "images/flowers.jpg"
image_w = "out_serial.png"

# L0 minimization parameters
kappa = 2.0;
_lambda = 2e-2;

if __name__ == '__main__':
  # Parse arguments
  parser = argparse.ArgumentParser(
      description="Parallel implementation of image smoothing via L0 gradient minimization")
  parser.add_argument('image_r', help="input image file")
  parser.add_argument('image_w', help="output image file")
  parser.add_argument('-k', type=float, default=2.0,
      metavar='kappa', help='updating weight (default 2.0)')
  parser.add_argument('-l', type=float, default=2e-2,
      metavar='lambda', help='smoothing weight (default 2e-2)')
  args = parser.parse_args()

  # Set parameters
  kappa = args.k
  _lambda = args.l

  image_r = args.image_r
  image_w = args.image_w

  # Read image I
  image = cv2.imread(image_r)

  # Timers
  step_1 = 0.0
  step_2 = 0.0
  step_2_fft = 0.0

  # Start time
  start_time = time.time()

  # Validate image format
  N, M, D = image.shape
  assert D == 3, "Error: input must be 3-channel RGB image"
  print "Processing %d x %d RGB image" % (M, N)

  ### Compile and initialize CUDA kernels and FFT plans
  mtf_kernel = cuda_compile(mtf_kernel_source, "mtf_kernel")
  hv_kernel = cuda_compile(hv_kernel_source, "hv_kernel")
  Sa_kernel = cuda_compile(Sa_kernel_source, "Sa_kernel")
  Sb_kernel = cuda_compile(Sb_kernel_source, "Sb_kernel")
  d_kernel = cuda_compile(d_kernel_source, "d_kernel")
  merge_r_kernel = cuda_compile(merge_r_kernel_source, "merge_r_kernel")
  plan = cu_fft.Plan((N, M), np.complex64, np.complex64)

  ### CUDA kernel settings
  Nx, Ny = np.int32(M), np.int32(N)
  x_tpb = 32
  y_tpb = 16
  x_blocks = int(np.ceil(Nx * 1.0/x_tpb))
  y_blocks = int(np.ceil(Ny * 1.0/y_tpb))
  blocksize = (x_tpb, y_tpb, 1)
  gridsize  = (x_blocks, y_blocks)

  # Initialize S with I and normalize RGB values
  S = np.float32(image) / 256

  ### Allocate memory on GPU

  # Channel dimensions
  c1 = (N, M)
  c3 = (N, M, D)

  # I, S images
  S_d = gpu.to_gpu(S)                        # 3-channel real S
  FS_d = gpu.GPUArray(c3, np.complex64)      # 3-channel complex FS

  FIR_d = gpu.GPUArray(c1, np.complex64)     # 1-channel complex FI
  FIG_d = gpu.GPUArray(c1, np.complex64)
  FIB_d = gpu.GPUArray(c1, np.complex64)

  MTF_d = gpu.GPUArray(c1, np.complex64)     # 1-channel complex MTF

  # h, v estimators for dxSp, dySp
  h_d = gpu.GPUArray(c3, np.float32)         # 3-channel real hp
  v_d = gpu.GPUArray(c3, np.float32)         # 3-channel real vp

  # normalizing denominator
  d_d = gpu.GPUArray(c1, np.complex64)       # 1-channel complex d

  # input / output for CUDA FFT
  FFToR_d = gpu.GPUArray(c1, np.complex64)   # 1-channel complex FFT
  FFToG_d = gpu.GPUArray(c1, np.complex64)
  FFToB_d = gpu.GPUArray(c1, np.complex64)

  # initially load starting image I for CUDA FFT
  FFTiR_d = gpu.to_gpu(np.array(S[:,:,0], dtype=np.complex64))
  FFTiG_d = gpu.to_gpu(np.array(S[:,:,1], dtype=np.complex64))
  FFTiB_d = gpu.to_gpu(np.array(S[:,:,2], dtype=np.complex64))

  ### Initial Computation

  # Compute image OTF
  size_2D = [N, M]
  fx = np.int32([[1, -1]])
  fy = np.int32([[1], [-1]])
  otfFx = L0_helpers.psf2otf(fx, size_2D)
  otfFy = L0_helpers.psf2otf(fy, size_2D)

  # Compute MTF
  otfFx_d = gpu.to_gpu(otfFx)
  otfFy_d = gpu.to_gpu(otfFy)
  mtf_kernel(MTF_d, otfFx_d, otfFy_d, Nx, Ny, block=blocksize, grid=gridsize)

  # Compute Fourier transform of original image 
  cu_fft.fft(FFTiR_d, FIR_d, plan)
  cu_fft.fft(FFTiG_d, FIG_d, plan)
  cu_fft.fft(FFTiB_d, FIB_d, plan)

  ### Iteration settings
  beta_max = 1e5
  beta = 2 * _lambda
  iteration = 0

  # Done initializing  
  init_time = time.time()

  ### Iterate until desired convergence in similarity
  while beta < beta_max:

    print "ITERATION %i" % iteration

    ### Step 1: estimate (h, v) subproblem

    # subproblem 1 start time
    s_time = time.time()

    # compute piecewise solution for hp, vp
    threshold = np.float32(_lambda / beta)
    hv_kernel(h_d, v_d, S_d, Nx, Ny, threshold, block=blocksize, grid=gridsize)

    # subproblem 1 end time
    e_time = time.time()
    step_1 = step_1 + e_time - s_time
    print "-subproblem 1: estimate (h,v)"
    print "--time: %f (s)" % (e_time - s_time)

    ### Step 2: estimate S subproblem

    # subproblem 2 start time
    s_time = time.time()

    # find S delta in original domain
    Sa_kernel(FFTiR_d, FFTiG_d, FFTiB_d, h_d, v_d, Nx, Ny, block=blocksize, grid=gridsize)

    # find S delta in Fourier domain in each color channel
    fft_s = time.time()
    cu_fft.fft(FFTiR_d, FFToR_d, plan)
    cu_fft.fft(FFTiG_d, FFToG_d, plan)
    cu_fft.fft(FFTiB_d, FFToB_d, plan)
    fft_e = time.time()
    step_2_fft += fft_e - fft_s

    # solve for normalizing denominator
    d_kernel(d_d, MTF_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)

    # solve for S + 1 in Fourier domain in each color channel
    Sb_kernel(FFTiR_d, FIR_d, FFToR_d, d_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)
    Sb_kernel(FFTiG_d, FIG_d, FFToG_d, d_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)
    Sb_kernel(FFTiB_d, FIB_d, FFToB_d, d_d, np.float32(beta), Nx, Ny, block=blocksize, grid=gridsize)

    # inverse FFT to compute S + 1 in each color channel
    fft_s = time.time()
    cu_fft.ifft(FFTiR_d, FFToR_d, plan, scale=True)
    cu_fft.ifft(FFTiG_d, FFToG_d, plan, scale=True)
    cu_fft.ifft(FFTiB_d, FFToB_d, plan, scale=True)
    fft_e = time.time()
    step_2_fft += fft_e - fft_s

    # merge real components of 3 complex color channels
    merge_r_kernel(S_d, FFToR_d, FFToG_d, FFToB_d, Nx, Ny, block=blocksize, grid=gridsize)

    # subproblem 2 end time
    e_time = time.time()
    step_2 =  step_2 + e_time - s_time
    print "-subproblem 2: estimate S + 1"
    print "--time: %f (s)" % (e_time - s_time)
    print ""

    # update beta for next iteration
    beta *= kappa
    iteration += 1

  # Clean up handle to FFT plan
  del plan

  # Rescale final image output
  final = S_d.get() * 256

  # Total end time
  final_time = time.time()

  print "Total Time: %f (s)" % (final_time - start_time)
  print "Setup: %f (s)" % (init_time - start_time)
  print "Step 1: %f (s)" % (step_1)
  print "Step 2: %f (s)" % (step_2)
  print "Step 2 (FFT): %f (s)" % (step_2_fft)
  print "Iterations: %d" % (iteration)

  # Write image output to file
  cv2.imwrite(image_w, final)

