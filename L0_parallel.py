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
#define SUM_RGB(pixel) (pixel.x + pixel.y + pixel.z)
#define SOS_RGB(pixel) (pixel.x * pixel.x + pixel.y * pixel.y + pixel.z * pixel.z)

#define ADD_PIXELS(a, b) (make_float3(a.x + b.x, a.y + b.y, a.z + b.z))
#define SUB_PIXELS(a, b) (make_float3(a.x - b.x, a.y - b.y, a.z - b.z))

__global__ void Sa_kernel(float* R, float* G, float* B, float3* h, float3* v, int Nx, int Ny)
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
    R[x + y * Nx] = sum.x;
    G[x + y * Nx] = sum.y;
    B[x + y * Nx] = sum.z;
  }
}
"""

# Image File Path
image_file = "flower.jpg"

# L0 minimization parameters
kappa = 2.0;
_lambda = 2e-2;

if __name__ == '__main__':
  ### Initialize CUDA kernels
  hv_kernel = cuda_compile(hv_kernel_source, "hv_kernel")
  Sa_kernel = cuda_compile(Sa_kernel_source, "Sa_kernel")

  # Read image I
  image = cv2.imread(image_file)

  # Validate image format
  N, M, D = np.int32(image.shape)
  assert D == 3, "Error: input must be 3-channel RGB image"
  print "Processing %d x %d RGB image" % (M, N)

  # Initialize S as I
  S = np.float32(image) / 256

  # Compute image OTF
  size_2D = [N, M]
  fx = np.int32([[1, -1]])
  fy = np.int32([[1], [-1]])
  otfFx = L0_helpers.psf2otf(fx, size_2D)
  otfFy = L0_helpers.psf2otf(fy, size_2D)

  # Compute F(I)
  FI = np.complex64(np.zeros((N, M, D)))
  FI[:,:,0] = np.fft.fft2(S[:,:,0])
  FI[:,:,1] = np.fft.fft2(S[:,:,1])
  FI[:,:,2] = np.fft.fft2(S[:,:,2])

  # Compute MTF
  MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
  MTF = np.tile(MTF[:, :, np.newaxis], (1, 1, D))

  # Initialize buffers
  h = np.float32(np.zeros((N, M, D)))
  v = np.float32(np.zeros((N, M, D)))

  dxhp = np.float32(np.zeros((N, M, D)))
  dyvp = np.float32(np.zeros((N, M, D)))
  FS = np.complex64(np.zeros((N, M, D)))

  r_channel = np.float32(np.zeros((N, M)))
  c_channel = np.complex64(np.zeros((N, M)))

  # Allocate memory on disk
  S_d = gpu.to_gpu(S)               # 3-channel Sp

  h_d = gpu.to_gpu(h)               # 3-channel hp
  v_d = gpu.to_gpu(v)               # 3-channel vp

  RR_d = gpu.to_gpu(r_channel)      # 1-channel real
  RG_d = gpu.to_gpu(r_channel)
  RB_d = gpu.to_gpu(r_channel)

  CR_d = gpu.to_gpu(c_channel)      # 1-channel complex
  CG_d = gpu.to_gpu(c_channel)
  CB_d = gpu.to_gpu(c_channel)

  # Iteration settings
  beta_max = 1e5;
  beta = 2 * _lambda

  # Iterate until desired convergence in similarity
  while beta < beta_max:

    S_d = gpu.to_gpu(S)

    ### Step 1: estimate (h, v) subproblem

    # kernel block and grid size
    Nx, Ny = M, N
    x_tpb = 32
    y_tpb = 8
    x_blocks = int(np.ceil(Nx * 1.0/x_tpb))
    y_blocks = int(np.ceil(Ny * 1.0/y_tpb))
    blocksize = (x_tpb, y_tpb, 1)
    gridsize  = (x_blocks, y_blocks)

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
    Sa_kernel(RR_d, RG_d, RB_d, h_d, v_d, Nx, Ny, block=blocksize, grid=gridsize)

    FS[:,:,0] = np.fft.fft2(RR_d.get())
    FS[:,:,1] = np.fft.fft2(RG_d.get())
    FS[:,:,2] = np.fft.fft2(RB_d.get())

    # solve for S + 1 in Fourier domain
    denorm = 1 + beta * MTF;
    FS[:,:,:] = (FI + beta * FS) / denorm

    # inverse FFT to compute S + 1
    S[:,:,0] = np.float32((np.fft.ifft2(FS[:,:,0])).real)
    S[:,:,1] = np.float32((np.fft.ifft2(FS[:,:,1])).real)
    S[:,:,2] = np.float32((np.fft.ifft2(FS[:,:,2])).real)

    beta *= kappa

    print "."

  cv2.imwrite("out_parallel.png", S * 256)

