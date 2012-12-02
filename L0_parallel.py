# Import Libraries
import numpy as np
import cv2
import time

# Import User Libraries
import L0_helpers

# Import PyCUDA
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu

# Initialize the CUDA device
import pycuda.autoinit

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

# Kernel for Step 1
diff_kernel_source = \
"""
__global__ void diff_kernel(int N, int M, int* h, int*v, int* S)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int tid = x + y*M;

  if(x < M-1)
    h[tid] = S[tid+1] - S[tid];
  else
    h[tid] = S[y*M] - S[tid];

  if(y < N-1)
    result[tid] = S[tid+M] - S[tid];
  else
    result[tid] = S[x] - S[tid];
}
"""
  
# Image File Path
image_file = "flower.jpg"

# L0 minimization parameters
kappa = 2.0;
_lambda = 2e-2;

if __name__ == '__main__':
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

  # Iteration settings
  beta_max = 1e5;
  beta = 2 * _lambda

  # Iterate until desired convergence in similarity
  while beta < beta_max:

    ### Step 1: estimate (h, v) subproblem

    # kernels
    diff_kernel = cuda_compile(diff_kernel_source, "diff_kernel")

    # allocate memory
    h_d = gpu.to_gpu(h)
    v_d = gpu.to_gpu(v)
    S_d = gpu.to_gpu(S)

    # block and grid size
    blocksize = (32, 32, 1)
    gridsize  = (int(M*1.0/32+1), int(N*1.0/32+1))

    # compute dxSp and dySp
    diff_h_kernel(np.int32(N), np.int32(M), h_d, v_d, S_d)

    # compute minimum energy E = dxSp^2 + dySp^2 <= _lambda/beta
    t = np.sum(np.power(h, 2) + np.power(v, 2), axis=2) < _lambda / beta
    t = np.tile(t[:, :, np.newaxis], (1, 1, 3))

    # compute piecewise solution for hp, vp
    h[t] = 0
    v[t] = 0


    ### Step 2: estimate S subproblem

    # compute dxhp
    dxhp[:,0:1,:] = h[:,M-1:M,:] - h[:,0:1,:]
    dxhp[:,1:M,:] = -(np.diff(h, 1, 1))

    # compute dyvp
    dyvp[0:1,:,:] = v[N-1:N,:,:] - v[0:1,:,:]
    dyvp[1:N,:,:] = -(np.diff(v, 1, 0))

    normin = dxhp + dyvp
    FS[:,:,0] = np.fft.fft2(normin[:,:,0])
    FS[:,:,1] = np.fft.fft2(normin[:,:,1])
    FS[:,:,2] = np.fft.fft2(normin[:,:,2])

    # solve for S + 1 in Fourier domain
    denorm = 1 + beta * MTF;
    FS[:,:,:] = (FI + beta * FS) / denorm

    # inverse FFT to compute S + 1
    S[:,:,0] = np.float32((np.fft.ifft2(FS[:,:,0])).real)
    S[:,:,1] = np.float32((np.fft.ifft2(FS[:,:,1])).real)
    S[:,:,2] = np.float32((np.fft.ifft2(FS[:,:,2])).real)

    beta *= kappa

    print "."

  cv2.imwrite("out.png", S * 256)
  