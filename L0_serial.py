# Import Libraries
import numpy as np
import cv2
import time

# Import User Libraries
import L0_helpers

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

    # compute dxSp
    h[:,0:M-1,:] = np.diff(S, 1, 1)
    h[:,M-1:M,:] = S[:,0:1,:] - S[:,M-1:M,:]

    # compute dySp
    v[0:N-1,:,:] = np.diff(S, 1, 0)
    v[N-1:N,:,:] = S[0:1,:,:] - S[N-1:N,:,:]

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

