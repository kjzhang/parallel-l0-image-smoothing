# Import Libraries
import numpy as np
import cv2
import argparse
import time

# Import User Libraries
import L0_helpers

# Image File Path
image_r = "images/flowers.jpg"
image_w = "out_serial.png"

# L0 minimization parameters
kappa = 2.0;
_lambda = 2e-2;

# Verbose output
verbose = False;

if __name__ == '__main__':
  # Parse arguments
  parser = argparse.ArgumentParser(
      description="Serial implementation of image smoothing via L0 gradient minimization")
  parser.add_argument('image_r', help="input image file")
  parser.add_argument('image_w', help="output image file")
  parser.add_argument('-k', type=float, default=2.0,
      metavar='kappa', help='updating weight (default 2.0)')
  parser.add_argument('-l', type=float, default=2e-2,
      metavar='lambda', help='smoothing weight (default 2e-2)')
  parser.add_argument('-v', '--verbose', action='store_true',
      help='enable verbose logging for each iteration')
  args = parser.parse_args()

  # Set parameters
  kappa = args.k
  _lambda = args.l

  image_r = args.image_r
  image_w = args.image_w

  verbose = args.verbose

  # Read image I
  image = cv2.imread(image_r)

  # Timers
  step_1 = 0.0
  step_2 = 0.0
  step_2_fft = 0.0

  # Start time
  start_time = time.time()

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
  iteration = 0

  # Done initializing  
  init_time = time.time()

  # Iterate until desired convergence in similarity
  while beta < beta_max:

    if verbose:
      print "ITERATION %i" % iteration

    ### Step 1: estimate (h, v) subproblem

    # subproblem 1 start time
    s_time = time.time()

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

    # subproblem 1 end time
    e_time = time.time()
    step_1 = step_1 + e_time - s_time
    if verbose:
      print "-subproblem 1: estimate (h,v)"
      print "--time: %f (s)" % (e_time - s_time)

    ### Step 2: estimate S subproblem

    # subproblem 2 start time
    s_time = time.time()

    # compute dxhp + dyvp
    dxhp[:,0:1,:] = h[:,M-1:M,:] - h[:,0:1,:]
    dxhp[:,1:M,:] = -(np.diff(h, 1, 1))
    dyvp[0:1,:,:] = v[N-1:N,:,:] - v[0:1,:,:]
    dyvp[1:N,:,:] = -(np.diff(v, 1, 0))
    normin = dxhp + dyvp

    fft_s = time.time()
    FS[:,:,0] = np.fft.fft2(normin[:,:,0])
    FS[:,:,1] = np.fft.fft2(normin[:,:,1])
    FS[:,:,2] = np.fft.fft2(normin[:,:,2])
    fft_e = time.time()
    step_2_fft += fft_e - fft_s

    # solve for S + 1 in Fourier domain
    denorm = 1 + beta * MTF;
    FS[:,:,:] = (FI + beta * FS) / denorm

    # inverse FFT to compute S + 1
    fft_s = time.time()
    S[:,:,0] = np.float32((np.fft.ifft2(FS[:,:,0])).real)
    S[:,:,1] = np.float32((np.fft.ifft2(FS[:,:,1])).real)
    S[:,:,2] = np.float32((np.fft.ifft2(FS[:,:,2])).real)
    fft_e = time.time()
    step_2_fft += fft_e - fft_s

    # subproblem 2 end time
    e_time = time.time()
    step_2 =  step_2 + e_time - s_time
    if verbose:
      print "-subproblem 2: estimate S + 1"
      print "--time: %f (s)" % (e_time - s_time)
      print ""

    # update beta for next iteration
    beta *= kappa
    iteration += 1

  # Rescale image
  S = S * 256

  # Total end time
  final_time = time.time()

  print "Total Time: %f (s)" % (final_time - start_time)
  print "Setup: %f (s)" % (init_time - start_time)
  print "Step 1: %f (s)" % (step_1)
  print "Step 2: %f (s)" % (step_2)
  print "Step 2 (FFT): %f (s)" % (step_2_fft)
  print "Iterations: %d" % (iteration)

  cv2.imwrite(image_w, S)
