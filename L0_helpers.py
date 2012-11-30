import numpy as np
import scipy as sp
import math

def psf2otf(psf, outSize=None):
  psf = np.float32(psf)

  # Determine PSF / OTF shapes
  psfSize = np.int32(psf.shape)
  if not outSize:
    outSize = psfSize
  outSize = np.int32(outSize)

  # Pad the psf to output
  new_psf = np.float32(np.zeros(outSize))
  new_psf[:psfSize[0],:psfSize[1]] = psf[:,:]
  psf = new_psf

  # Circularly shift the otf so that PSF center is at (0,0)
  shift = -(psfSize / 2)
  psf = circshift(psf, shift)

  # Compute the otf
  otf = np.fft.fftn(psf)

  return np.complex64(otf)

def circshift(A, shift):
  for i in xrange(shift.size):
    A = np.roll(A, shift[i], axis=i)
  return A
