# Parallel Image Smoothing via L0 Gradient Minimization#
## CS 205 Fall 2012 ##

### Project Members: ###
* Kevin Zhang : kzhang@college.harvard.edu
* Han He : han.he@college.harvard.edu

Our final project is created based on the image smoothing method described in the following paper: "Image Smoothing via L0 Gradient Minimization", Li Xu, Cewu Lu, Yi Xu, Jiayi Jia, ACM Transcations on Graphics, (SIGGRAPH Asia 2011), 2011.

### Dependencies: ###
* numpy 1.6.2
* pycuda 2012.1
* scikits.cuda 0.041
* cv2 2.4.3-rc

The above packages should already be installed on the resonance.seas GPU cluster for CS 205.

On a resonance node, `module load courses/cs205/2012` will explicitly load the required modules.

Alternatively, `virtualenv` from the `packages/epd/7.3-1` module an be used to create new virtual environment on resonance.seas, although numpy, pycuda, and scikits.cuda must be manually installed and configured to run the project code.

### Usage: ###
Run `L0_serial.py` or `L0_parallel.py` with the `-h` or `--help` option to print a detailed help message.

    -bash-3.2$ python L0_parallel.py -h
    usage: L0_parallel.py [-h] [-k kappa] [-l lambda] [-v] image_r image_w
    
    Parallel implementation of image smoothing via L0 gradient minimization
    
    positional arguments:
      image_r        input image file
      image_w        output image file
    
    optional arguments:
      -h, --help     show this help message and exit
      -k kappa       updating weight (default 2.0)
      -l lambda      smoothing weight (default 2e-2)
      -v, --verbose  enable verbose logging for each iteration

For example, to run the code with default smoothing parameters:

    python L0_parallel.py images/flower.jpg output.png

Or with custom smoothing parameters:

    python L0_parallel.py images/flower.jpg output.png -k 1.05 -l 0.015

### Other: ###
Sample Image: [Bunch of Flowers](http://andrewfletcher.deviantart.com/art/Bunch-of-Flowers-294042509)
