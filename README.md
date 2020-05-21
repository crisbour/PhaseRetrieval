# Phase Retrieval Algorithm

Ultra fast computation of Gerchberg-Saxton algorithm and other similar phase retrieval algorithms, using CUDA. This repository contains a Python implementation along to demonstrate speed-up of the CUDA one.
![](GS_flowchart.png | width = 200)
Note that by far the most computational expensive tasks are the FFT and IFFT blocks for which Numpy's library is already optimised, therefore the time taken by the Python code should be on the same order of magnitude as an equivalent C/C++ sequential implementation.
### Features:
1. Image
- Image Creation:
  - Illumination pattern
  - Desired pattern
- Image show in COLORMAP of phase,illumination pattern, desired pattern and actual output.
2. Solver
- Mathematical operation blocks using CUDA:
  - Compose: Create complex signal from amplitude and phase
  - Decompose: Find amplitude and phase of a complex signal
  - SLM_To_Obj: FFT => Projection in image plane
  - Obj_To_Obj: IFFT => Projection from image plane to SLM plane (exit pupils plane multiplied by constant phase term)
  - Normalize: <a href="https://www.codecogs.com/eqnedit.php?latex=u_{norm}=(u-u_{min})/(u_{max}-u_{min})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{norm}=(u-u_{min})/(u_{max}-u_{min})" title="u_{norm}=(u-u_{min})/(u_{max}-u_{min})" /></a>
- Call Algorithm to iterate
3. Algorithm
- Factory method to choose an algorithm to solve the problem. Implemented algorithms:
  - Gerchberg Saxton
  - Weighted GS
  - MRAF
  - Uniformity Controlled MRAF
