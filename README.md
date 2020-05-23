# Phase Retrieval Algorithm
## Ultra fast computation of Gerchberg-Saxton algorithm and other similar phase retrieval algorithms, using CUDA.

Note: This repository contains a Python implementation along to demonstrate speed-up of the CUDA one.
### Algorithm description
Below the core of the algorithm is presented. The problem concerns with producing a irradiance pattern (image) through phase modulation. A constant illumination profile (i.e. laser beam) is provided and we are allowed to modulate its phase by means of a phase modulating device such as a spatial light modulator (SLM).
<img src="https://github.com/cristi-bourceanu/PhaseRetrieval/blob/master/GS_flowchart.png" width="70%">

### Speed-up
Looking at the table below, the CUDA implementation is faster by two orders of magnitude than the Numpy one. Time per cycle neglects memory transfer between the host (CPU) and the device (GPU) because data can be loaded in GPU's buffer memory. Hence the discrepancy between "Time per cycle" and "Time to solve" in CUDA case.

Note that by far the most computational expensive tasks are the FFT and IFFT blocks for which Numpy's library is already optimised, therefore the time taken by the Python code should be on the same order of magnitude as an equivalent C/C++ sequential implementation.

| Method | Implementation | Number of iterations | Time to solve (ms) | Time per cycle (ms) |
|:------:|:--------------:|:--------------------:|:------------------:|:-------------------:|
|   GS   |      Numpy     |          50          |        12222       |        244.4        |
|        |      CUDA      |          50          |        277.8       |        3.682        |
|  MRAF  |      Numpy     |          50          |        20000       |         400         |
|        |      CUDA      |          50          |         298        |        4.120        |
|   WGS  |      Numpy     |          50          |        11434       |        228.7        |
|        |      CUDA      |          50          |        345.3       |        4.998        |
| UCMRAF |      Numpy     |          50          |        20774       |        415.5        |
|        |      CUDA      |          50          |        302.5       |        4.338        |

### New method introduced: Uniformity Controlled MRAF
 A new phase retrieval algorithm has been introduced here. This is an improvement of the MRAF by allowing optimisation of the parameter from MRAF feedback equation.
 
### Methods compared

<img src="https://github.com/cristi-bourceanu/PhaseRetrieval/blob/master/CUDA%20Implementation/Data/Figures/Uniformity.png" width="60%">

<img src="https://github.com/cristi-bourceanu/PhaseRetrieval/blob/master/CUDA%20Implementation/Data/Figures/Accuracy.png" width="60%">

<img src="https://github.com/cristi-bourceanu/PhaseRetrieval/blob/master/CUDA%20Implementation/Data/Figures/Efficiency.png" width="60%">


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
  - AND MANY OTHERS
- Call Algorithm to iterate
3. Algorithm
- Factory method to choose an algorithm to solve the problem. Implemented algorithms:
  - Gerchberg Saxton
  - Weighted GS
  - MRAF
  - Uniformity Controlled MRAF
