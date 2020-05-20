CXX=g++

CUDA_INSTALL_PATH=/opt/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcublas -lcudart -lcufft -lcurand `pkg-config --libs opencv`

all:
	nvcc $(CFLAGS) -c  algorithm.cu -o /Debug/algorithm.o
	nvcc $(CFLAGS) -c  kernels.cu -o /Debug/kernels.o
	$(CXX) $(CFLAGS) -c -Wall -Wextra .display.cpp -o /Debug/display.o
	$(CXX) $(CFLAGS) -c -Wall -Wextra main.cpp -o /Debug/main.o
	$(CXX) $(LDFLAGS) /Debug/main.o /Debug/algorithm.o /Debug/kernels.o /Debug/display.o -o /Debug/showGS

clean:
	rm -f Debug/*.o Debug/showFFT