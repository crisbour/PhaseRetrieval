CXX=g++

CUDA_INSTALL_PATH=/opt/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcufft `pkg-config --libs opencv`

all:
	$(CXX) $(CFLAGS) -c main.cpp -o ../Debug/main.o
	nvcc $(CFLAGS) -c algorithm.cu -o ../Debug/algorithm.o
	$(CXX) $(LDFLAGS) ../Debug/main.o ../Debug/algorithm.o -o ../Debug/showFFT

clean:
	rm -f Debug/*.o Debug/showFFT