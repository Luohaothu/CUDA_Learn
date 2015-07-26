#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DIMX 1920
#define DIMY 1080
struct cuComplex
{
	float r;
	float i;
	__device__ cuComplex(float a, float b) :r(a), i(b){};
	__device__ cuComplex(cuComplex& c) :r(c.r), i(c.i){};
	__device__ float magnitude2(void)
	{
		return r*r + i*i;
	}
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia(int x, int y);

__global__ void kernel(unsigned char *ptr);