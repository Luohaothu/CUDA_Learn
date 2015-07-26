#include "kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if (!prop.deviceOverlap)
	{
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	int *host_a, *host_b, *host_c;
	int *dev_a0, *dev_b0, *dev_c0;
	int *dev_a1, *dev_b1, *dev_c1;

	cudaMalloc((void**)&dev_a0, N * sizeof(int));
	cudaMalloc((void**)&dev_b0, N * sizeof(int));
	cudaMalloc((void**)&dev_c0, N * sizeof(int));

	cudaMalloc((void**)&dev_a1, N * sizeof(int));
	cudaMalloc((void**)&dev_b1, N * sizeof(int));
	cudaMalloc((void**)&dev_c1, N * sizeof(int));

	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	for (int i = 0; i < FULL_DATA_SIZE; i += N * 2)
	{
		cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		kernel << <N / 256, 256, 0, stream0 >> >(dev_a0, dev_b0, dev_c0);

		cudaMemcpyAsync(dev_b1, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		kernel << <N / 256, 256, 0, stream1 >> >(dev_a1, dev_b1, dev_c1);
		
		cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c + i, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time taken: %3.1f ms\n", elapsedTime);

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_a1);
	cudaFree(dev_b1);
	cudaFree(dev_c1);

	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	return 0;
}