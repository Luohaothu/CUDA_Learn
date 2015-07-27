#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

#define CUDA_CALL(x){	const cudaError_t a = (x); \
						if (a != cudaSuccess) \
												{\
							printf("\nCUDA Error: %s (err_num = %d) \n", cudaGetErrorString(a), a);\
							cudaDeviceReset(); assert(0); \
												}\
										}
#define KERNEL_LOOP 65536

typedef unsigned int u32;
__constant__ static const u32 const_data_01 = 0x55555555;
__constant__ static const u32 const_data_02 = 0x77777777;
__constant__ static const u32 const_data_03 = 0x33333333;
__constant__ static const u32 const_data_04 = 0x11111111;

__device__ static u32 data_01 = 0x55555555;
__device__ static u32 data_02 = 0x77777777;
__device__ static u32 data_03 = 0x33333333;
__device__ static u32 data_04 = 0x11111111;

__global__ void const_test_gpu_gmem(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < num_elements)
	{
		u32 d = 0x55555555;
		for (int i = 0; i < KERNEL_LOOP; i++)
		{
			d ^= data_01;
			d |= data_02;
			d &= data_03;
			d |= data_04;
		}

		data[tid] = d;
	}
}

__global__ void const_test_gpu_const(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < num_elements)
	{
		u32 d = const_data_01;
		for (int i = 0; i < KERNEL_LOOP; i++)
		{
			d ^= const_data_01;
			d |= const_data_02;
			d &= const_data_03;
			d |= const_data_04;
		}

		data[tid] = d;
	}
}

__host__ void wait_exit(void)
{
	char ch;
	printf("\nPress any key to exit\n");
	ch = getch();
}

__host__ void cuda_error_check(const char * prefix, const char * postfix)
{
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s%s%s\n", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
		cudaDeviceReset();
		wait_exit();
		exit(1);
	}
}

__host__ void gpu_kernel(void)
{
	const u32 num_elements = (128 * 1024);
	const u32 num_threads = 256;
	const u32 num_blocks = (num_elements + (num_threads - 1)) / num_threads;
	const u32 num_bytes = num_elements * sizeof(u32);
	int max_device_num;
	const int max_runs = 6;

	CUDA_CALL(cudaGetDeviceCount(&max_device_num));

	for (int device_num = 0; device_num < max_device_num; device_num++)
	{
		CUDA_CALL(cudaSetDevice(device_num));

		for (int num_test = 0; num_test < max_runs; num_test++)
		{
			u32 * data_gpu;
			cudaEvent_t kernel_start1, kernel_stop1;
			cudaEvent_t kernel_start2, kernel_stop2;
			float delta_time1 = 0.0f, delta_time2 = 0.0f;
			cudaDeviceProp device_prop;
			char device_prefix[261];

			CUDA_CALL(cudaMalloc(&data_gpu, num_bytes));
			CUDA_CALL(cudaEventCreate(&kernel_start1));
			CUDA_CALL(cudaEventCreate(&kernel_start2));
			CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop1, cudaEventBlockingSync));
			CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop2, cudaEventBlockingSync));

			CUDA_CALL(cudaGetDeviceProperties(&device_prop, device_num));
			sprintf(device_prefix, "ID: %d Name: %s", device_num, device_prop.name);

			//Warm up run
			const_test_gpu_gmem << <num_blocks, num_threads >> >(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal startup kernel");

			//Do the literal kernel
			CUDA_CALL(cudaEventRecord(kernel_start1, 0));
			const_test_gpu_gmem << <num_blocks, num_threads >> >(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal runtime kernel");

			CUDA_CALL(cudaEventRecord(kernel_stop1, 0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop1));
			CUDA_CALL(cudaEventElapsedTime(&delta_time1, kernel_start1, kernel_stop1));
			//printf("\nLiteral Elapsed time: %.3fms\n", delta_time1);

			//Warm up run
			const_test_gpu_const << <num_blocks, num_threads >> >(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from constant startup kernel");

			//Do the constant kernel
			CUDA_CALL(cudaEventRecord(kernel_start2, 0));

			const_test_gpu_const << <num_blocks, num_threads >> >(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from constatn runtime kernel");

			CUDA_CALL(cudaEventRecord(kernel_stop2, 0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop2));
			CUDA_CALL(cudaEventElapsedTime(&delta_time2, kernel_start2, kernel_stop2));
			//printf("\nConst Elapsed time: %.3fms\n", delta_time2);

			if (delta_time1 > delta_time2)
			{
				printf("\n%s Constant version is faster by: %.2fms (Const=%.2fms vs. Literal=%.2fms)",
					device_prefix, delta_time1 - delta_time2, delta_time1, delta_time2);
			}
			else
			{
				printf("\n%s Literal version is faster by: %.2fms (Const=%.2fms vs. Literal=%.2fms)",
					device_prefix, delta_time2 - delta_time1, delta_time1, delta_time2);
			}

			CUDA_CALL(cudaEventDestroy(kernel_start1));
			CUDA_CALL(cudaEventDestroy(kernel_start2));
			CUDA_CALL(cudaEventDestroy(kernel_stop1));
			CUDA_CALL(cudaEventDestroy(kernel_stop2));
			CUDA_CALL(cudaFree(data_gpu));
		}

		CUDA_CALL(cudaDeviceReset());
		printf("\n");
	}

	wait_exit();
}

int main()
{
	gpu_kernel();
	return 0;
}