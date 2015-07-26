#include <device_launch_parameters.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c);