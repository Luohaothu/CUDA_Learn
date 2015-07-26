#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "julia.h"
#include "graph.h"
#include <pshpack2.h>
int main()
{
	struct
	{
		BYTE b;
		BYTE g;
		BYTE r;
	}graph[DIMX][DIMY];
	unsigned char *dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, DIMX*DIMY * 3);
	dim3 grid(DIMX, DIMY);
	kernel << <grid, 1 >> >(dev_bitmap);
	cudaMemcpy(&graph[0][0], dev_bitmap, DIMX*DIMY * 3, cudaMemcpyDeviceToHost);

	Snapshot((BYTE*)graph, DIMX, DIMY, "D:\\graph.bmp");
	return 0;
}

//void main()
//{
//	int i = 0, j = 0;
//	struct {
//		BYTE b;
//		BYTE g;
//		BYTE r;
//	} pRGB[240][320];  // 定义位图数据
//
//	memset(pRGB, 0, sizeof(pRGB)); // 设置背景为黑色
//
//	// 在中间画一个100*100的矩形
//	for (i = 70; i<170; i++){
//		for (j = 110; j<210; j++){
//			pRGB[i][j].r = 0xff;
//		}
//	}
//
//	// 生成BMP图片
//	Snapshot((BYTE*)pRGB, 320, 240, "D:\\rgb.bmp");
//}
//
