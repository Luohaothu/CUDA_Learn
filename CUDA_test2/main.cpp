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
//	} pRGB[240][320];  // ����λͼ����
//
//	memset(pRGB, 0, sizeof(pRGB)); // ���ñ���Ϊ��ɫ
//
//	// ���м仭һ��100*100�ľ���
//	for (i = 70; i<170; i++){
//		for (j = 110; j<210; j++){
//			pRGB[i][j].r = 0xff;
//		}
//	}
//
//	// ����BMPͼƬ
//	Snapshot((BYTE*)pRGB, 320, 240, "D:\\rgb.bmp");
//}
//
