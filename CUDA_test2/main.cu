#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "julia.h"
#include "graph.h"
#include <pshpack2.h>
int main()
{
//	BYTE graph[DIMX*DIMY*3];
	BYTE *graph = (BYTE*) malloc(sizeof(BYTE)*DIMX*DIMY * 3);
	unsigned char *dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, DIMX*DIMY * 3);
	cudaMemset((void*)dev_bitmap, 255, DIMX*DIMY * 3);
	const dim3 grid(DIMX/32, 1);
	const dim3 thread(32, 1);
	kernel << <grid, thread >> >(dev_bitmap);
	cudaMemcpy(graph, dev_bitmap, DIMX*DIMY * 3, cudaMemcpyDeviceToHost);
	Snapshot((BYTE*)graph, DIMX, DIMY, "D:\\graph.bmp");
	free(graph);
	cudaFree(dev_bitmap);
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
