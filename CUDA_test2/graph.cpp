#include "graph.h"

void Snapshot(BYTE * pData, int width, int height, char * filename)
{
	int size = width*height * 3; // 每个像素点3个字节

	// 位图第一部分，文件信息
	BMPFILEHEADER_T bfh;
	bfh.bfType = 0x4d42;  //bm
	bfh.bfSize = size  // data size
		+ sizeof(BMPFILEHEADER_T) // first section size
		+ sizeof(BMPINFOHEADER_T) // second section size
		;
	bfh.bfReserved1 = 0; // reserved
	bfh.bfReserved2 = 0; // reserved
	bfh.bfOffBits = bfh.bfSize - size;



	// 位图第二部分，数据信息
	BMPINFOHEADER_T bih;
	bih.biSize = sizeof(BMPINFOHEADER_T);
	bih.biWidth = width;
	bih.biHeight = height;
	bih.biPlanes = 1;
	bih.biBitCount = 24;
	bih.biCompression = 0;
	bih.biSizeImage = size;
	bih.biXPelsPerMeter = 0;
	bih.biYPelsPerMeter = 0;
	bih.biClrUsed = 0;
	bih.biClrImportant = 0;

	FILE * fp = fopen(filename, "wb");
	if (!fp) return;

	fwrite(&bfh, 1, sizeof(BMPFILEHEADER_T), fp);

	fwrite(&bih, 1, sizeof(BMPINFOHEADER_T), fp);

	fwrite(pData, 1, size, fp);

	fclose(fp);

}
