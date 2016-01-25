#pragma once
#include <iostream>
#include <time.h>
#include <stdio.h>
using namespace std;
typedef struct tagBITMAPFILEHEADER { 
	unsigned short    bfType; 
	unsigned long   bfSize; 
	unsigned short    bfReserved1; 
	unsigned short    bfReserved2; 
	unsigned long   bfOffBits; 
} BITMAPFILEHEADER, *PBITMAPFILEHEADER; 
typedef struct tagRGBQUAD {
	unsigned char    rgbBlue; 
	unsigned char    rgbGreen; 
	unsigned char    rgbRed; 
	unsigned char    rgbReserved; 
} RGBQUAD; 
typedef struct tagBITMAPINFOHEADER{
	unsigned long  biSize; 
	long   biWidth; 
	long   biHeight; 
	unsigned short   biPlanes; 
	unsigned short   biBitCount; 
	unsigned long  biCompression; 
	unsigned long  biSizeImage; 
	long   biXPelsPerMeter; 
	long   biYPelsPerMeter; 
	unsigned long  biClrUsed; 
	unsigned long  biClrImportant; 
} BITMAPINFOHEADER, *PBITMAPINFOHEADER; 
class WTiffIO
{
public:
	WTiffIO(void);
	~WTiffIO(void);
	bool ReadMyTiffHeader(FILE* fp);
	bool ReadMyTiffTag(FILE* fp,unsigned short* tag,unsigned short* type,unsigned long* length,unsigned long* offset);
	void ReadMyTiff(char* FileName,unsigned short*& Image,int &width,int &height);
	bool WriteMyTiffHeader(FILE* fp);
	bool WriteMyTiffTag(FILE* fp,unsigned short* tag,unsigned short* type,unsigned long* length,unsigned long* offset);
	void WriteMyTiff(char* FileName,unsigned short*& Image,int &width,int &height);
	void SaveTiff(char* filename,unsigned short * data,int width,int height,int realwidth,int realheight);
	void SaveFloatAsTiff(char* filename, int width, int height, float *img,int sign);
	void SaveBitmap( const char *filename, const short width, const short height, const unsigned char *data );
	void SaveAsBitmap( const char *filename, const short width, const short height, const unsigned short *data );
	void SaveFloatAsBmp(char* filename, int width, int height, float *img);
private:
	void fputLong(FILE* fp,int n);
	void fputWord(FILE* fp,int n);
	void WriteTiffHeader(FILE* fp);
	void WriteTiffTag(FILE* fp,int tag,int type,long length,long offset);
	// 	bool WTiffIO::FindMaxMin(float *pData, int Length, float *max, float *min);
	// 	bool WTiffIO::FindMaxMin(unsigned short *pData, int Length, float *max, float *min);
	bool FindMaxMin(float *pData, int Length, float *max, float *min);
	bool FindMaxMin(unsigned short *pData, int Length, float *max, float *min);

};
