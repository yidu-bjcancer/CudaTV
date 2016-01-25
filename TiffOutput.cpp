#include <iostream>
#include "WTiffIO.h"


void TiffOutput(float *buffer, int Height, int Width, int PageNO)
{
	// ***********  Date: 2014-11-7
	// Purpose:    Output float variables for debug
	// Input: 
	//       buffer:	float data for debug
	//		 Height:	Tiff Height(e.g SampleSliceNO for prj, and ImgSize for slice)
	//		 Width:		Tiff Width(e.g SampleSize for prj, and ImgSize for slice)
	//		 PageNo:	number of pages to be outputted
	// Directory and Name:
	//		 predefined in BufferFileName. D:\CTData\buffer****.tiff

	WTiffIO tiff;

	char BufferFileName[200];

	int BufferPageDataNO = Height * Width;

	for (int ii=0; ii<PageNO; ii++)
	{
		sprintf(BufferFileName,"%s%d%d%d%d%s","buffer",ii/1000,(ii%1000)/100,(ii%100)/10,ii%10,".tiff");

		// last parameter: 1 for discard negative pixel value
		//tiff.SaveFloatAsTiff(BufferFileName, Width, Height, buffer + ii*BufferPageDataNO, 1);

		// last parameter: 0 for not
		tiff.SaveFloatAsTiff(BufferFileName, Width, Height, buffer + ii*BufferPageDataNO, 0);

	}


}


