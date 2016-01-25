
#include <iostream>
#include "WTiffIO.h"
#include "tiff.h"
#include <stdio.h>
//#include "Windows.h"
#ifndef GRAY_LEVEL
#define GRAY_LEVEL 60000
#endif

WTiffIO::WTiffIO(void)
{
/*	
	ImageWidth = 0;
	ImageLength = 0;
	BitsPerSample = 0;
	Compression = 0;
	PhotometricInterp = 0;
	StripOffsets = 0;
	XResolution = 0;
	YResolution = 0;
	ResolutionUnit = 0;
	Software = 0;
	DateTime = 0;
	Artist = 0;
*/
}

WTiffIO::~WTiffIO(void)
{
}
bool WTiffIO::ReadMyTiffHeader(FILE* fp)
{
/*	fputWord(fp,'II');
	fputWord(fp,42);
	fputLong(fp,0L);
*/	
	char tag[2];
	if(fp==NULL)
	{
		cout<<"File pointer is NULL"<<endl;
		return false;
	}
	fread(tag,sizeof(char),2,fp);
	if(tag[0]!='I'||tag[1]!='I')
	{
		cout<<"Read My Tiff Header Error"<<endl;
		return false;
	}
	fread(tag,sizeof(char),2,fp);//read 42
	//should check 42,but i did not
	fread(tag,sizeof(char),2,fp);
	fread(tag,sizeof(char),2,fp);//read long
	return true;
}
bool WTiffIO::ReadMyTiffTag(FILE* fp,unsigned short* tag,unsigned short* type,unsigned long* length,unsigned long* offset)
{
/*	fputWord(fp,tag);
	fputWord(fp,type);
	fputLong(fp,length);
	fputLong(fp,offset);
*/
	if(fp==NULL)
	{
		cout<<"file pointer is null in readmytifftag"<<endl;
		return false;
	}
	fread(tag,sizeof(unsigned short),1,fp);
	fread(type,sizeof(unsigned short),1,fp);
	fread(length,sizeof(unsigned long),1,fp);
	fread(offset,sizeof(unsigned long),1,fp);
	return true;
}
void WTiffIO::ReadMyTiff(char* FileName,unsigned short *& Image,int &width,int &height)
{
	FILE* fp;
	long l;			//记录读写文件时指针的位置
	char Strartist[50];
	long lartist;
	char Strsoftware[30];
	long lsoftware;
	char Strdatetime[20];
	long ldatetime;
	char temp[15];
	memset(Strartist,0,50);
	memset(Strsoftware,0,30);
	memset(Strdatetime,0,20);
	strcpy(Strartist,"Copyright Tsinghua INET CT Research.");
	strcpy(Strsoftware,"Navy ICT System");
	_strdate(Strdatetime);
	_strtime(temp);
	strcat(Strdatetime," ");
	strcat(Strdatetime,temp);

	if((fp=fopen(FileName,"rb"))==NULL)
	{
		cout<<"open file error!"<<endl;
		return;
	}
	//写入文件头
//	WriteTiffHeader(fp);
	ReadMyTiffHeader(fp);

	//写入
	lartist=ftell(fp);//get current position of file
//	fputs(Strartist,fp);
	fgets(Strartist, strlen(Strartist), fp);
//	fputc(0,fp);
	int re=fgetc(fp);
	if(strlen(Strartist)%2==0)
	{
		//fputc(0,fp);
		re=fgetc(fp);
	}
	re=fgetc(fp);
	lsoftware=ftell(fp);//get current pointer position
//	fputs(Strsoftware,fp);
	
	re=fgetc(fp);
//	re=fgetc(fp);
//	re=fgetc(fp);
//	re=fgetc(fp);


	fgets(Strsoftware,strlen(Strsoftware)+1,fp);//Navy ICT System
//	fputc(0,fp);
	re=fgetc(fp);
	if(strlen(Strsoftware)%2==0)
		re=fgetc(fp);
	ldatetime=ftell(fp);
//	fputs(Strdatetime,fp);
	fgets(Strdatetime,strlen(Strdatetime)+1,fp);//length of data is 17chars;

	//计算X和Y方向的分辨率
	int xRes;//=10*width/realwidth;	//以cm为一个单位
	int yRes;//=10*height/realheight;	//其中包括的象素数
	long datastart;
	l=ftell(fp);
//	fputWord(fp,12);			//共写入12个tags
	re=fgetc(fp);
	unsigned short tagNO=0;
	fread(&tagNO,sizeof(unsigned short),1,fp);
	datastart=l+2+12*tagNO+4;		//图象数据的起始地址
	//写入IDF
/*	WriteTiffTag(fp,ImageWidth,		TIFFlong,1,width);
	WriteTiffTag(fp,ImageLength,	TIFFlong,1,height);
	WriteTiffTag(fp,BitsPerSample,	TIFFshort,1,16);		//16位灰度数据
	WriteTiffTag(fp,Compression,	TIFFshort,1,COMPnone);
	WriteTiffTag(fp,PhotometricInterp,TIFFshort,1,1);
	WriteTiffTag(fp,StripOffsets,	TIFFlong,1,datastart);
	WriteTiffTag(fp,XResolution,	TIFFlong,1,xRes);
	WriteTiffTag(fp,YResolution,	TIFFlong,1,yRes);
	WriteTiffTag(fp,ResolutionUnit,	TIFFlong,1,3);
	WriteTiffTag(fp,Software,		TIFFlong,strlen(Strsoftware),lsoftware);
	WriteTiffTag(fp,DateTime,		TIFFlong,20,ldatetime);
	WriteTiffTag(fp,Artist,			TIFFlong,strlen(Strartist),lartist);
*/
	unsigned short tag=0,type=0;
	unsigned long length=0,offset=0;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==ImageWidth)
		width=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==ImageLength)
		height=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==BitsPerSample)
		int bitpersample=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==Compression)
		int compression=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==PhotometricInterp)
		int photometricinterp=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==StripOffsets)
		datastart=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==XResolution)
		xRes=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==YResolution)
		yRes=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==ResolutionUnit)
		int ResUnit=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==Software)
		int len_software=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==DateTime)
		int len_dateTime=offset;
	ReadMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==Artist)
		int len_artist=offset;
//	Image=(WORD*)calloc(width*height,sizeof(WORD));
	if(Image!=NULL)
		delete Image;
	Image=new unsigned short[width*height];
	memset(Image,0,width*height*sizeof(unsigned short));
	fseek(fp,4L,SEEK_SET);
//	fputLong(fp,l);
	char tempchar=fgetc(fp);
	tempchar=fgetc(fp);
	tempchar=fgetc(fp);
	tempchar=fgetc(fp);
	//写入数据
	fseek(fp,datastart,SEEK_SET);
//	for(int i=0;i<height;i++)	for(int j=0;j<width;j++)
//		fgetc(fp,image[i*width+j]);
	int count=fread(Image,sizeof(unsigned short),width*height,fp);
	if(count!=width*height)
	{
		cout<<"fread image data error"<<endl;
	}
	fclose(fp);
}
bool WTiffIO::WriteMyTiffHeader(FILE* fp)
{
/*	fputWord(fp,'II');
	fputWord(fp,42);
	fputLong(fp,0L);
*/	
	char tag[2];
	if(fp==NULL)
	{
		cout<<"File pointer is NULL"<<endl;
		return false;
	}
	fwrite(tag,sizeof(char),2,fp);
	if(tag[0]!='I'||tag[1]!='I')
	{
		cout<<"Read My Tiff Header Error"<<endl;
		return false;
	}
	fwrite(tag,sizeof(char),2,fp);//read 42
	//should check 42,but i did not
	fwrite(tag,sizeof(char),2,fp);
	fwrite(tag,sizeof(char),2,fp);//read long
	return true;
}
bool WTiffIO::WriteMyTiffTag(FILE* fp,unsigned short* tag,unsigned short* type,unsigned long* length,unsigned long* offset)
{
/*	fputWord(fp,tag);
	fputWord(fp,type);
	fputLong(fp,length);
	fputLong(fp,offset);
*/
	if(fp==NULL)
	{
		cout<<"file pointer is null in readmytifftag"<<endl;
		return false;
	}
	fwrite(tag,sizeof(unsigned short),1,fp);
	fwrite(type,sizeof(unsigned short),1,fp);
	fwrite(length,sizeof(unsigned long),1,fp);
	fwrite(offset,sizeof(unsigned long),1,fp);
	return true;
}
void WTiffIO::WriteMyTiff(char* FileName,unsigned short *& Image,int &width,int &height)
{
	FILE* fp;
	long l;			//记录读写文件时指针的位置
	char Strartist[50];
	long lartist;
	char Strsoftware[30];
	long lsoftware;
	char Strdatetime[20];
	long ldatetime;
	char temp[15];
	memset(Strartist,0,50);
	memset(Strsoftware,0,30);
	memset(Strdatetime,0,20);
	strcpy(Strartist,"Copyright Tsinghua INET CT Research.");
	strcpy(Strsoftware,"Navy ICT System");
	_strdate(Strdatetime);
	_strtime(temp);
	strcat(Strdatetime," ");
	strcat(Strdatetime,temp);

	if((fp=fopen(FileName,"rb"))==NULL)
	{
		cout<<"open file error!"<<endl;
		return;
	}
	//写入文件头
//	WriteTiffHeader(fp);
	WriteMyTiffHeader(fp);

	//写入
	lartist=ftell(fp);//get current position of file
	fputs(Strartist,fp);
//	fgets(Strartist, strlen(Strartist), fp);
	fputc(0,fp);
//	int re=fgetc(fp);
	if(strlen(Strartist)%2==0)
	{
		fputc(0,fp);
	//	re=fgetc(fp);
	}
//	re=fgetc(fp);
	lsoftware=ftell(fp);//get current pointer position
	fputs(Strsoftware,fp);
	
//	re=fgetc(fp);
//	re=fgetc(fp);
//	re=fgetc(fp);
//	re=fgetc(fp);


//	fgets(Strsoftware,strlen(Strsoftware)+1,fp);//Navy ICT System
	fputc(0,fp);
//	re=fgetc(fp);
//	if(strlen(Strsoftware)%2==0)
//		re=fgetc(fp);
	ldatetime=ftell(fp);
	fputs(Strdatetime,fp);
//	fgets(Strdatetime,strlen(Strdatetime)+1,fp);//length of data is 17chars;

	//计算X和Y方向的分辨率
	int xRes;//=10*width/realwidth;	//以cm为一个单位
	int yRes;//=10*height/realheight;	//其中包括的象素数
	long datastart;
	l=ftell(fp);
	fputWord(fp,12);			//共写入12个tags
//	re=fgetc(fp);
	unsigned short tagNO=0;
	fwrite(&tagNO,sizeof(unsigned short),1,fp);
	datastart=l+2+12*tagNO+4;		//图象数据的起始地址
	//写入IDF
/*	WriteTiffTag(fp,ImageWidth,		TIFFlong,1,width);
	WriteTiffTag(fp,ImageLength,	TIFFlong,1,height);
	WriteTiffTag(fp,BitsPerSample,	TIFFshort,1,16);		//16位灰度数据
	WriteTiffTag(fp,Compression,	TIFFshort,1,COMPnone);
	WriteTiffTag(fp,PhotometricInterp,TIFFshort,1,1);
	WriteTiffTag(fp,StripOffsets,	TIFFlong,1,datastart);
	WriteTiffTag(fp,XResolution,	TIFFlong,1,xRes);
	WriteTiffTag(fp,YResolution,	TIFFlong,1,yRes);
	WriteTiffTag(fp,ResolutionUnit,	TIFFlong,1,3);
	WriteTiffTag(fp,Software,		TIFFlong,strlen(Strsoftware),lsoftware);
	WriteTiffTag(fp,DateTime,		TIFFlong,20,ldatetime);
	WriteTiffTag(fp,Artist,			TIFFlong,strlen(Strartist),lartist);
*/
	unsigned short tag=0,type=0;
	unsigned long length=0,offset=0;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==ImageWidth)
		width=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==ImageLength)
		height=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==BitsPerSample)
		int bitpersample=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==Compression)
		int compression=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==PhotometricInterp)
		int photometricinterp=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==StripOffsets)
		datastart=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==XResolution)
		xRes=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==YResolution)
		yRes=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==ResolutionUnit)
		int ResUnit=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==Software)
		int len_software=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==DateTime)
		int len_dateTime=offset;
	WriteMyTiffTag(fp,&tag,&type,&length,&offset);
	if(tag==Artist)
		int len_artist=offset;
//	Image=(WORD*)calloc(width*height,sizeof(WORD));
	if(Image!=NULL)
		delete Image;
	Image=new unsigned short[width*height];
	memset(Image,0,width*height*sizeof(unsigned short));
	fseek(fp,4L,SEEK_SET);
	fputLong(fp,l);
//	char tempchar=fgetc(fp);
//	tempchar=fgetc(fp);
//	tempchar=fgetc(fp);
//	tempchar=fgetc(fp);
	//写入数据
	fseek(fp,datastart,SEEK_SET);
//	for(int i=0;i<height;i++)	for(int j=0;j<width;j++)
//		fgetc(fp,image[i*width+j]);
	int count=fwrite(Image,sizeof(unsigned short),width*height,fp);
	if(count!=width*height)
	{
		cout<<"fread image data error"<<endl;
	}
	fclose(fp);
}
void WTiffIO::SaveFloatAsTiff(char* filename, int width, int height,float *img,  int sign)
{//
	unsigned short* image;
	image=new unsigned short[width*height];
	memset(image,0,width*height*sizeof(unsigned short));
	float Max,Min;
	FindMaxMin(img,width*height,&Max,&Min);
	switch(sign)
	{
	case 0:
		{//最大最小映射到灰度范围
			for(int i=0;i<height;i++)
			{
				for(int j=0;j<width;j++)
				{
					image[i*width+j]=(unsigned short)((img[i*width+j]-Min)*GRAY_LEVEL/(Max-Min));
				//	TRACE("image[i*width+j]=%d",image[i*width+j]);
				}
			}
			break;
		}	
	case 1:
		{//放弃了小于零的那部分数据
			for(int i=0;i<height;i++)
			{
				for(int j=0;j<width;j++)
				{
					if(img[i*width+j]<0)
						img[i*width+j]=0;
					image[i*width+j]=(unsigned short)((img[i*width+j])*GRAY_LEVEL/(Max-Min));
				}
			}
			break;
		}
	default:
		{
			cout<<"Bad  sign in SaveFloatAsTiff"<<endl;
			break;
		}
	}

	int realwidth=1,realheight=1;
	SaveTiff(filename,image,width,height,realwidth,realheight);
	delete image;

}

void WTiffIO::SaveFloatAsBmp(char* filename, int width, int height, float *img)
{//
	unsigned char* image;
	image=new unsigned char[width*height];
	memset(image,0,width*height*sizeof(unsigned char));
	float Max,Min;
	FindMaxMin(img,width*height,&Max,&Min);
	//最大最小映射到灰度范围
		for(int i=0;i<height;i++)
		{
			for(int j=0;j<width;j++)
			{
				image[i*width+j]=(unsigned char)((img[i*width+j]-Min)*255/(Max-Min));
			//	TRACE("image[i*width+j]=%d",image[i*width+j]);
			}
		}	
	SaveBitmap(filename,width,height,image);
	delete image;

}

//保存图象文件为TIFF格式
void WTiffIO::SaveTiff(char* filename,unsigned short* data,int width,int height,int realwidth,int realheight)
{
	FILE* fp;
	long l;			//记录读写文件时指针的位置
	char Strartist[50];
	long lartist;
	char Strsoftware[30];
	long lsoftware;
	char Strdatetime[20];
	long ldatetime;
	char temp[15];
	memset(Strartist,0,50);
	memset(Strsoftware,0,30);
	memset(Strdatetime,0,20);
	strcpy(Strartist,"Copyright Tsinghua INET CT Research.");
	strcpy(Strsoftware,"Navy ICT System");
	_strdate(Strdatetime);
	_strtime(temp);
	strcat(Strdatetime," ");
	strcat(Strdatetime,temp);

	if((fp=fopen(filename,"w+b"))==NULL)
	{
		cout<<"open file error!"<<endl;
		printf("open file error!\n");
		return;
	}
	//写入文件头
	WriteTiffHeader(fp);

	//写入
	lartist=ftell(fp);
	fputs(Strartist,fp);
	fputc(0,fp);
	if(strlen(Strartist)%2==0)
		fputc(0,fp);
	lsoftware=ftell(fp);
	fputs(Strsoftware,fp);
	fputc(0,fp);
	if(strlen(Strsoftware)%2==0)
		fputc(0,fp);
	ldatetime=ftell(fp);
	fputs(Strdatetime,fp);
	fputc(0,fp);

	//计算X和Y方向的分辨率
	int xRes=10*width/realwidth;	//以cm为一个单位
	int yRes=10*height/realheight;	//其中包括的象素数
	long datastart;
	l=ftell(fp);
	fputWord(fp,12);			//共写入12个tags
	datastart=l+2+12*12+4;		//图象数据的起始地址
	//写入IDF
	WriteTiffTag(fp,ImageWidth,		TIFFlong,1,width);
	WriteTiffTag(fp,ImageLength,	TIFFlong,1,height);
	WriteTiffTag(fp,BitsPerSample,	TIFFshort,1,16);		//16位灰度数据
	WriteTiffTag(fp,Compression,	TIFFshort,1,COMPnone);
	WriteTiffTag(fp,PhotometricInterp,TIFFshort,1,1);
	WriteTiffTag(fp,StripOffsets,	TIFFlong,1,datastart);
	WriteTiffTag(fp,XResolution,	TIFFlong,1,xRes);
	WriteTiffTag(fp,YResolution,	TIFFlong,1,yRes);
	WriteTiffTag(fp,ResolutionUnit,	TIFFlong,1,3);
	WriteTiffTag(fp,Software,		TIFFlong,strlen(Strsoftware),lsoftware);
	WriteTiffTag(fp,DateTime,		TIFFlong,20,ldatetime);
	WriteTiffTag(fp,Artist,			TIFFlong,strlen(Strartist),lartist);
	fseek(fp,4L,SEEK_SET);
	fputLong(fp,l);
	//写入数据
	fseek(fp,datastart,SEEK_SET);
	for(int i=0;i<height;i++)	for(int j=0;j<width;j++)
		fputWord(fp,data[i*width+j]);

	fclose(fp);
}

//写入tag
void WTiffIO::WriteTiffTag(FILE* fp,int tag,int type,long length,long offset)
{
	fputWord(fp,tag);
	fputWord(fp,type);
	fputLong(fp,length);
	fputLong(fp,offset);
}

//写入TIFF文件头 指向tag的指针暂为零
void WTiffIO::WriteTiffHeader(FILE* fp)
{
	fputWord(fp,'II');
	fputWord(fp,42);
	fputLong(fp,0L);
}

//向文件写入一个字 即写入两字节
void WTiffIO::fputWord(FILE* fp,int n)
{
	fputc(n,fp);
	fputc(n>>8,fp);
}

//向文件写入一个长整型数 即写入四个字节
void WTiffIO::fputLong(FILE* fp,int n)
{
	fputc(n,fp);
	fputc(n>>8,fp);
	fputc(n>>16,fp);
	fputc(n>>24,fp);
}
bool WTiffIO::FindMaxMin(float *pData, int Length, float *max, float *min)
{
	*max=pData[0];
	*min=pData[0];
//	char str="";
	for(int i=0;i<Length;i++)
	{
		if(pData[i]>*max)*max=pData[i];
		if(pData[i]<*min)*min=pData[i];
	/*	if(pData[i]<-2)
		{
			str.Format("i=%d",i);
			AfxMessageBox(str);
		}
	*/
	}
	return true;

}
bool WTiffIO::FindMaxMin(unsigned short *pData, int Length, float *max, float *min)
{
	*max=pData[0];
	*min=pData[0];
//	CString str="";
	for(int i=0;i<Length;i++)
	{
		if(pData[i]>*max)*max=pData[i];
		if(pData[i]<*min)*min=pData[i];
	/*	if(pData[i]<-2)
		{
			str.Format("i=%d",i);
			AfxMessageBox(str);
		}
*/
	}
	return true;

}
void WTiffIO::SaveAsBitmap( const char *filename, const short width, const short height, const unsigned short *data )
{
	unsigned char* bmp;
	bmp=new unsigned char[width*height];
	memset(bmp,0,sizeof(unsigned char)*width*height);
	for(int i=0;i<height*width;i++)
		bmp[i]=data[i]/256.0;
		

}
void WTiffIO::SaveBitmap( const char *filename, const short width, const short height, const unsigned char *data )
{
	BITMAPFILEHEADER bmfh;
	BITMAPINFOHEADER bmih;
	RGBQUAD			aColors[256];
//	BYTE			*aBitmapBits;
	FILE			*fd;
//	register short i,colorNum;
	register short i;
//	short			bfType;

	fd = fopen(filename, "wb");
	if (fd == NULL)
	{
		printf("File %s write error.\n",filename);
		return;
	}

//  bfType = 'M' * 0x100 + 'B';
//  fwrite(&bfType,1,sizeof(short),fd);

	bmfh.bfType = 'M' * 0x100 + 'B';	//
//	bmfh.bfOffBits = sizeof(short) + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD);
	bmfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD);
	bmfh.bfSize = bmfh.bfOffBits + width * height;
//  bmfh.bfReserved = 0;
	bmfh.bfReserved1 = 0;		//
	bmfh.bfReserved2 = 0;		//
	fwrite(&bmfh,1,sizeof(BITMAPFILEHEADER),fd);
	bmih.biSize = sizeof(BITMAPINFOHEADER);
//  bmih.biCompression = 0;
	bmih.biCompression = 0L;	//#define BI_RGB        0L
//  bmih.biBitCount = 0x00080001;
	bmih.biBitCount = 8;		//	this statement is right
	bmih.biWidth = width;
//	bmih.biHeight = height;
	bmih.biHeight = height;	//图像是以左上角为坐标零点的
	bmih.biPlanes = 1 ;		//
	bmih.biSizeImage = width * height;
//  bmih.biSizeImage = 0;	//
	bmih.biXPelsPerMeter = 0x0;
	bmih.biYPelsPerMeter = 0x0;
	bmih.biClrUsed = 256;
	bmih.biClrImportant = 0;
	fwrite(&bmih,1,sizeof(BITMAPINFOHEADER),fd);

//  for (i = 0;i < 256;++i)
	for(i=0;i<256;i++)		//
	{
		aColors[i].rgbBlue = aColors[i].rgbGreen = aColors[i].rgbRed = (unsigned char)i;
		aColors[i].rgbReserved = 0 ;	//
	}
	fwrite(aColors,256,sizeof(RGBQUAD),fd);
	for(i=height-1;i>=0;i--)
		fwrite(data+i*width, sizeof(unsigned char), (width+3)/4*4, fd);

	fclose(fd);
	return;



}