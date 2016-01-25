
#define TagCount			44		//tag 的数目
//tag的名称					10进制	16进制表示数
#define NewSubfileType		254		//0xfe
#define SubfileType			255		//0xff
#define ImageWidth			256		//0x100
#define ImageLength			257		//0x101
#define BitsPerSample		258		//0x102
#define Compression			259		//0x103

#define PhotometricInterp	262		//0x106
#define Threshholding		263		//0x107
#define CellWidth			264		//0x108
#define CellLength			265		//0x109
#define FillOrder			266		//0x10a

#define DocumentName		269		//0x10d
#define ImageDescription	270		//0x10e
#define Make				271		//0x10f
#define Model				272		//0x110
#define StripOffsets		273		//0x111
#define Orientation			274		//0x112

#define SamplesPerPixel		277		//0x115
#define RowsPerStrip		278		//0x116
#define StripByteCounts		279		//0x117
#define MinSampleValue		280		//0x118
#define MaxSampleValue		281		//0x119
#define XResolution			282		//0x11a
#define YResolution			283		//0x11b
#define PlanarConfiguration	284		//0x11c
#define PageName			285		//0x11d
#define XPosition			286		//0x11e
#define YPosition			287		//0x11f
#define FreeOffsets			288		//0x120
#define FreeByteCounts		289		//0x121
#define GrayResponseUnit	290		//0x122
#define GrayResponseCurve	291		//0x123
#define Group3Options		292		//0x124
#define Group4Options		293		//0x125

#define ResolutionUnit		296		//0x128
#define PageNumber			297		//0x129

#define ColorResponseUnit	300		//0x12c
#define ColorResponseCurves	301		//0x12d

#define Software			305		//0x131
#define DateTime			306		//0x132

#define Artist				315		//0x13b
#define HostComputer		316		//0x13c
#define Predictor			317		//0x13d
#define WhitePoint			318		//0x13e

//tag的数据类型
#define TIFFbyte		1
#define TIFFascii		2
#define TIFFshort		3
#define TIFFlong		4
#define TIFFrational	5

//压缩方式
#define COMPnone		1
#define COMPhuff		2
#define COMPfax3		3
#define COMPfax4		4
#define COMPwrd1		0x8003
#define COMPmpnt		0x8005
