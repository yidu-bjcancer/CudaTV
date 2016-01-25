/* This file is part of cudaTV.  See the file COPYING for further details. */
#ifndef PGM_IO_H
#define PGM_IO_H

/*--------------------------------------------------------------------------*/
/*------------------------------- IMAGE I/O --------------------------------*/
/*--------------------------------------------------------------------------*/

#include "fimage.h"
#include <iostream>

int verbose = 0;

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

/*--------------------------------------------------------------------------*/
static void skip_what_should_be_skipped(FILE * f)
{
	int c;
	do
	{
		while (isspace(c = getc(f))); /* skip spaces */
		if (c == '#') while ((c = getc(f)) != '\n'); /* skip comments */
	} while (c == '#');
	ungetc(c, f);
}

void TiffOutput(float *buffer, int Height, int Width, int PageNO);


/*--------------------------------------------------------------------------*/
/* Decimal read*/
static float get_num(FILE *f)
{
	float c;
	fscanf(f, "%g ", &c);
	return c;
}

static int get_num2(FILE * f)
{
	int num, c;

	while (isspace(c = getc(f)));
	if (!isdigit(c)) fprintf(stderr, "Corrupted PGM file.");
	num = c - '0';
	while (isdigit(c = getc(f))) num = 10 * num + c - '0';

	return num;
}

/*--------------------------------------------------------------------------*/
static Fimage read_pgm_fimage_stream(FILE *f)
{
	int c, bin, x, y;
	int xsize, ysize, depth;
	Fimage image;

	if (f == NULL) fprintf(stderr, "can't read input file.");

	/* read headder */
/*
	if (getc(f) != 'P') fprintf(stderr, "Not a PGM file!");
	if ((c = getc(f)) == '2') bin = FALSE;
	else if (c == '5') bin = TRUE;
	else fprintf(stderr, "Not a PGM file!");
	skip_what_should_be_skipped(f);
	fscanf(f, "%d", &xsize);
	skip_what_should_be_skipped(f);
	fscanf(f, "%d", &ysize);
	skip_what_should_be_skipped(f);
	fscanf(f, "%d", &depth);
*/

	xsize = 256;
	ysize = 256;
	fseek(f, 28, 0);
	std::cout<<ftell(f);
	/* get memory */
	image = new_fimage2(xsize, ysize);
	image->ncol = xsize;
	image->nrow = ysize;

	/* read data */
//	skip_what_should_be_skipped(f);

	for (y = 0; y < ysize; y++)
	{
		for (x = 0; x < xsize; x++)
		{
			//image->gray[x + y * xsize] = (float)(bin ? getc(f) : get_num(f));

			//std::cout << getc(f) << "    ";
			
			float tmp = (float)(getc(f));
			image->gray[x + y * xsize] = tmp;
			
		}
		std::cout << std::endl;
	}

	fseek(f, 28, 0);

	unsigned char Img[256 * 256] = { 0 };

	fread(Img, 256 * 256, 1, f);

	float fImg[256 * 256] = { 0 };

	for (int ii = 0; ii < 256 * 256;ii++)
	{
		fImg[ii] = (float)(Img[ii]);
	}

	for (int ii = 0; ii < 256 * 256; ii++)
	{
		image->gray[ii] = fImg[ii];
	}

	if (verbose) fprintf(stderr, "input image: xsize %d ysize %d depth %d\n",
		image->ncol, image->nrow, depth);

	/* close file */

	//TiffOutput(fImg, 256, 256, 1);

	return image;
}

/*--------------------------------------------------------------------------*/
static Fimage read_pgm_fimage(char * name)
{
	FILE * f;
	int c, bin, x, y;
	int xsize, ysize, depth;
	Fimage image;

	/* open file */
	f = fopen(name, "r");
	if (f == NULL) fprintf(stderr, "can't open input file.");

	/* read from stream */
	image = read_pgm_fimage_stream(f);

	/* close file */
	fclose(f);

	return image;
}
/*--------------------------------------------------------------------------*/
static void write_pgm_fimage_stream(Fimage image, FILE *f)
{
	int x, y, n;

	if (f == NULL) fprintf(stderr, "can't open output file.");

	/* write headder */
	fprintf(f, "P2\n");
	fprintf(f, "%d %d\n", image->ncol, image->nrow);
	fprintf(f, "255\n");

	/* write data */
	for (n = 1, y = 0; y < image->nrow; y++)
		for (x = 0; x < image->ncol; x++, n++)
		{
			/*fprintf(f,"%d ",(int)image->gray[ x + y * image->ncol ]);*/
			fprintf(f, "%g ", (float)image->gray[x + y * image->ncol]);
			if (n == 16)
			{
				fprintf(f, "\n");
				n = 0;
			}
		}
}

/*--------------------------------------------------------------------------*/
static void write_pgm_fimage(Fimage image, char * name)
{
	FILE * f;
	int x, y, n;

	/* open file */
	f = fopen(name, "w");
	if (f == NULL) fprintf(stderr, "can't open output file.");

	/* write the file*/
	write_pgm_fimage_stream(image, f);

	/* close file */
	fclose(f);
}
#endif
