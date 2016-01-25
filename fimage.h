/* This file is part of cudaTV.  See the file COPYING for further details. */
#ifndef FIMAGE_FACCIOLO_H
#define FIMAGE_FACCIOLO_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* declarations */
typedef struct fimage {
	int nrow;
	int ncol;
	float *gray;
} *Fimage;

static Fimage new_fimage(void);


static Fimage new_fimage2(int nx, int ny);

static void del_fimage(Fimage i);

static Fimage copy_fimage(Fimage dest, Fimage src);


/* definitions */
Fimage new_fimage(void)
{
	Fimage image;

	if (!(image = (Fimage)(malloc(sizeof(struct fimage)))))
	{
		fprintf(stderr, "[new_fimage] Not enough memory\n");
		exit(1);
		return(NULL);
	}

	image->nrow = image->ncol = 0;
	image->gray = NULL;
	return(image);
}


Fimage new_fimage2(int nx, int ny){
	Fimage t = new_fimage();
	t->ncol = nx;
	t->nrow = ny;
	t->gray = (float *)malloc(sizeof(float)*t->ncol*t->nrow);
	return (t);
}


void del_fimage(Fimage i) {
	if (i->gray) free(i->gray);
	free(i);
}

Fimage copy_fimage(Fimage dest, Fimage src){
	if ((dest->ncol == src->ncol) && (dest->nrow == src->nrow)){
		memcpy(dest->gray, src->gray, src->ncol*src->nrow*sizeof(float));
	}
	else {
		fprintf(stderr, "[copy_fimage] Image sizes does not match\n");
	}
	return dest;
}
#endif

