/* This file is part of cudaTV.  See the file COPYING for further details. */
// #include <stdio.h>
// #include <math.h>
#include <iostream>
#include  "fimage.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include  "pgm_io_float.h"

#include "referenceChambolle.h"
#include "chambolleTV.h"
#include "laplacian.h"
// #include "primaldualTV.h"



void listmethods() {
	printf("    1  reference implementation grad div laplacian\n");
	printf("    2  reference implementation laplacian stencil\n");
	printf("    3  cuda implementation grad div laplacian\n");
	printf("    4  cuda implementation laplacian stencil\n");
	printf("    5  reference implementation TV chambolle\n");
	printf("    6  cuda TV chambolle\n");
	printf("    7  cuda TV chambolle 2\n");
	printf("    8  cuda TV chambolle 2 shared memory\n");
	printf("    9  cuda TV chambolle 2 with stencil\n");
	printf("    10 cuda TV chambolle 2 with interleaved data vectors\n");
	printf("    11 cuda TV chambolle 2 with structure of arrays (FASTER)\n");
	printf("    12 cuda TV chambolle 2 dummy\n");
	printf("    13 cuda TV primal dual\n");
	printf("    14 cuda TV primal dual with structure of arrays\n");
	printf("    15 cuda TV primal dual : PDHG (zhu-chan) with structure of arrays (FASTEST)\n");
}

int main(int argc, char ** argv)
{
	int i, nx, ny;
	Fimage u;
	float lambda;
	int iter = 200;
	int method = 11;

	/* read input */
	if (argc <= 3)
	{
		fprintf(stderr, "%s - test Total Variation implementations in cuda\n", argv[0]);
		fprintf(stderr, "usage: %s lambda in.pgm out.pgm [method def=11] [iterations def=200]\n", argv[0]);
		fprintf(stderr, "methods are:\n");
		listmethods();
		exit(1);
	}
	if (argc >= 6) iter = atoi(argv[5]);
	if (argc >= 5) method = atoi(argv[4]);
	lambda = atof(argv[1]);
	u = read_pgm_fimage(argv[2]);
	nx = u->ncol; ny = u->nrow;

	/* run Method */
	float *input = u->gray;
	printf("tau = %f, lambda = %f, iterations = %d\n", 0.25, lambda, iter);
	if (method == 1){
		printf("reference implementation grad div\n");
		reflap(input, iter, nx, ny);
	}
	else if (method == 2){
		printf("reference implementation laplacian stencil\n");
		reflap2(input, iter, nx, ny);
	}
	else if (method == 3){
		printf("cuda implementation grad div\n");
		cudalap(input, iter, nx, ny);
	}
	else if (method == 4){
		printf("cuda implementation laplacian stencil\n");
		cudalap2(input, iter, nx, ny);
	}
	else if (method == 5){
		printf("reference implementation TV chambolle\n");
		refTVcha(input, iter, nx, ny, lambda);
	}
	else if (method == 6){
		printf("cuda TV chambolle\n");
		cudaTVcha(input, iter, nx, ny, lambda);
	}
	else if (method == 7){
		printf("cuda TV chambolle 2\n");
		cudaTVcha2(input, iter, nx, ny, lambda);
	}
	else if (method == 8){
		printf("cuda TV chambolle 2 shared memory\n");
		cudaTVcha2_shared(input, iter, nx, ny, lambda);
	}
	else if (method == 9){
		printf("cuda TV chambolle 2 with stencil\n");
		cudaTVcha2_stencil(input, iter, nx, ny, lambda);
	}
	else if (method == 10){
		printf("cuda TV chambolle 2 with interleaved data \n");
		cudaTVcha2_inter(input, iter, nx, ny, lambda);
	}
	else if (method == 11){
		printf("cuda TV chambolle 2 with structure of arrays (FASTER)\n");
		cudaTVcha2_structure_of_arrays(input, iter, nx, ny, lambda);
	}
	else if (method == 12){
		printf("cuda TV chambolle 2 dummy\n");
		cudaTVcha2_dummy(input, iter, nx, ny, lambda);
	}
	else if (method == 13){
		printf("cuda TV primal dual\n");
//		cudaTVpd(input, iter, nx, ny, lambda);
	}
	else if (method == 14){
		printf("cuda TV primal dual with structure of arrays\n");
//		cudaTVpd_SoA(input, iter, nx, ny, lambda);
	}
	else {
		printf("cuda TV primal dual : PDHG (zhu-chan) with structure of arrays (FASTEST)\n");
//		cudaTVpdhg_SoA(input, iter, nx, ny, lambda);
	}

	/* extract output */
	for (i = 0; i < nx*ny; i++) { u->gray[i] = (int)u->gray[i]; }
	write_pgm_fimage(u, argv[3]);

	del_fimage(u);
	return 0;
}

