/* This file is part of cudaTV.  See the file COPYING for further details. */
/*
#include <stdio.h>
#include <math.h>
*/

#include <iostream>


void diver(float *z, float * d, int nx, int ny)
{
	double a, b;
	int i, j, adr;

	adr = 0;
	for (j = 0; j < ny; j++) {
		for (i = 0; i < nx; i++)
		{

			if (i == 0) 			a = z[2 * adr];
			else if (i == nx - 1) a = -z[2 * (adr - 1)];
			else 					a = z[2 * adr] - z[2 * (adr - 1)];

			if (j == 0) 			b = z[(2 * adr) + 1];
			else if (j == ny - 1) b = -z[2 * (adr - nx) + 1];
			else 					b = z[2 * adr + 1] - z[2 * (adr - nx) + 1];

			d[adr] = a + b;
			adr++;
		}
	}
}
void nabla(float *u, float *g, int nx, int ny)
{
	int i, j, adr;

	adr = 0;
	for (j = 0; j < ny; j++)
		for (i = 0; i < nx; i++)
		{
			if (i == (nx - 1)) 	g[2 * adr + 0] = 0;
			else 					g[2 * adr + 0] = u[adr + 1] - u[adr];

			if (j == (ny - 1)) 	g[2 * adr + 1] = 0;
			else 					g[2 * adr + 1] = u[adr + nx] - u[adr];
			adr++;
		}
}
void lapla(float *a, float *b, int nx, int ny)
{
	int x, y, idx = 0;
	for (y = 0; y < ny; y++) for (x = 0; x < nx; x++)
	{
		float AX = 0, BX = 0;
		if (x > 0)   { BX += a[idx - 1]; AX++; }
		if (y > 0)   { BX += a[idx - nx]; AX++; }
		if (x < nx - 1){ BX += a[idx + 1]; AX++; }
		if (y < ny - 1){ BX += a[idx + nx]; AX++; }
		b[idx] = -AX*a[idx] + BX;
		idx++;
	}
}

/* reference laplacian as gradient divergence */
void reflap(float *input, int it, int nx, int ny)
{
	float *mem_in;
	float *mem_g;
	float *mem_d;
	size_t size;
	int N = nx*ny;
	int i, j;


	size = nx*ny * sizeof(float);
	mem_in = input;
	mem_g = (float*)malloc(2 * size);
	mem_d = (float*)malloc(size);

	for (j = 0; j < it; j++) {
		nabla(mem_in, mem_g, nx, ny);
		diver(mem_g, mem_d, nx, ny);
		for (i = 0; i < N; i++) mem_in[i] += mem_d[i] * .125;
	}


	free(mem_d);
	free(mem_g);
}


/* reference laplacian as stencil*/
void reflap2(float *input, int it, int nx, int ny)
{
	float *di;
	size_t size;
	int i, j;
	int N = nx*ny;

	size = nx*ny * sizeof(float);
	di = (float*)malloc(size);

	for (j = 0; j < it; j++) {
		lapla(input, di, nx, ny);
		for (i = 0; i < N; i++) input[i] += di[i] * .125;
	}


	free(di);
}

/* reference chambolle */
#define SWP(a,b) {float *swap=a;a=b;b=swap;}
void refTVcha(float *input, int it, int nx, int ny, float lambda)
{
	float *g, *z, *z0, *div, *div0;
	size_t size;
	int N = nx*ny;
	int i, j;
	float alpha;
	float tau = 0.125;


	size = nx*ny * sizeof(float);
	g = input;
	div = (float*)malloc(size);
	div0 = (float*)malloc(size);
	z = (float*)malloc(2 * size);
	z0 = (float*)malloc(2 * size);

	for (i = 0; i < N; i++) { div0[i] = 0; }

	for (j = 0; j < it; j++) {
		/* div contient*/
		for (i = 0; i < N; i++) { div[i] = div0[i] - g[i] / lambda; }

		/*tau*lambda*Q^-1(div(z))+Q^-1(b) */
		nabla(div, z, nx, ny); /* zx et zy sont en fait nabla(div+.) */
		for (i = 0; i < N; i++)
		{
			int ix = 2 * i + 0;
			int iy = 2 * i + 1;
			alpha = 1.0 / (1.0 + tau*sqrtf(z[ix] * z[ix] + z[iy] * z[iy]));

			z[ix] = (z0[ix] + tau*z[ix])*alpha;  /* point fixe de Chambolle */
			z[iy] = (z0[iy] + tau*z[iy])*alpha;
		}

		diver(z, div, nx, ny); /* div contient le vrai div */
		SWP(z, z0);
		SWP(div, div0);

	}/* div contient*/

	for (i = 0; i < N; i++) { input[i] = g[i] - div0[i] * lambda; }

	free(div);
	free(div0);
	free(z);
	free(z0);
}

