/* This file is part of cudaTV.  See the file COPYING for further details. */
//#include <stdio.h>
#include <iostream>
//#include <cuda.h>
#include "cuda.h"
#include "cuda_runtime.h"

/*****************************
CUDA KERNELS
*****************************/

__global__ void gradientL(float *u, float *g, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	/*
	if  (idx<N)
	{
	g[2*idx+0] = 0;
	g[2*idx+1] = 0;
	}
	if ((idx< N) && px<(nx-1)) g[2*idx+0] = u[idx+1 ] - u[idx];
	if ((idx< N) && py<(ny-1)) g[2*idx+1] = u[idx+nx] - u[idx];
	*/
	if (px<nx && py<ny)
	{
		g[2 * idx + 0] = 0;
		g[2 * idx + 1] = 0;
		if (px<(nx - 1)) g[2 * idx + 0] = u[idx + 1] - u[idx];
		if (py<(ny - 1)) g[2 * idx + 1] = u[idx + nx] - u[idx];
	}
	//a[idx] =0;
}

__global__ void divergenceL(float *v, float *d, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;

	/*
	float AX = 0;
	if ((idx < N) && (px<(nx-1))) AX += v[2*(idx   )+0];
	if ((idx < N) && (px>0))      AX -= v[2*(idx-1 )+0];

	if ((idx < N) && (py<(ny-1))) AX += v[2*(idx   )+1];
	if ((idx < N) && (py>0))      AX -= v[2*(idx-nx)+1];

	if (idx < N)              d[idx] = AX;
	*/

	if(px<nx && py<ny)
	{
		float AX = 0;
		if((px<(nx - 1))) AX += v[2 * (idx)+0];
		if((px>0))      AX -= v[2 * (idx - 1) + 0];

		if((py<(ny - 1))) 
			AX += v[2 * (idx)+1];
		if((py>0))      
			AX -= v[2 * (idx - nx) + 1];

		d[idx] = AX;
	}
}


__global__ void lap(float *a, float *b, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x<nx && y<ny)
	{
		float AX = 0, BX = 0;
		if (x>0)   { BX += a[idx - 1]; AX++; }
		if (y>0)   { BX += a[idx - nx]; AX++; }
		if (x<nx - 1){ BX += a[idx + 1]; AX++; }
		if (y<ny - 1){ BX += a[idx + nx]; AX++; }
		b[idx] = -AX*a[idx] + BX;
	}
}

__global__ void add(float *a, float *b, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;
	if (x<nx && y<ny)   b[idx] += a[idx] * .125;

}

__global__ void square_array(float *a, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<N) a[idx] = a[idx] * a[idx];
}



/*****************************
END CUDA KERNELS
*****************************/




/* this is needed to interface with matlab mex */

/* cuda simple function */
void cudafunc(float *input, int N)
{
	float *cuda_mem;
	size_t size;

	size = N * sizeof(float);
	cudaMalloc((void **)&cuda_mem, size);
	cudaMemcpy(cuda_mem, input, size, cudaMemcpyHostToDevice);

	int block_size = 8;
	int n_blocks = N / block_size + (N%block_size == 0 ? 0 : 1);
	square_array << < n_blocks, block_size >> > (cuda_mem, N);

	cudaMemcpy(input, cuda_mem, size, cudaMemcpyDeviceToHost);
	cudaFree(cuda_mem);
}


/* cuda laplacian as gradient-divergence */
void cudalap(float *input, int it, int nx, int ny)
{
	float *cuda_mem_in;
	float *cuda_mem_g;
	float *cuda_mem_d;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	/* allocate device memory */
	cudaMalloc((void **)&cuda_mem_in, size);
	cudaMalloc((void **)&cuda_mem_g, 2 * size);
	cudaMalloc((void **)&cuda_mem_d, size);

	/* Copy input to device*/
	cudaMemcpy(cuda_mem_in, input, size, cudaMemcpyHostToDevice);

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/* setup a 2D thread grid, with 16x16 blocks */
	/* each block is will use nearby memory*/
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/* call the functions */
	for (j = 0; j<it; j++) {
		gradientL << < n_blocks, block_size >> > (cuda_mem_in, cuda_mem_g, nx, ny);
		divergenceL << < n_blocks, block_size >> > (cuda_mem_g, cuda_mem_d, nx, ny);
		add << < n_blocks, block_size >> > (cuda_mem_d, cuda_mem_in, nx, ny);
		//  cudaMemcpy(tmp, cuda_mem_d, size, cudaMemcpyDeviceToHost);
		//for (i=0;i<N;i++) printf(":%g\n", tmp[i]);
	}

	/* recover the output from device memory*/
	cudaMemcpy(input, cuda_mem_in, size, cudaMemcpyDeviceToHost);

	/* free device memory */
	cudaFree(cuda_mem_in);
	cudaFree(cuda_mem_d);
	cudaFree(cuda_mem_g);
}


/* cuda laplacian as stencil*/
void cudalap2(float *input, int it, int nx, int ny)
{
	float *cuda_mem_in;
	float *cuda_mem_d;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	/* allocate device memory */
	cudaMalloc((void **)&cuda_mem_in, size);
	cudaMalloc((void **)&cuda_mem_d, size);
	cudaMemcpy(cuda_mem_in, input, size, cudaMemcpyHostToDevice);

	/* Copy input to device*/
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/* call the functions */
	for (j = 0; j<it; j++) {
		lap << < n_blocks, block_size >> > (cuda_mem_in, cuda_mem_d, nx, ny);
		add << < n_blocks, block_size >> > (cuda_mem_d, cuda_mem_in, nx, ny);
	}

	/* recover the output from device memory*/
	cudaMemcpy(input, cuda_mem_in, size, cudaMemcpyDeviceToHost);

	/* free device memory */
	cudaFree(cuda_mem_in);
	cudaFree(cuda_mem_d);
}

