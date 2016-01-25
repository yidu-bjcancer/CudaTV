/* This file is part of cudaTV.  See the file COPYING for further details. */
//#include <stdio.h>
#include <iostream>
//#include <cuda.h>
#include "cuda.h"
#include "cuda_runtime.h"


/*****************************
CUDA KERNELS
*****************************/



// f=  (1 - tu*lambda) * f_old + tauu .* div(z) + tauu*lambda.*g;
__global__ void updF(float *f, float *z, float *g, float tf, float lambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((px<(nx - 1))) DIVZ += z[2 * (idx)+0];
		if ((px>0))      DIVZ -= z[2 * (idx - 1) + 0];

		if ((py<(ny - 1))) DIVZ += z[2 * (idx)+1];
		if ((py>0))      DIVZ -= z[2 * (idx - nx) + 1];

		// update f
		//f[idx] = (1.-tf*lambda)*f[idx] + tf * DIVZ + tf*lambda*g[idx];
		f[idx] = (f[idx] + tf * DIVZ + tf*lambda*g[idx]) / (1 + tf*lambda);
	}

}


// z= zold + tauz.* grad(u);	
// and normalize z:  n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) + beta) ); z/=n;
__global__ void updZ(float *z, float *f, float tz, float beta, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float a, b, t;

	if (px<nx && py<ny)
	{
		// compute the gradient
		a = 0;
		b = 0;
		if (px<(nx - 1)) a = f[idx + 1] - f[idx];
		if (py<(ny - 1)) b = f[idx + nx] - f[idx];

		// update z

		a = z[2 * idx + 0] + tz*a;
		b = z[2 * idx + 1] + tz*b;

		t = sqrtf(beta + a*a + b*b);
		t = t<1. ? 1. : 1. / t;

		z[2 * idx + 0] = a*t;
		z[2 * idx + 1] = b*t;
	}
}


/*******************************************************/

// f=  (1 - tu*lambda) * f_old + tauu .* div(z) + tauu*lambda.*g;
__global__ void updF_SoA(float *f, float *z1, float *z2, float *g, float tf, float lambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((px<(nx - 1))) DIVZ += z1[idx];
		if ((px>0))      DIVZ -= z1[idx - 1];

		if ((py<(ny - 1))) DIVZ += z2[idx];
		if ((py>0))      DIVZ -= z2[idx - nx];

		// update f
		//f[idx] = (1.-tf*lambda)*f[idx] + tf * DIVZ + tf*lambda*g[idx];
		f[idx] = (f[idx] + tf * DIVZ + tf*lambda*g[idx]) / (1 + tf*lambda);
	}

}


// z= zold + tauz.* grad(u);	
// and normalize z:  n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) + beta) ); 
// z/=n;
__global__ void updZ_SoA(float *z1, float *z2, float *f, float tz, float beta, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;

	if (px<nx && py<ny)
	{
		// compute the gradient
		float a = 0;
		float b = 0;
		float fc = f[idx];
		if (px<(nx - 1)) a = f[idx + 1] - fc;
		if (py<(ny - 1)) b = f[idx + nx] - fc;

		// update z

		a = z1[idx] + tz*a;
		b = z2[idx] + tz*b;

		float t = 0;
		t = sqrtf(beta + a*a + b*b);
		t = t<1. ? 1. : 1. / t;
		/*
		float t = 0;
		t = sqrtf(a*a+b*b);
		t=t<0.00001?0.:1./t;
		*/

		z1[idx] = a*t;
		z2[idx] = b*t;
	}
}






/*******************************************************/



// u=  (1 - tu) * uold + tu .* ( f + 1/lambda*div(z) );
__global__ void updhgF_SoA(float *f, float *z1, float *z2, float *g, float tf, float invlambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((px<(nx - 1))) DIVZ += z1[idx];
		if ((px>0))      DIVZ -= z1[idx - 1];

		if ((py<(ny - 1))) DIVZ += z2[idx];
		if ((py>0))      DIVZ -= z2[idx - nx];

		// update f
		f[idx] = (1 - tf) *f[idx] + tf * (g[idx] + invlambda*DIVZ);
	}

}


// z= zold + tz*lambda* grad(u);	
// and normalize z:  
//n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) ) ); 
// z= z/n;
__global__ void updhgZ_SoA(float *z1, float *z2, float *f, float tz, float lambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;

	if (px<nx && py<ny)
	{
		// compute the gradient
		float a = 0;
		float b = 0;
		float fc = f[idx];
		if (px<(nx - 1)) a = f[idx + 1] - fc;
		if (py<(ny - 1)) b = f[idx + nx] - fc;

		// update z

		a = z1[idx] + tz*lambda*a;
		b = z2[idx] + tz*lambda*b;

		// project
		float t = 0;
		t = sqrtf(a*a + b*b);
		t = (t <= 1 ? 1. : t);

		z1[idx] = a / t;
		z2[idx] = b / t;
	}
}





/*****************************
END CUDA KERNELS
*****************************/



/* this is needed to interface with matlab mex */


static void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}





/* cuda primal dual TV implementation */
/*	formulated by: Zhu Chan - PDHG algorithm*/
void cudaTVpd(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z;
	float *cuda_f;
	size_t size;
	int j;
	float tz = .25, tf = .25, beta = 0.0001;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/* allocate device memory */
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_f, size);
	cudaMalloc((void **)&cuda_z, 2 * size);

	/* Copy input to device*/
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_f, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_z, 0, 2 * size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/* setup a 2D thread grid, with 16x16 blocks */
	/* each block is will use nearby memory*/
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/* call the functions */
	for (j = 0; j < it; j++) {

		// z= zold + tauz.* grad(u);	
		// and normalize z:  n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) + beta) ); z/=n;
		updZ << < n_blocks, block_size >> > (cuda_z, cuda_f, tz, beta, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		// u=  (1 - tauu*lambda) * uold + tauu .* div(z) + tauu*lambda.*f;
		updF << < n_blocks, block_size >> > (cuda_f, cuda_z, cuda_g, tf, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	/* recover the output from device memory*/
	cudaMemcpy(input, cuda_f, size, cudaMemcpyDeviceToHost);

	/* free device memory */
	cudaFree(cuda_f);
	cudaFree(cuda_z);
	cudaFree(cuda_g);
}

/* cuda primal dual TV implementation */
void cudaTVpd_SoA(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z1;
	float *cuda_z2;
	float *cuda_f;
	size_t size;
	int j;
	float tz = 2, tf = .2, beta = 0.0001;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/* allocate device memory */
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_f, size);
	cudaMalloc((void **)&cuda_z1, size);
	cudaMalloc((void **)&cuda_z2, size);

	/* Copy input to device*/
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_f, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_z1, 0, size);
	cudaMemset(cuda_z2, 0, size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/* setup a 2D thread grid, with 16x16 blocks */
	/* each block is will use nearby memory*/
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/* call the functions */
	for (j = 0; j < it; j++) {

		// z= zold + tauz.* grad(u);	
		// and normalize z:  n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) + beta) ); z/=n;
		updZ_SoA << < n_blocks, block_size >> > (cuda_z1, cuda_z2, cuda_f, tz, beta, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		// u=  (1 - tauu*lambda) * uold + tauu .* div(z) + tauu*lambda.*f;
		updF_SoA << < n_blocks, block_size >> > (cuda_f, cuda_z1, cuda_z2, cuda_g, tf, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	/* recover the output from device memory*/
	cudaMemcpy(input, cuda_f, size, cudaMemcpyDeviceToHost);

	/* free device memory */
	cudaFree(cuda_f);
	cudaFree(cuda_z1);
	cudaFree(cuda_z2);
	cudaFree(cuda_g);
}


/* cuda primal dual TV implementation
PDHG - zhu chan*/
void cudaTVpdhg_SoA(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z1;
	float *cuda_z2;
	float *cuda_f;
	size_t size;
	int j;
	float tz = 2, tf = .2; /* default step sizes */

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/* allocate device memory */
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_f, size);
	cudaMalloc((void **)&cuda_z1, size);
	cudaMalloc((void **)&cuda_z2, size);

	/* Copy input to device*/
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_f, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_z1, 0, size);
	cudaMemset(cuda_z2, 0, size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/* setup a 2D thread grid, with 16x16 blocks */
	/* each block is will use nearby memory*/
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/* call the functions */
	for (j = 0; j < it; j++) {
		/* adaptive step sizes: from the paper */
		tz = 0.2 + 0.08*j;
		tf = (0.5 - 5. / (15 + j)) / tz;

		// z= zold + tz*lambda* grad(u);	
		// and normalize z:  
		//n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) ) ); 
		// z= z/n;
		updhgZ_SoA << < n_blocks, block_size >> > (cuda_z1, cuda_z2, cuda_f, tz, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		// u=  (1 - tu) * uold + tu .* ( f - 1/lambda*div(z) );
		updhgF_SoA << < n_blocks, block_size >> > (cuda_f, cuda_z1, cuda_z2, cuda_g, tf, lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	/* recover the output from device memory*/
	cudaMemcpy(input, cuda_f, size, cudaMemcpyDeviceToHost);

	/* free device memory */
	cudaFree(cuda_f);
	cudaFree(cuda_z1);
	cudaFree(cuda_z2);
	cudaFree(cuda_g);
}
/* END EXTERN C */

