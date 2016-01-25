/* This file is part of cudaTV.  See the file COPYING for further details. */
//#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*****************************
CUDA KERNELS
*****************************/

__global__ void gradient(float *u, float *g, int nx, int ny)
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

__global__ void divergence(float *v, float *d, int nx, int ny)
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

	if (px<nx && py<ny)
	{
		float AX = 0;
		if ((px<(nx - 1))) AX += v[2 * (idx)+0];
		if ((px>0))      AX -= v[2 * (idx - 1) + 0];

		if ((py<(ny - 1))) AX += v[2 * (idx)+1];
		if ((py>0))      AX -= v[2 * (idx - nx) + 1];

		d[idx] = AX;
	}
}





/* div = div0 - g/lambda*/
__global__ void sub0(float *div0, float *div, float *g, float lambda, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;
	if (x<nx && y<ny)   div[idx] = div0[idx] - g[idx] / lambda;
}

/*z = (z0 + tau z)/(1+tau|z|)*/
__global__ void zupdate(float *z, float *z0, float tau, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;
	if (x<nx && y<ny)
	{
		float a = z[2 * idx + 0];
		float b = z[2 * idx + 1];
		float t = 1 / (1 + tau*sqrtf(a*a + b*b));
		z[2 * idx + 0] = (z0[2 * idx + 0] + tau*z[2 * idx + 0])*t;
		z[2 * idx + 1] = (z0[2 * idx + 1] + tau*z[2 * idx + 1])*t;
	}
}

/* f = div z - g*invlambda*/
__global__ void fupdate(float *f, float *z, float *g, float invlambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if (!(px == (nx - 1))) DIVZ += z[2 * (idx)+0];
		if (!(py == (ny - 1))) DIVZ += z[2 * (idx)+1];
		if (!(px == 0))      DIVZ -= z[2 * (idx - 1) + 0];
		if (!(py == 0))      DIVZ -= z[2 * (idx - nx) + 1];

		// update f
		f[idx] = DIVZ - g[idx] * invlambda;
	}
}


/*z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)*/
__global__ void zupdate2(float *z, float *f, float tau, int nx, int ny)
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
		float fc = f[idx];
		if (!(px == (nx - 1))) a = f[idx + 1] - fc;
		if (!(py == (ny - 1))) b = f[idx + nx] - fc;

		// update z
		t = 1 / (1 + tau*sqrtf(a*a + b*b));
		z[2 * idx + 0] = (z[2 * idx + 0] + tau*a)*t;
		z[2 * idx + 1] = (z[2 * idx + 1] + tau*b)*t;
	}
}

/* u = -f * lambda*/
__global__ void solution(float *f, float lambda, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x<nx && y<ny)   f[idx] = -f[idx] * lambda;
}


const int BLOCK_SIZE_x = 32;
const int BLOCK_SIZE_y = 8;
__shared__ float F[BLOCK_SIZE_x*BLOCK_SIZE_y];
__shared__ float G[BLOCK_SIZE_x*BLOCK_SIZE_y];
__shared__ float Z1[BLOCK_SIZE_x*BLOCK_SIZE_y];
__shared__ float Z2[BLOCK_SIZE_x*BLOCK_SIZE_y];

const int aprx = 4;
const int apry = 1;

/*z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)*/
__global__ void zupdate3(float *z1, float *z2, float *f, float tau, int nx, int ny)
{
	int px = blockIdx.x * (blockDim.x - 2 * aprx) + threadIdx.x;
	int py = blockIdx.y * (blockDim.y - 2 * apry) + threadIdx.y;
	int idx = px + py*nx;
	float a, b, t;

	// SHARED MEMORY LOAD 
	int thidx = threadIdx.x + blockDim.x*threadIdx.y;

	F[thidx] = f[idx];

	__syncthreads();

	if (threadIdx.x >= aprx && threadIdx.x<BLOCK_SIZE_x - aprx && threadIdx.y >= apry && threadIdx.y<BLOCK_SIZE_y - apry)
		if (px<nx && py<ny)
		{
			// compute the gradient
			a = 0;
			b = 0;
			//if (px<(nx-1)) a = f[idx+1 ] - f[idx];
			//if (py<(ny-1)) b = f[idx+nx] - f[idx];
			float fc = F[thidx];
			float fr = F[thidx + 1];
			float fu = F[thidx + blockDim.x];
			if (px<(nx - 1)) a = fr - fc;
			if (py<(ny - 1)) b = fu - fc;

			// update z
			t = 1 / (1 + tau*sqrtf(a*a + b*b));
			z1[idx] = (z1[idx] + tau*a)*t;
			z2[idx] = (z2[idx] + tau*b)*t;
		}

}

/* f = div z - g*invlambda*/
__global__ void fupdate2(float *f, float *z1, float *z2, float *g, float invlambda, int nx, int ny)
{

	//TODO THIS MEMORY IS NOT OK FOR BACKWARD DIFFERENCES
	int px = blockIdx.x * (blockDim.x - 2 * aprx) + threadIdx.x;
	int py = blockIdx.y * (blockDim.y - 2 * apry) + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	// SHARED MEMORY LOAD 
	int thidx = threadIdx.x + blockDim.x*threadIdx.y;
	Z1[thidx] = z1[idx];
	Z2[thidx] = z2[idx];

	__syncthreads();

	if (threadIdx.x >= aprx && threadIdx.x<BLOCK_SIZE_x - aprx && threadIdx.y >= apry && threadIdx.y<BLOCK_SIZE_y - apry)
		if (px<nx && py<ny)
		{
			// compute the divergence
			DIVZ = 0;
			if ((px<(nx - 1))) DIVZ += Z1[(thidx)];
			if ((px>0))      DIVZ -= Z1[(thidx - 1)];

			if ((py<(ny - 1))) DIVZ += Z2[(thidx)];
			if ((py>0))      DIVZ -= Z2[(thidx - blockDim.x)];

			// update f
			f[idx] = DIVZ - g[idx] * invlambda;
		}
}


/* u = -f * lambda*/
__global__ void solution2(float *f, float lambda, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;


	if (x<nx && y<ny)   f[idx] = -f[idx] * lambda;
}

/* f = div z - g*(1/lambda)*/
/*z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)*/


/* f = div z - g*(1/lambda)*/
/* u = -f * lambda*/
/* u = -div z *lambda + g */
__global__ void solution_stencil(float *zx, float * zy, float *g, float lambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;

	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((px<(nx - 1))) DIVZ += zx[(idx)];
		if ((px>0))      DIVZ -= zx[(idx - 1)];

		if ((py<(ny - 1))) DIVZ += zy[(idx)];
		if ((py>0))      DIVZ -= zy[(idx - nx)];

		// update f
		g[idx] = -DIVZ*lambda + g[idx];
	}
}


/* f = div z - g*(1/lambda)*/
/* z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)*/
__global__ void zupdate_stencil(float *zx, float *zy, float *zoutx, float *zouty, float *g, float tau, float invlambda, int  nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	int tidx, tpx, tpy;
	float a, b, t;
	float DIVZ;

	/* compute simultaneously
	f= div z -g /lambda at the positions
	right, center north*/
	float fr = 0, fc = 0, fu = 0;

	////////////////////////////////////////////////////////
	//
	//		(zul)		(zu)	
	//
	//					____
	//		(zl)		|zc|		(zr)
	//					----
	//
	//					(zd)		(zdr)	
	//
	//		if the pixel is not inside the region then put 0
	//
	//		fc = z1c - z1l + z2c - z2d
	//		fr = z1r - z1c + z2r - z2dr
	//		fu = z1u - z1ul + z2u - z2c
	//
	////////////////////////////////////////////////////////

	tidx = idx;
	tpx = px;
	tpy = py;
	if (tpx<nx && tpy<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((tpx<(nx - 1))) DIVZ += zx[tidx];
		if ((tpx>0))      DIVZ -= zx[tidx - 1];

		if ((tpy<(ny - 1))) DIVZ += zy[tidx];
		if ((tpy>0))      DIVZ -= zy[tidx - nx];

		fc = DIVZ;
	}
	////////////////////////////////////////////////////////

	tidx = idx + 1;
	tpx = px + 1;
	tpy = py;
	if (tpx<nx && tpy<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((tpx<(nx - 1))) DIVZ += zx[tidx];
		if ((tpx>0))      DIVZ -= zx[tidx - 1];

		if ((tpy<(ny - 1))) DIVZ += zy[tidx];
		if ((tpy>0))      DIVZ -= zy[tidx - nx];

		fr = DIVZ;
	}
	////////////////////////////////////////////////////////

	tidx = idx + nx;
	tpx = px;
	tpy = py + 1;
	if (tpx<nx && tpy<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((tpx<(nx - 1))) DIVZ += zx[tidx];
		if ((tpx>0))      DIVZ -= zx[tidx - 1];

		if ((tpy<(ny - 1))) DIVZ += zy[tidx];
		if ((tpy>0))      DIVZ -= zy[tidx - nx];

		fu = DIVZ;
	}

	fr = fr - g[idx + 1] * invlambda;
	fc = fc - g[idx] * invlambda;
	fu = fu - g[idx + nx] * invlambda;

	////////////////////////////////////////////////////////

	if (px<nx && py<ny)
	{
		// compute the gradient
		a = 0;
		b = 0;
		if (px<(nx - 1)) a = fr - fc;
		if (py<(ny - 1)) b = fu - fc;

		// update z
		t = 1 / (1 + tau*sqrtf(a*a + b*b));
		zoutx[idx] = (zx[idx] + tau*a)*t;
		zouty[idx] = (zy[idx] + tau*b)*t;
	}
}










/* f = div z - g*invlambda*/
__global__ void fupdate_inter(float *z, float *g, float invlambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		if ((px<(nx - 1))) DIVZ += z[3 * (idx)+0];
		if ((px>0))      DIVZ -= z[3 * (idx - 1) + 0];

		if ((py<(ny - 1))) DIVZ += z[3 * (idx)+1];
		if ((py>0))      DIVZ -= z[3 * (idx - nx) + 1];

		// update f
		z[3 * idx + 2] = DIVZ - g[idx] * invlambda;
	}
}


/*z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)*/
__global__ void zupdate_inter(float *z, float tau, int nx, int ny)
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
		if (px<(nx - 1)) a = z[3 * (idx + 1) + 2] - z[3 * idx + 2];
		if (py<(ny - 1)) b = z[3 * (idx + nx) + 2] - z[3 * idx + 2];

		// update z
		t = 1 / (1 + tau*sqrtf(a*a + b*b));
		z[3 * idx + 0] = (z[3 * idx + 0] + tau*a)*t;
		z[3 * idx + 1] = (z[3 * idx + 1] + tau*b)*t;
	}
}

/* u = -f * lambda*/
__global__ void solution_inter(float *z, float *g, float lambda, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x<nx && y<ny)   g[idx] = -z[3 * idx + 2] * lambda;
}





/* f = div z - g*invlambda*/
__global__ void fupdate_SoA(float *f, float *z1, float *z2, float *g, float invlambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		float Z1c = z1[(idx)];
		float Z2c = z2[(idx)];
		float Z1l = z1[(idx - 1)];
		float Z2d = z2[(idx - nx)];
		if (!(px == (nx - 1))) DIVZ += Z1c;
		if (!(px == 0))      DIVZ -= Z1l;
		if (!(py == (ny - 1))) DIVZ += Z2c;
		if (!(py == 0))      DIVZ -= Z2d;

		// update f
		f[idx] = DIVZ - g[idx] * invlambda;
	}
}


/*z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)*/
__global__ void zupdate2_SoA(float *z1, float *z2, float *f, float tau, int nx, int ny)
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
		float fc = f[idx];
		float fr = f[idx + 1];
		float fu = f[idx + nx];
		if (!(px == (nx - 1))) a = fr - fc;
		if (!(py == (ny - 1))) b = fu - fc;

		// update z
		t = 1 / (1 + tau*sqrtf(a*a + b*b));
		z1[idx] = (z1[idx] + tau*a)*t;
		z2[idx] = (z2[idx] + tau*b)*t;
	}
}

/* u = -f * lambda*/
__global__ void solution_SoA(float *f, float lambda, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x<nx && y<ny)   f[idx] = -f[idx] * lambda;
}



/* f = div z - g*invlambda*/
__global__ void fupdate_dummy(float *f, float *z1, float *z2, float *g, float invlambda, int nx, int ny)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny)
	{
		// compute the divergence
		DIVZ = 0;
		float Z1c = z1[(idx)];
		float Z2c = z2[(idx)];
		//	float Z1l=z1[(idx-1 )];
		//	float Z2d=z2[(idx-nx)];
		if (!(px == (nx - 1))) DIVZ += Z1c;
		//  if (!(px==0))      DIVZ -= Z1l;
		if (!(py == (ny - 1))) DIVZ += Z2c;
		//  if (!(py==0))      DIVZ -= Z2d;

		// update f
		f[idx] = DIVZ - g[idx] * invlambda;
	}
}


/*z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)*/
__global__ void zupdate2_dummy(float *z1, float *z2, float *f, float tau, int nx, int ny)
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
		float fc = f[idx];
		//		float fr=f[idx+1];
		//		float fu=f[idx+nx];
		//    if (!(px==(nx-1))) a = fr - fc;
		//     if (!(py==(ny-1))) b = fu - fc;
		a = fc;
		b = fc;

		// update z
		t = 1 / (1 + tau*sqrtf(a*a + b*b));
		z1[idx] = (z1[idx] + tau*a)*t;
		z2[idx] = (z2[idx] + tau*b)*t;
	}
}

/* u = -f * lambda*/
__global__ void solution_dummy(float *f, float lambda, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x<nx && y<ny)   f[idx] = -f[idx] * lambda;
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



/* cuda chambolle's TV implementation 2x faster then CPU */
void cudaTVcha(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z;
	float *cuda_z0;
	float *cuda_div;
	float *cuda_div0;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/// * allocate device memory * /
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_div, size);
	cudaMalloc((void **)&cuda_div0, size);
	cudaMalloc((void **)&cuda_z, 2 * size);
	cudaMalloc((void **)&cuda_z0, 2 * size);

	/// * Copy input to device* /
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_div, 0, size);
	cudaMemset(cuda_div0, 0, size);
	cudaMemset(cuda_z, 0, 2 * size);
	cudaMemset(cuda_z0, 0, 2 * size);

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/// * setup a 2D thread grid, with 16x16 blocks * /
	/// * each block is will use nearby memory* /
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/// * call the functions * /
	for (j = 0; j<it; j++) {
		/// * div = div0 - g/lambda* /
		sub0 << < n_blocks, block_size >> > (cuda_div0, cuda_div, cuda_g, lambda, nx, ny);

		gradient << < n_blocks, block_size >> > (cuda_div, cuda_z, nx, ny);

		/// *z = (z0 + tau z)/(1+tau|z|)* /
		zupdate << < n_blocks, block_size >> > (cuda_z, cuda_z0, 0.125, nx, ny);

		divergence << < n_blocks, block_size >> > (cuda_z, cuda_div, nx, ny);

		/// * SWAPs * /
		float *tmp;
		tmp = cuda_div;
		cuda_div = cuda_div0;
		cuda_div0 = tmp;

		tmp = cuda_z;
		cuda_z = cuda_z0;
		cuda_z0 = tmp;
	}

	/// *div (SOLUTION) = g -div0*lambda;	* /
	sub0 << < n_blocks, block_size >> > (cuda_g, cuda_div, cuda_div0, 1.0 / lambda, nx, ny);

	/// * recover the output from device memory* /
	cudaMemcpy(input, cuda_div, size, cudaMemcpyDeviceToHost);

	/// * free device memory * /
	cudaFree(cuda_g);
	cudaFree(cuda_div);
	cudaFree(cuda_div0);
	cudaFree(cuda_z);
	cudaFree(cuda_z0);
}

/* cuda chambolle's TV implementation 4x faster then CPU */
void cudaTVcha2(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z;
	float *cuda_f;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/// * allocate device memory * /
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_f, size);
	cudaMalloc((void **)&cuda_z, 2 * size);

	/// * Copy input to device* /
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_f, 0, size);
	cudaMemset(cuda_z, 0, 2 * size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/// * setup a 2D thread grid, with 16x16 blocks * /
	/// * each block is will use nearby memory* /
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/// * call the functions * /
	for (j = 0; j<it; j++) {
		/// * f = div z - g*(1/lambda)* /
		fupdate << < n_blocks, block_size >> > (cuda_f, cuda_z, cuda_g, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		/// *z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)* /
		zupdate2 << < n_blocks, block_size >> > (cuda_z, cuda_f, 0.125, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	checkCUDAError("memcpy");

	/// * u = -f * lambda* /
	solution << < n_blocks, block_size >> > (cuda_f, lambda, nx, ny);

	/// * recover the output from device memory* /
	cudaMemcpy(input, cuda_f, size, cudaMemcpyDeviceToHost);

	/// * free device memory * /
	cudaFree(cuda_f);
	cudaFree(cuda_z);
	cudaFree(cuda_g);
}

/// * cuda chambolle's TV implementation 4x faster then CPU* /
void cudaTVcha2_structure_of_arrays(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z1;
	float *cuda_z2;
	float *cuda_f;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/// * allocate device memory * /
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_f, size);
	cudaMalloc((void **)&cuda_z1, size);
	cudaMalloc((void **)&cuda_z2, size);

	/// * Copy input to device* /
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_f, 0, size);
	cudaMemset(cuda_z1, 0, size);
	cudaMemset(cuda_z2, 0, size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/// * setup a 2D thread grid, with 16x16 blocks * /
	/// * each block is will use nearby memory* /
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/// * call the functions * /
	for (j = 0; j<it; j++) {
		/// * f = div z - g*(1/lambda)* /
		fupdate_SoA << < n_blocks, block_size >> > (cuda_f, cuda_z1, cuda_z2, cuda_g, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		/// *z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)* /
		zupdate2_SoA << < n_blocks, block_size >> > (cuda_z1, cuda_z2, cuda_f, 0.125, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	checkCUDAError("memcpy");

	/// * u = -f * lambda* /
	solution_SoA << < n_blocks, block_size >> > (cuda_f, lambda, nx, ny);

	/// * recover the output from device memory* /
	cudaMemcpy(input, cuda_f, size, cudaMemcpyDeviceToHost);

	/// * free device memory * /
	cudaFree(cuda_f);
	cudaFree(cuda_z1);
	cudaFree(cuda_z2);
	cudaFree(cuda_g);
}

/// * cuda chambolle's TV implementation 4x faster then CPU* /
void cudaTVcha2_shared(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z1;
	float *cuda_z2;
	float *cuda_f;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/// * allocate device memory * /
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_f, size);
	cudaMalloc((void **)&cuda_z1, size);
	cudaMalloc((void **)&cuda_z2, size);

	/// * Copy input to device* /
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_f, 0, size);
	cudaMemset(cuda_z1, 0, size);
	cudaMemset(cuda_z2, 0, size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/// * setup a 2D thread grid, with 16x16 blocks * /
	/// * each block is will use nearby memory* /
	dim3 block_size(BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 n_blocks((nx + block_size.x - aprx * 2 - 1) / (block_size.x - aprx * 2),
		(ny + block_size.y - aprx * 2 - 1) / (block_size.y - apry * 2));

	/// * call the functions * /
	for (j = 0; j<it; j++) {
		/// * f = div z - g*(1/lambda)* /
		fupdate2 << < n_blocks, block_size >> > (cuda_f, cuda_z1, cuda_z2, cuda_g, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		/// *z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)* /
		zupdate3 << < n_blocks, block_size >> > (cuda_z1, cuda_z2, cuda_f, 0.125, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	checkCUDAError("memcpy");

	/// * u = -f * lambda* /
	solution2 << < n_blocks, block_size >> > (cuda_f, lambda, nx, ny);

	/// * recover the output from device memory* /
	cudaMemcpy(input, cuda_f, size, cudaMemcpyDeviceToHost);

	/// * free device memory * /
	cudaFree(cuda_f);
	cudaFree(cuda_z1);
	cudaFree(cuda_z2);
	cudaFree(cuda_g);
}

/// * cuda chambolle's TV implementation 4x faster then CPU* /
void cudaTVcha2_stencil(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z1x;
	float *cuda_z1y;
	float *cuda_z2x;
	float *cuda_z2y;
	float *tmp;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/// * allocate device memory * /
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_z1x, size);
	cudaMalloc((void **)&cuda_z2x, size);
	cudaMalloc((void **)&cuda_z1y, size);
	cudaMalloc((void **)&cuda_z2y, size);

	/// * Copy input to device* /
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_z1x, 0, size);
	cudaMemset(cuda_z2x, 0, size);
	cudaMemset(cuda_z1y, 0, size);
	cudaMemset(cuda_z2y, 0, size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/// * setup a 2D thread grid, with 16x16 blocks * /
	/// * each block is will use nearby memory* /
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/// * call the functions * /
	for (j = 0; j<it; j++) {
		/// * f = div z - g*(1/lambda)* /
		/// *z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)* /
		zupdate_stencil << < n_blocks, block_size >> > (cuda_z1x, cuda_z1y, cuda_z2x, cuda_z2y, cuda_g, 0.125, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		// swap
		tmp = cuda_z1x;
		cuda_z1x = cuda_z2x;
		cuda_z2x = tmp;

		tmp = cuda_z1y;
		cuda_z1y = cuda_z2y;
		cuda_z2y = tmp;
	}

	checkCUDAError("memcpy");

	/// * f = div z - g*(1/lambda)* /
	/// * u = -f * lambda* /
	/// * u = -div z *lambda + g * /
	solution_stencil << < n_blocks, block_size >> > (cuda_z1x, cuda_z1y, cuda_g, lambda, nx, ny);

	/// * recover the output from device memory* /
	cudaMemcpy(input, cuda_g, size, cudaMemcpyDeviceToHost);

	/// * free device memory * /
	cudaFree(cuda_z1x);
	cudaFree(cuda_z2x);
	cudaFree(cuda_z1y);
	cudaFree(cuda_z2y);
	cudaFree(cuda_g);
}

/// * cuda chambolle's TV implementation with INTERLEAVED DATA SLOW* /
void cudaTVcha2_inter(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/// * allocate device memory * /
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_z, 3 * size);

	/// * Copy input to device* /
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_z, 0, 3 * size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/// * setup a 2D thread grid, with 16x16 blocks * /
	/// * each block is will use nearby memory* /
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/// * call the functions * /
	for (j = 0; j<it; j++) {
		/// * f = div z - g*(1/lambda)* /
		fupdate_inter << < n_blocks, block_size >> > (cuda_z, cuda_g, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		/// *z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)* /
		zupdate_inter << < n_blocks, block_size >> > (cuda_z, 0.125, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	checkCUDAError("memcpy");

	/// * u = -f * lambda* /
	solution_inter << < n_blocks, block_size >> > (cuda_z, cuda_g, lambda, nx, ny);

	/// * recover the output from device memory* /
	cudaMemcpy(input, cuda_g, size, cudaMemcpyDeviceToHost);

	/// * free device memory * /
	cudaFree(cuda_z);
	cudaFree(cuda_g);
}



/// * cuda DUMMY * /
void cudaTVcha2_dummy(float *input, int it, int nx, int ny, float lambda)
{
	float *cuda_g;
	float *cuda_z1;
	float *cuda_z2;
	float *cuda_f;
	size_t size;
	int j;

	size = nx*ny * sizeof(float);

	printf("lambda=%g, it=%d, nx=%d, ny=%d\n", lambda, it, nx, ny);

	/// * allocate device memory * /
	cudaMalloc((void **)&cuda_g, size);
	cudaMalloc((void **)&cuda_f, size);
	cudaMalloc((void **)&cuda_z1, size);
	cudaMalloc((void **)&cuda_z2, size);

	/// * Copy input to device* /
	cudaMemcpy(cuda_g, input, size, cudaMemcpyHostToDevice);
	cudaMemset(cuda_f, 0, size);
	cudaMemset(cuda_z1, 0, size);
	cudaMemset(cuda_z2, 0, size);

	checkCUDAError("memcpy");

	//int block_size = 16;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	/// * setup a 2D thread grid, with 16x16 blocks * /
	/// * each block is will use nearby memory* /
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	/// * call the functions * /
	for (j = 0; j<it; j++) {
		/// * f = div z - g*(1/lambda)* /
		fupdate_dummy << < n_blocks, block_size >> > (cuda_f, cuda_z1, cuda_z2, cuda_g, 1 / lambda, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();

		/// *z^t+1 = (z + tau grad(f) )/(1+tau|grad(f)|)* /
		zupdate2_dummy << < n_blocks, block_size >> > (cuda_z1, cuda_z2, cuda_f, 0.125, nx, ny);

		// block until the device has completed
		cudaThreadSynchronize();
	}

	checkCUDAError("memcpy");

	/// * u = -f * lambda* /
	solution_dummy << < n_blocks, block_size >> > (cuda_f, lambda, nx, ny);

	/// * recover the output from device memory* /
	cudaMemcpy(input, cuda_f, size, cudaMemcpyDeviceToHost);

	/// * free device memory * /
	cudaFree(cuda_f);
	cudaFree(cuda_z1);
	cudaFree(cuda_z2);
	cudaFree(cuda_g);
}


