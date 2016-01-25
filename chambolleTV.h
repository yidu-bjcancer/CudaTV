/* This file is part of cudaTV.  See the file COPYING for further details. */
void cudaTVcha(float *input, int it, int nx, int ny, float lam);
void cudaTVcha2(float *input, int it, int nx, int ny, float lam);
void cudaTVcha2_structure_of_arrays(float *input, int it, int nx, int ny, float lam);
void cudaTVcha2_inter(float *input, int it, int nx, int ny, float lam);
void cudaTVcha2_shared(float *input, int it, int nx, int ny, float lam);
void cudaTVcha2_stencil(float *input, int it, int nx, int ny, float lam);

void cudaTVcha2_dummy(float *input, int it, int nx, int ny, float lam);
