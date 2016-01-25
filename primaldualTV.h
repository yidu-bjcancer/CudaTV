/* This file is part of cudaTV.  See the file COPYING for further details. */
void cudaTVpd(float *input, int it, int nx, int ny, float lam);
void cudaTVpd_SoA(float *input, int it, int nx, int ny, float lam);
void cudaTVpdhg_SoA(float *input, int it, int nx, int ny, float lam);
