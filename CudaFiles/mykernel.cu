

__global__ void mykernel( double *aplusb, 
						  const double *a, 
						  const double *b, 
						  const int N  
                ) {
	/* "const" because the value is not changing in kernel, N = size of vectors */

	// compute index of this kernel
	int n = blockIdx.x*blockDim.x + threadIdx.x;

	// if index is smaller than size, then compute something
	if(n<N){
		aplusb[n] = a[n] + b[n];
	} else {
		/* put your feet on the table */
	}

}
