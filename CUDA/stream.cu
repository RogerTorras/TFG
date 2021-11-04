# define _XOPEN_SOURCE 600

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <string.h>
# include <limits.h>
# include <sys/time.h>
# include <time.h>

//# include "mpi.h"

//# include <omp.h>




//N is Total Mem to be used and the size when only one vector
//is used in a kernel
#ifndef N
#define N	2048
#endif

#define DATA_TYPE float
//N2 for kernels with two vectors
#define N2 N/2
//N3 for kernels with three vectors
#define N3 N/3

#ifndef NTIMES
#define NTIMES	10
#endif 
#define NBENCH	16

#define SCALAR 0.42

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif



// Some compilers require an extra keyword to recognize the "restrict" qualifier.
DATA_TYPE * __restrict__ a1, * __restrict__ b2, * __restrict__ c2, * __restrict__ d3, * __restrict__ e3, * __restrict__ f3, * __restrict__ vxmo, * __restrict__ vxmi, * __restrict__ mat_atax;
DATA_TYPE * d_a1, *  d_b2, *  d_c2, * __restrict__ d_d3, * __restrict__ d_e3, * __restrict__ d_f3;


DATA_TYPE * __restrict__ rand_list;

DATA_TYPE red_var = 0.0f;

size_t		array_elements, array_elements2, array_elements3, array_bytes, array_bytes2, array_bytes3, array_bytes_vxm, array_bytes_mat_atax, array_alignment, sq_array_elements, cb_array_elements, sq_array_elements3, n_vxm;
static double	avgtime[NBENCH], maxtime[NBENCH],
		mintime[NBENCH];

static char	*label[NBENCH] = {"Copy", "Scale",
    "Add", "Triad", "Reduction", "2PStencil", "2D4PStencil", "MatxVec",
    "MatMult", "Stride2", "Stride4", "Stride16", "Stride64", "Rows", "MatMultNoOpt", "Stencil"};


//Se puede eliminar o corregir
static long int	bytes[NBENCH] = {
    2 * sizeof(DATA_TYPE) * N,
    2 * sizeof(DATA_TYPE) * N,
    3 * sizeof(DATA_TYPE) * N,
    3 * sizeof(DATA_TYPE) * N,
	1 * sizeof(DATA_TYPE) * N,
	3 * sizeof(DATA_TYPE) * N,
	5 * sizeof(DATA_TYPE) * N,
  2 * sizeof(DATA_TYPE) * N,
  3 * sizeof(DATA_TYPE) * N,
  3 * sizeof(DATA_TYPE) * N,
  3 * sizeof(DATA_TYPE) * N,
  3 * sizeof(DATA_TYPE) * N,
  3 * sizeof(DATA_TYPE) * N,
  2 * sizeof(DATA_TYPE) * N,
  8 * sizeof(DATA_TYPE) * N,
  8 * sizeof(DATA_TYPE) * N,
    };


/*#ifdef _OPENMP
extern int omp_get_num_threads();
#endif*/

double		times[NBENCH][NTIMES];



int rank = -1;




__global__ void  Kernel_Copy_CUDA ( float *b2, float *c2 )
{
	int i = threadIdx.x;

	c2[i] = b2[i];
 }

void __attribute__((noinline)) Kernel_Copy( int k )
{
	
	clock_t start_t, end_t;

	int j;
	start_t = clock();
	// kernel 1: Copy
	Kernel_Copy_CUDA<<<N2/1024, 1024>>>(d_b2, d_c2);
	/*
	for (j=0; j<N2; j++)
		c2[j] = b2[j];
	*/
	end_t = clock();
	times[0][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	
	//Out
	/*FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[0]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);*/
}

void __attribute__((noinline)) Kernel_Scale( int k, DATA_TYPE scalar )
{


	
	clock_t start_t, end_t;
	int j;
	start_t = clock();
	// kernel 2: Scale
	for (j=0; j<N2; j++)
		c2[j] = scalar*b2[j];
	end_t = clock();
	times[1][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	/*FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[1]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);*/

}

void __attribute__((noinline)) Kernel_Add( int k )
{

	
	clock_t start_t, end_t;
	int j;
	start_t = clock();
	// kernel 3: Add
	for (j=0; j<N3; j++)
		f3[j] = d3[j]+e3[j];
	end_t = clock();
	times[2][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	/*FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<N3)--F\n[",label[2]);
	for(j=0; j<N3; j++)
		fprintf(logFile,",%f",f3[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);*/

}

void __attribute__((noinline)) Kernel_Triad( int k, double scalar )
{

	
	clock_t start_t, end_t;
	int j;
	// kernel 4: Triad
	start_t = clock();
	for (j=0; j<N3; j++)
		f3[j] = d3[j]+scalar*e3[j];
	end_t = clock();
	times[3][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<N3)--F\n[",label[3]);
	for(j=0; j<N3; j++)
		fprintf(logFile,",%f",f3[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}
//TODO hauria de funcionar

void __attribute__((noinline)) Kernel_Reduction( int k )
{

	
	clock_t start_t, end_t;
	int j;
	double reduc = 0.0f;
	// kernel 5: Reduction
	start_t = clock();
	for (j=0; j<N; j++)
		reduc +=a1[j];
	red_var = fmod(reduc+red_var, FLT_MAX );
	end_t = clock();
	times[4][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<N)--reduc\n[",label[4]);
	for(j=0; j<N; j++)
		fprintf(logFile,",%f",reduc);
	fprintf(logFile,"]\n");
	fclose(logFile);

}


void __attribute__((noinline)) Kernel_2PStencil( int k )
{

	
	
	clock_t start_t, end_t;
	int j;
	start_t = clock();
	// kernel 6: 2PStencil
	for (j=1; j<N2-1; j++)
		c2[j] = (b2[j-1]+b2[j+1])*0.5;	
	end_t = clock();
	times[5][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<N2-1)--C\n[",label[5]);
	for(j=1; j<N2-1; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}


void __attribute__((noinline)) Kernel_2D4PStencil( int k )
{


	
	clock_t start_t, end_t;
	int n = sq_array_elements;
	int j, i;
	start_t = clock();
	// kernel 7: 2D4PStencil
	for ( j=1; j < n-1; j++ )
		for ( i=1; i < n-1; i++ )
			c2[j*n+i] = (b2[j*n+i-1]+b2[j*n+i+1]+b2[(j-1)*n+i]+b2[(j+1)*n+i])*0.25f;

	end_t = clock();
	times[6][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j1<n-1)(i1<n-1)--C\n[",label[6]);
	for(j=1; j<n-1; j++)
		for(i=1; i<n-1; i++)
			fprintf(logFile,",%f",c2[j*n+i]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}


//atax_1 -- from polybench
void __attribute__((noinline)) Kernel_MatxVec( int k )
{

	clock_t start_t, end_t;
	int i, j;
	int n = n_vxm;
	// kernel 9: Scatter
	start_t = clock();
	for (i = 0; i < n; i++)
	{
		vxmo[i] = 0.0;
		for (j = 0; j < n; j++)
			vxmo[i] = vxmo[i] + mat_atax[i*n+j] * vxmi[j];
	}
	end_t = clock();
	times[7][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;

	//Out
	FILE *logFile = fopen("results.txt","a");
	//fprintf(logFile,"\n%d\n",n);
	fprintf(logFile,"-----%s---(i<n)--vxmo\n[",label[7]);
	for(i=0; i<n; i++)
		fprintf(logFile,",%f",vxmo[i]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}


//matrixmult -- read correctly
void __attribute__((noinline)) Kernel_MatMult( int k )
{

	

	clock_t start_t, end_t;
	int i, j, z;
	int n = sq_array_elements3;
	// kernel 9: Scatter
	start_t = clock();
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
		{
			d3[i*n+j] = 0.0;
			for (z = 0; z < n; ++z)
			  d3[i*n+j] += e3[i*n+z] * f3[j*n+z];
		}
	end_t = clock();
	times[8][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(i<n)(j<n)--D\n[",label[8]);
	for(i=0; i<n; i++)
		for(j=0; j<n; j++)
			fprintf(logFile,",%f",d3[i*n+j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}


//Stride 2
void __attribute__((noinline)) Kernel_Stride2( int k )
{
	
	
	
	clock_t start_t, end_t;
	int j, i;
	// kernel 10: Stride2
	start_t = clock();
	for ( j=0; j < N2; j++ )
	{
	  unsigned long int index = j*2;
	  index = (index+(unsigned long int)(index/N2))%N2;
	  c2[index] = b2[index];
	}
	end_t = clock();
	times[9][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[9]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");
	//fprintf(logFile,"Index[%d]\n",index);
	fclose(logFile);

}


void __attribute__((noinline)) Kernel_Stride4( int k )
{


	
	clock_t start_t, end_t;
	int j, i;
	// kernel 11: Stride
	start_t = clock();
	for ( j=0; j < N2; j++ )
	{
	  unsigned long int index = j*4;
	  index = (index+(unsigned long int)(index/N2))%N2;
	  c2[index] = b2[index];
	}	
	end_t = clock();
	times[10][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[10]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");//Index[%d]\n",index);
	fclose(logFile);

}



void __attribute__((noinline)) Kernel_Stride16( int k )
{
	
	
	
	clock_t start_t, end_t;
	int j, i;
	// kernel 12: Stride16
	start_t = clock();
	for ( j=0; j < N2; j++ )
	{
	  unsigned long int index = j*16;
	  index = (index+(unsigned long int)(index/N2))%N2;
	  c2[index] = b2[index];
	}
	end_t = clock();
	times[11][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[11]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");//Index[%d]\n",index);
	fclose(logFile);

}


void __attribute__((noinline)) Kernel_Stride64( int k )
{


	
	clock_t start_t, end_t;
	int n = sq_array_elements;
	int j, i;
	// kernel 13: Stride64
	start_t = clock();
	for ( j=0; j < N2; j++ )
  	{
		unsigned long int index = j*64;
		index = (index+(unsigned long int)(index/N2))%N2;
		c2[index] = b2[index];
 	}
	end_t = clock();
	times[12][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[12]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");//Index[%d]\n",index);
	fclose(logFile);

}


void __attribute__((noinline)) Kernel_Rows( int k )
{
	
	
	
	clock_t start_t, end_t;
	int n = sq_array_elements;
	int j, i;
	// kernel 14: Rows
	start_t = clock();
	for ( j=0; j < n; j++ )
		for ( i=0; i < n; i++ )
			c2[i*n+j] = b2[i*n+j];

	end_t = clock();
	times[13][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<n)(i<n)--C\n[",label[13]);
	for(j=0; j<n; j++)
		for(i=0;i<n;i++)
			fprintf(logFile,",%f",c2[i*n+j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}



//mm_fc -- polybench
void __attribute__((noinline)) Kernel_MatMultNoOpt( int k )
{
  	

	

	clock_t start_t, end_t;
	int i, j, z;
	int n = sq_array_elements3;
	// kernel 16: mm_fc
	start_t = clock();
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
		{
			d3[i*n+j] = 0.0;
			for (z = 0; z < n; ++z)
			  d3[i*n+j] += e3[i*n+z] * f3[z*n+j];
		}

	end_t = clock();
	times[14][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(i<n)(j<n)--D\n[",label[14]);
	for(i=0;i<n;i++)
		for(j=0; j<n; j++)
			fprintf(logFile,",%f",d3[i*n+j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

	
}


//stencil
void __attribute__((noinline)) Kernel_Stencil( int k )
{
  
  	clock_t start_t, end_t;
  	int i, j, z;
	int n = cb_array_elements;
	int n2 = n*n;
	// kernel 15: Test
	start_t = clock();
	for (i = 1; i < n-1; i++) {
		for (j = 1; j < n-1; j++) {
			for (z = 1; z < n-1; z++) {
				c2[i*n2+j*n+z] =  0.125 * (b2[(i+1)*n2+j*n+z] - 2.0 * b2[i*n2+j*n+z] + b2[(i-1)*n2+j*n+z])
								+ 0.125 * (b2[i*n2+(j+1)*n+z] - 2.0 * b2[i*n2+j*n+z] + b2[i*n2+(j-1)*n+z])
								+ 0.125 * (b2[i*n2+j*n+(z+1)] - 2.0 * b2[i*n2+j*n+z] + b2[i*n2+j*n+(z-1)])
								+ b2[i*n2+j*n+z];
			  
		   }
	   }
   }
	end_t = clock();
	times[15][k] = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(i1<n-1)(j1<n-1)(z1<n-1)--C\n[",label[15]);
	for(i=1; i<n-1; i++)
		for(j=1; j<n-1; j++)
			for(z=1; z<n-1; z++)
				fprintf(logFile,",%f",c2[i*n2+j*n+z]);
	fprintf(logFile,"]\n");

}


int
main(int argc, char *argv[])
    {
    int			BytesPerWord;
    int			i,k;
    ssize_t		j;
    DATA_TYPE		scalar;
    double		t;
	double		*TimesByRank;
	int         rc, numranks = 1, myrank;
	
	char* affinity=NULL;

	affinity = argv[1];
	


    /* --- SETUP --- call MPI_Init() before anything else! --- */

    
    
    
   
    //size_matmul = sqrt(N);
	// if either of these fail there is something really screwed up!
    /* --- NEW FEATURE --- distribute requested storage across MPI ranks --- */
	array_elements = N / numranks;		// don't worry about rounding vs truncation
	array_elements2 = N2 / numranks;		// don't worry about rounding vs truncation
	array_elements3 = N3 / numranks;		// don't worry about rounding vs truncation
	sq_array_elements = sqrt(N2);
	sq_array_elements3 = sqrt(N3);
	cb_array_elements = cbrt(N2);
	n_vxm = sqrt(N+1)-1; 
	printf("n_vxm: %ld\n", n_vxm);
	
    array_alignment = 64;						// Can be modified -- provides partial support for adjusting relative alignment

	// Dynamically allocate the three arrays using "posix_memalign()"
	// NOTE that the OFFSET parameter is not used in this version of the code!
    array_bytes = array_elements * sizeof(DATA_TYPE);
	array_bytes2 = array_elements2 * sizeof(DATA_TYPE);
	array_bytes3 = array_elements3 * sizeof(DATA_TYPE);
	array_bytes_vxm = n_vxm * sizeof(DATA_TYPE);
	array_bytes_mat_atax = n_vxm * n_vxm * sizeof(DATA_TYPE);
    k = posix_memalign((void **)&a1, array_alignment, array_bytes);
    if (k != 0) {
        printf("Rank %d: Allocation of array a failed, return code is %d\n",myrank,k);
        exit(1);
    }
    k = posix_memalign((void **)&b2, array_alignment, array_bytes2);
    if (k != 0) {
        printf("Rank %d: Allocation of array b2 failed, return code is %d\n",myrank,k);
        exit(1);
    }
    k = posix_memalign((void **)&c2, array_alignment, array_bytes2);
    if (k != 0) {
        printf("Rank %d: Allocation of array c2 failed, return code is %d\n",myrank,k);
        exit(1);
    }
    k = posix_memalign((void **)&d3, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array d3 failed, return code is %d\n",myrank,k);
        exit(1);
    }
    k = posix_memalign((void **)&e3, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array e3 failed, return code is %d\n",myrank,k);
        exit(1);
    }
    k = posix_memalign((void **)&f3, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array f3 failed, return code is %d\n",myrank,k);
        exit(1);
    }    
    k = posix_memalign((void **)&rand_list, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
        exit(1);
    }
	
	k =  posix_memalign((void **)&vxmo, array_alignment, array_bytes_vxm);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
        exit(1);
    }
	
	k =  posix_memalign((void **)&vxmi, array_alignment, array_bytes_vxm);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
        exit(1);
    }
	
	k =  posix_memalign((void **)&mat_atax, array_alignment, array_bytes_mat_atax);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
        exit(1);
    }

	cudaMalloc(&d_b2, array_elements2 * sizeof(DATA_TYPE)); 
	cudaMalloc(&d_c2, array_elements2 * sizeof(DATA_TYPE)); 
    /* --- SETUP --- initialize arrays and estimate precision of timer --- */

    for (j=0; j<N; j++) {
	    a1[j] = 1.0;
	}
	cudaMemcpy(d_a1, a1, array_bytes, cudaMemcpyHostToDevice);
    for (j=0; j<N2; j++) {
	    b2[j] = 1.0;
		c2[j] = 0.0;
	}
    for (j=0; j<N3; j++) {
	    d3[j] = 1.0;
		e3[j] = 1.0;
		f3[j] = 1.0;
	}
   for (j=0; j<n_vxm; j++){
	vxmi[j] = 1.0;
	vxmo[j] = 0.0;
	}
   for(j=0; j<n_vxm*n_vxm; j++){
        mat_atax[j] = 1.0;
        }

    srand(0);
    for (j = 0; j < N3; j++)
	    rand_list[j] = ((float)rand()/RAND_MAX)*N3;
	
	
    //printf(" \n");
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    // This code has more barriers and timing calls than are actually needed, but
    // this should not cause a problem for arrays that are large enough to satisfy
    // the STREAM run rules.
	// MAJOR FIX!!!  Version 1.7 had the start timer for each loop *after* the
	// MPI_Barrier(), when it should have been *before* the MPI_Barrier().
    // 
	

	cudaMemcpy(d_b2, b2, array_bytes2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c2, c2, array_bytes2, cudaMemcpyHostToDevice);

	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"--------------------------------------------\n\n\n");
	fclose(logFile);
	scalar = SCALAR;
	sleep(1);

	for (k=0; k<NTIMES; k++)
		Kernel_Copy( k );
	cudaMemcpy(c2, d_c2, array_bytes2, cudaMemcpyDeviceToHost);
	float sum =0.0;
	for(k=0;k<array_elements2; k++)
	{
		sum+=c2[k];
	}
	printf("DEBUG: Final result %f \n",sum);
	for (k=0; k<NTIMES; k++)
		Kernel_Scale( k, scalar );
	for (k=0; k<NTIMES; k++)
		Kernel_Add( k );
		
	for (k=0; k<NTIMES; k++)	
		Kernel_Triad( k, scalar );
	for (k=0; k<NTIMES; k++)
		Kernel_Reduction( k );
	for (k=0; k<NTIMES; k++)
		Kernel_2PStencil( k );
	for (k=0; k<NTIMES; k++)
		Kernel_2D4PStencil( k );
 	for (k=0; k<NTIMES; k++)
		Kernel_MatxVec( k );
	for (k=0; k<NTIMES; k++)
		Kernel_MatMult( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Stride2( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Stride4( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Stride16( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Stride64( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Rows( k );
	for (k=0; k<NTIMES; k++)
		Kernel_MatMultNoOpt( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Stencil( k );
	
	for(int y = 0; y < NBENCH ; y++)
	{
		float m = 0.0;
		for(int z = 0; z <NTIMES ; z++)
		{
			m += times[y][z];
		}
		printf("DEBUG: Final Timing %s: %f seconds\n",label[y],m);
	}


    //	--- SUMMARY --- 


	printf("red_var %f\n", red_var);
	// Rank 0 processes all timing data
	
	
	free(a1);
	cudaFree(d_a1);
	free(b2);
	free(c2);
	free(d3);
	free(e3);
	free(f3);
	free(rand_list);
	/*if (myrank == 0) {
		free(TimesByRank);
	}*/

    //MPI_Finalize();
    
	return(0);
}

