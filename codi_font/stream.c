# define _XOPEN_SOURCE 600

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <string.h>
# include <limits.h>
# include <sys/time.h>
# include "mpi.h"

# include <omp.h>




//N is Total Mem to be used and the size when only one vector
//is used in a kernel
#ifndef N
#define N	1020
#endif

#define DATA_TYPE float
//N2 for kernels with two vectors
#define N2 N/2
//N3 for kernels with three vectors
#define N3 N/3

#ifndef NTIMES
#define NTIMES	1
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


DATA_TYPE * __restrict__ rand_list;

DATA_TYPE red_var = 0.0f;

size_t		array_elements, array_elements2, array_elements3, array_bytes, array_bytes2, array_bytes3, array_bytes_vxm, array_bytes_mat_atax, array_alignment, sq_array_elements, cb_array_elements, sq_array_elements3, n_vxm;
static double	avgtime[NBENCH], maxtime[NBENCH],
		mintime[NBENCH];

static char	*label[NBENCH] = {"Copy", "Scale",
    "Add", "Triad", "Reduction", "2PStencil", "2D4PStencil", "Gather",
    "Scatter", "Stride2", "Stride4", "Stride16", "Stride64", "Rows", "Test", "Stencil"};


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


#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

double		times[NBENCH][NTIMES];



int rank = -1;


void __attribute__((noinline)) Kernel_Copy( int k )
{
	
	
	double t0, t1;
	int j;
	// kernel 1: Copy
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for
	for (j=0; j<N2; j++)
		c2[j] = b2[j];
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[0][k] = t1 - t0;
	
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[0]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);
}

void __attribute__((noinline)) Kernel_Scale( int k, DATA_TYPE scalar )
{

	
	
	
	int j;
	double t0, t1;
	// kernel 2: Scale
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for
	for (j=0; j<N2; j++)
		c2[j] = scalar*b2[j];
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[1][k] = t1-t0;
	
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s-----C\n[",label[1]);
	for(j=0; j<N2; j++)
		fprintf(logFile,",%f",c2[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}

void __attribute__((noinline)) Kernel_Add( int k )
{

	

	int j;
	double t0, t1;
	// kernel 3: Add
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for
	for (j=0; j<N3; j++)
		f3[j] = d3[j]+e3[j];
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[2][k] = t1-t0;
	
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<N3)--F\n[",label[2]);
	for(j=0; j<N3; j++)
		fprintf(logFile,",%f",f3[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}

void __attribute__((noinline)) Kernel_Triad( int k, double scalar )
{

	

	int j;
	double t0, t1;
	// kernel 4: Triad
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for
	for (j=0; j<N3; j++)
		f3[j] = d3[j]+scalar*e3[j];
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[3][k] = t1-t0;	
	
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<N3)--F\n[",label[3]);
	for(j=0; j<N3; j++)
		fprintf(logFile,",%f",f3[j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}

void __attribute__((noinline)) Kernel_Reduction( int k )
{

	

	int j;
	double t0, t1;
	double reduc = 0.0f;
	// kernel 5: Reduction
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for reduction(+:reduc)
	for (j=0; j<N; j++)
		reduc +=a1[j];
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	red_var = fmod(reduc+red_var, FLT_MAX );
	times[4][k] = t1-t0;	
	
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

	
	

	int j;
	double t0, t1;
	// kernel 6: 2PStencil
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for
	for (j=1; j<N2-1; j++)
		c2[j] = (b2[j-1]+b2[j+1])*0.5;
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[5][k] = t1-t0;	
	
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


	
	
	int n = sq_array_elements;
	int j, i;
	double t0, t1;
	// kernel 7: 2D4PStencil
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=1; j < n-1; j++ )
		for ( i=1; i < n-1; i++ )
			c2[j*n+i] = (b2[j*n+i-1]+b2[j*n+i+1]+b2[(j-1)*n+i]+b2[(j+1)*n+i])*0.25f;
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[6][k] = t1-t0;	
	
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j1<n-1)(i1<n-1)--C\n[",label[6]);
	for(j=1; j<n-1; j++)
		for(i=1; i<n-1; i++)
			fprintf(logFile,",%f",c2[j*n+i]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}

/*
//Store seq, read random
void __attribute__((noinline)) Kernel_Gather( int k )
{
	int j, i;
	double t0, t1;
	// kernel 8: Gather
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=0; j < N3; j++ )
  {
	  f3[j] = d3[(unsigned int)rand_list[j]];
  }
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[7][k] = t1-t0;	
}

//Store random, read seq
void __attribute__((noinline)) Kernel_Scatter( int k )
{
	int j, i;
	double t0, t1;
	// kernel 9: Scatter
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=0; j < N3; j++ )
  {
	  f3[(unsigned int)rand_list[j]] = d3[j];
  }
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[8][k] = t1-t0;	
}*/


//atax_1 -- from polybench
void __attribute__((noinline)) Kernel_Gather( int k )
{


	

	
	int i, j;
	int n = n_vxm;
	double t0, t1;
	// kernel 9: Scatter
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(j)
	for (i = 0; i < n; i++)
	{
		vxmo[i] = 0.0;
		for (j = 0; j < n; j++)
			vxmo[i] = vxmo[i] + mat_atax[i*n+j] * vxmi[j];
	}
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[7][k] = t1-t0;	

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
void __attribute__((noinline)) Kernel_Scatter( int k )
{

	


	int i, j, z;
	int n = sq_array_elements3;
	double t0, t1;
	// kernel 9: Scatter
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i, j, z)
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
		{
			d3[i*n+j] = 0.0;
			for (z = 0; z < n; ++z)
			  d3[i*n+j] += e3[i*n+z] * f3[j*n+z];
		}
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[8][k] = t1-t0;	
	
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
	
	
	
	
	int j, i;
	double t0, t1;
	// kernel 10: Stride2
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=0; j < N2; j++ )
	{
	  unsigned long int index = j*2;
	  index = (index+(unsigned long int)(index/N2))%N2;
	  c2[index] = b2[index];
	}
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[9][k] = t1-t0;	
	
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


	
	
	int j, i;
	double t0, t1;
	// kernel 11: Stride4
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=0; j < N2; j++ )
	{
	  unsigned long int index = j*4;
	  index = (index+(unsigned long int)(index/N2))%N2;
	  c2[index] = b2[index];
	}
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[10][k] = t1-t0;	
	
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
	
	
	
	
	int j, i;
	double t0, t1;
	// kernel 12: Stride16
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=0; j < N2; j++ )
	{
	  unsigned long int index = j*16;
	  index = (index+(unsigned long int)(index/N2))%N2;
	  c2[index] = b2[index];
	}
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[11][k] = t1-t0;
	
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


	
	
	int n = sq_array_elements;
	int j, i;
	double t0, t1;
	// kernel 13: Stride64
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=0; j < N2; j++ )
  	{
		unsigned long int index = j*64;
		index = (index+(unsigned long int)(index/N2))%N2;
		c2[index] = b2[index];
 	}
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[12][k] = t1-t0;	
	
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
	
	
	
	
	int n = sq_array_elements;
	int j, i;
	double t0, t1;
	// kernel 14: Rows
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i,j)
	for ( j=0; j < n; j++ )
		for ( i=0; i < n; i++ )
			c2[i*n+j] = b2[i*n+j];
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[13][k] = t1-t0;	
	
	//Out
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"-----%s---(j<n)(i<n)--C\n[",label[13]);
	for(j=0; j<n; j++)
		for(i=0;i<n;i++)
			fprintf(logFile,",%f",c2[i*n+j]);
	fprintf(logFile,"]\n");
	fclose(logFile);

}
/*
//Random kernel
void __attribute__((noinline)) Kernel_Test( int k )
{
	double t0, t1;
	int j;
	// kernel 15: Test
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for
	for (j=0; j<N3; j++)
		f3[j] = (d3[j]*d3[j]+(e3[j]+e3[j])*d3[j])*d3[j]+e3[j];
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[14][k] = t1 - t0;
}*/



//mm_fc -- polybench
void __attribute__((noinline)) Kernel_Test( int k )
{
  	

	


	int i, j, z;
	int n = sq_array_elements3;
	double t0, t1;
	// kernel 16: mm_fc
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	#pragma omp parallel for private(i, j, z)
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
		{
			d3[i*n+j] = 0.0;
			for (z = 0; z < n; ++z)
			  d3[i*n+j] += e3[i*n+z] * f3[z*n+j];
		}
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[14][k] = t1 - t0;

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
  
  
  int i, j, z;
	int n = cb_array_elements;
	int n2 = n*n;
	double t0, t1;
	// kernel 15: Test
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	#pragma omp parallel for private(i, j, z)
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
   	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[15][k] = t1 - t0;

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
	double		t0,t1,tmin;
	int         rc, numranks, myrank;
	
	char* affinity=NULL;
	if(argc!=2)
	{
		printf("use of file: ./stream [close|spread|auto] \n");
		exit(EXIT_FAILURE);
		
	}

	affinity = argv[1];
	


    /* --- SETUP --- call MPI_Init() before anything else! --- */

    rc = MPI_Init(NULL, NULL);
    if (rc != MPI_SUCCESS) {
       printf("ERROR: MPI Initialization failed with return code %d\n",rc);
       exit(1);
    }
    
    
    
   
    //size_matmul = sqrt(N);
	// if either of these fail there is something really screwed up!
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	rank = myrank;
    /* --- NEW FEATURE --- distribute requested storage across MPI ranks --- */
	array_elements = N / numranks;		// don't worry about rounding vs truncation
	array_elements2 = N2 / numranks;		// don't worry about rounding vs truncation
	array_elements3 = N3 / numranks;		// don't worry about rounding vs truncation
	sq_array_elements = sqrt(N2);
	sq_array_elements3 = sqrt(N3);
	cb_array_elements = cbrt(N2);
	n_vxm = sqrt(N+1)-1; 
	printf("n_vxm: %d\n", n_vxm);
	
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
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
    k = posix_memalign((void **)&b2, array_alignment, array_bytes2);
    if (k != 0) {
        printf("Rank %d: Allocation of array b2 failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
    k = posix_memalign((void **)&c2, array_alignment, array_bytes2);
    if (k != 0) {
        printf("Rank %d: Allocation of array c2 failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
    k = posix_memalign((void **)&d3, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array d3 failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
    k = posix_memalign((void **)&e3, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array e3 failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
    k = posix_memalign((void **)&f3, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array f3 failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }    
    k = posix_memalign((void **)&rand_list, array_alignment, array_bytes3);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
	
	k =  posix_memalign((void **)&vxmo, array_alignment, array_bytes_vxm);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
	
	k =  posix_memalign((void **)&vxmi, array_alignment, array_bytes_vxm);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
	
	k =  posix_memalign((void **)&mat_atax, array_alignment, array_bytes_mat_atax);
    if (k != 0) {
        printf("Rank %d: Allocation of array rand_list failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }


	// Initial informational printouts -- rank 0 handles all the output
	if (myrank == 0) {
		BytesPerWord = sizeof(DATA_TYPE);
		printf("This system uses %d bytes per array element.\n",
		BytesPerWord);


		printf("Total Aggregate Array size N = %llu (elements)\n" , (unsigned long long) N);
		printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n", 
			BytesPerWord * ( (double) N / 1024.0/1024.0),
			BytesPerWord * ( (double) N / 1024.0/1024.0/1024.0));
		printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
			(1.0 * BytesPerWord) * ( (double) N / 1024.0/1024.),
			(1.0 * BytesPerWord) * ( (double) N / 1024.0/1024./1024.));
		printf("Data is distributed across %d MPI ranks\n",numranks);
		printf("Each fucntion will be executed %d times.\n", NTIMES);

		
		printf("Total Aggregate Array size N2 = %llu (elements)\n" , (unsigned long long) N2);
		printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n", 
			BytesPerWord * ( (double) N2 / 1024.0/1024.0),
			BytesPerWord * ( (double) N2 / 1024.0/1024.0/1024.0));
		printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
			(2.0 * BytesPerWord) * ( (double) N2 / 1024.0/1024.),
			(2.0 * BytesPerWord) * ( (double) N2 / 1024.0/1024./1024.));
		printf("Data is distributed across %d MPI ranks\n",numranks);
		printf("Each fucntion will be executed %d times.\n", NTIMES);

		
		printf("Total Aggregate Array size N3 = %llu (elements)\n" , (unsigned long long) N3);
		printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n", 
			BytesPerWord * ( (double) N3 / 1024.0/1024.0),
			BytesPerWord * ( (double) N3 / 1024.0/1024.0/1024.0));
		printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
			(3.0 * BytesPerWord) * ( (double) N3 / 1024.0/1024.),
			(3.0 * BytesPerWord) * ( (double) N3 / 1024.0/1024./1024.));
		printf("Data is distributed across %d MPI ranks\n",numranks);
		printf("Each function will be executed %d times.\n", NTIMES);


#ifdef _OPENMP
#pragma omp parallel 
		{
#pragma omp master
			{
				k = omp_get_num_threads();
				printf ("Number of Threads requested for each MPI rank = %i\n",k);
			}
		}
#endif


	}

    /* --- SETUP --- initialize arrays and estimate precision of timer --- */

	#pragma omp parallel for
    for (j=0; j<N; j++) {
	    a1[j] = 1.0;
	}
	#pragma omp parallel for
    for (j=0; j<N2; j++) {
	    b2[j] = 1.0;
		c2[j] = 0.0;
	}
	#pragma omp parallel for
    for (j=0; j<N3; j++) {
	    d3[j] = 1.0;
		e3[j] = 1.0;
		f3[j] = 1.0;
	}
   	#pragma omp parallel for
   for (j=0; j<n_vxm; j++){
	vxmi[j] = 1.0;
	vxmo[j] = 0.0;
	}
	#pragma omp parallel for
   for(j=0; j<n_vxm*n_vxm; j++){
        mat_atax[j] = 1.0;
        }
 
	if (myrank == 0) {
		// There are NBENCH*NTIMES timing values for each rank (always doubles)
		TimesByRank = (double *) malloc(NBENCH * NTIMES * sizeof(double) * numranks);
		if (TimesByRank == NULL) {
			printf("Ooops -- allocation of arrays to collect timing data on MPI rank 0 failed\n");
			MPI_Abort(MPI_COMM_WORLD, 3);
		}
		memset(TimesByRank,0,NBENCH*NTIMES*sizeof(double)*numranks);
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
	
	FILE *logFile = fopen("results.txt","a");
	fprintf(logFile,"--------------------------------------------\n\n\n");
	fclose(logFile);
	scalar = SCALAR;
	sleep(1);
	for (k=0; k<NTIMES; k++)
		Kernel_Copy( k );
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
		Kernel_Gather( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Scatter( k );
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
		Kernel_Test( k );
	for (k=0; k<NTIMES; k++)
		Kernel_Stencil( k );



    //	--- SUMMARY --- 

	// Because of the MPI_Barrier() calls, the timings from any thread are equally valid. 
    // The best estimate of the maximum performance is the minimum of the "outside the barrier"
    // timings across all the MPI ranks.

	// Gather all timing data to MPI rank 0
	MPI_Gather(times, NBENCH*NTIMES, MPI_DOUBLE, TimesByRank, NBENCH*NTIMES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	printf("red_var %f\n", red_var);
	// Rank 0 processes all timing data
	
	if (myrank == 0) {
		// for each iteration and each kernel, collect the minimum time across all MPI ranks
		// and overwrite the rank 0 "times" variable with the minimum so the original post-
		// processing code can still be used.
		for (k=0; k<NTIMES; k++) {
			for (j=0; j<NBENCH; j++) {
				tmin = 1.0e36;
				for (i=0; i<numranks; i++) {
					// printf("DEBUG: Timing: iter %d, kernel %lu, rank %d, tmin %f, TbyRank %f\n",k,j,i,tmin,TimesByRank[4*NTIMES*i+j*NTIMES+k]);
					tmin = MIN(tmin, TimesByRank[NBENCH*NTIMES*i+j*NTIMES+k]);
				}
				// printf("DEBUG: Final Timing: iter %d, kernel %lu, final tmin %f\n",k,j,tmin);
				times[j][k] = tmin;
			}
		}
		for (j=0; j< NBENCH; j++)
		{
			avgtime[j] = 0;
			maxtime[j] = 0;
			mintime[j] = FLT_MAX;
		}
		
  
	// Back to the original code, but now using the minimum global timing across all ranks
		for (k=2; k<NTIMES; k++) // note -- skip first iteration 
		{
			for (j=0; j<NBENCH; j++)
			{
				avgtime[j] = avgtime[j] + times[j][k];
				mintime[j] = MIN(mintime[j], times[j][k]);
				maxtime[j] = MAX(maxtime[j], times[j][k]);
			}
		}
    
		// note that "bytes[j]" is the aggregate array size, so no "numranks" is needed here
    FILE *logFile = fopen("time.log", "a");
    fprintf(logFile,"Tot_mem;Affinity;Function;Avg time;Min time;Max time\n");
		printf("Function\t    Best MB/s\t    Avg time     Min time     Max time\n");
		for (j=0; j<NBENCH; j++) {
			avgtime[j] = avgtime[j]/(double)(NTIMES-2);

			printf("%s\t\t  %11.1f\t %11.8f  %11.8f  %11.8f\n", label[j],
			   1.0E-06 * bytes[j]/mintime[j],
			   avgtime[j],
			   mintime[j],
			   maxtime[j]);
			fprintf(logFile,"%d;%s;%s;%11.8f;%11.8f;%11.8f\n", N, affinity, label[j],avgtime[j],mintime[j],maxtime[j]);
		}
   fclose(logFile);
	}
	free(a1);
	free(b2);
	free(c2);
	free(d3);
	free(e3);
	free(f3);
	free(rand_list);
	if (myrank == 0) {
		free(TimesByRank);
	}

    MPI_Finalize();
    
	return(0);
}

