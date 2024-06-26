#define BUCKET_WARP_LOG_SIZE	5
#define BUCKET_WARP_N			1

#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				128
#define SIZE (1 << 22)

#define DATA_SIZE (1024)
#define MAX_SOURCE_SIZE (0x100000)
#define HISTOGRAM_SIZE (1024 * sizeof(unsigned int))


#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include "bucketsort.h"
#include <time.h>

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
					 int divisions, float min, float max, float *pivotPoints,
					 float histo_width);

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
const int histosize = 1024;
unsigned int* h_offsets = NULL;
unsigned int* d_offsets = NULL;
cl_mem d_offsets_buff;
int *d_indice = NULL;
cl_mem d_indice_buff;
cl_mem d_input_buff;
cl_mem d_indice_input_buff;
float *pivotPoints = NULL;
float *historesult = NULL;
float *l_pivotpoints = NULL;
cl_mem l_pivotpoints_buff;
unsigned int *d_prefixoffsets = NULL;
unsigned int *d_prefixoffsets_altered = NULL;
cl_mem d_prefixoffsets_buff;
cl_mem d_prefixoffsets_input_buff;
unsigned int *l_offsets = NULL;
cl_mem l_offsets_buff;
unsigned int *d_Result1024;

static cl_device_id device_id;            // compute device id
static cl_context bucketContext;                 // compute context
static cl_context histoContext;
static cl_command_queue bucketCommands;          // compute command queue
static cl_command_queue histoCommands;
static cl_program bucketProgram;                 // compute program
static cl_program histoProgram;
static cl_kernel bucketcountKernel;                   // compute kernel
static cl_kernel histoKernel;
static cl_kernel bucketprefixKernel;
static cl_kernel bucketsortKernel;
static cl_mem histoInput;
static cl_mem histoOutput;
static cl_mem bucketOutput;
static cl_int err;
static cl_uint num_platforms;
static cl_event histoEvent;
static cl_event bucketCountEvent;
static cl_event bucketPrefixEvent;
static cl_event bucketSortEvent;
static double sum = 0;

////////////////////////////////////////////////////////////////////////////////
// Initialize the bucketsort algorithm
////////////////////////////////////////////////////////////////////////////////
void init_bucketsort(int listsize)
{
    cl_uint num = 0;
    err = clGetPlatformIDs(0, NULL, &num);
    if (num == 0)
    {
        printf("Error: Failed to get platforms!\n");
        exit(1);
    }

    cl_platform_id platformID[num];
    err = clGetPlatformIDs(num, platformID, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get platforms!\n");
        exit(1);
    }

    err = clGetDeviceIDs(platformID[0], CL_DEVICE_TYPE_ALL,0,NULL,&num);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device IDs!\n");
        exit(1);
    }
    
    cl_device_id devices[num];
    err = clGetDeviceIDs(platformID[0],CL_DEVICE_TYPE_ALL,num,devices,NULL);
//    int gpu = 1;
//    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_ALL, 2, &device_id, NULL);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        exit(1);
    }
    char name[128];
    clGetDeviceInfo(devices[0],CL_DEVICE_NAME,128,name,NULL);

    
    bucketContext = clCreateContext(0, 1, &devices[0], NULL, NULL, &err);

    bucketCommands = clCreateCommandQueue(bucketContext, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    
	h_offsets = (unsigned int *) malloc(DIVISIONS * sizeof(unsigned int));
    for(int i = 0; i < DIVISIONS; i++){
        h_offsets[i] = 0;
    }
    d_offsets_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, DIVISIONS * sizeof(unsigned int),NULL,NULL);
	pivotPoints = (float *)malloc(DIVISIONS * sizeof(float));
    
    d_indice_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, listsize * sizeof(int),NULL,NULL);
    d_indice_input_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, listsize * sizeof(int),NULL,NULL);
    d_indice = (int *)malloc(listsize * sizeof(int));
	historesult = (float *)malloc(histosize * sizeof(float));
    l_pivotpoints = (float *)malloc(DIVISIONS*sizeof(float));
	l_pivotpoints_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, DIVISIONS * sizeof(float), NULL, NULL);
	l_offsets_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, DIVISIONS * sizeof(unsigned int), NULL, NULL);
    
	int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
	d_prefixoffsets_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, blocks * BUCKET_BLOCK_MEMORY * sizeof(int), NULL, NULL);
    d_prefixoffsets = (unsigned int *)malloc(blocks*BUCKET_BLOCK_MEMORY*sizeof(int));
    d_prefixoffsets_altered = (unsigned int *)malloc(blocks*BUCKET_BLOCK_MEMORY*sizeof(int));
    d_prefixoffsets_input_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, blocks * BUCKET_BLOCK_MEMORY * sizeof(int), NULL, NULL);
    bucketOutput = clCreateBuffer(bucketContext, CL_MEM_READ_WRITE, (listsize + (DIVISIONS*4))*sizeof(float), NULL, NULL);
    FILE *fp;
    const char fileName[]="./bucketsort_kernels.cl";
    size_t source_size;
    char *source_str;
    
    fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load bucket kernel.\n");
		exit(1);
	}
    
    
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
    
    bucketProgram = clCreateProgramWithSource(bucketContext, 1, (const char **) &source_str, (const size_t*)&source_size, &err);
    if (!bucketProgram)
    {
        printf("Error: Failed to create bucket compute program!\n");
        exit(1);
    }
    
    err = clBuildProgram(bucketProgram, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build bucket program executable!\n");
        clGetProgramBuildInfo(bucketProgram, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
}

////////////////////////////////////////////////////////////////////////////////
// Uninitialize the bucketsort algorithm
////////////////////////////////////////////////////////////////////////////////
void finish_bucketsort()
{
    clReleaseMemObject(d_offsets_buff);
    clReleaseMemObject(d_indice_buff);
    clReleaseMemObject(l_pivotpoints_buff);
    clReleaseMemObject(l_offsets_buff);
    clReleaseMemObject(d_prefixoffsets_buff);
    clReleaseMemObject(d_input_buff);
    clReleaseMemObject(d_indice_input_buff);
    clReleaseMemObject(bucketOutput);
    clReleaseProgram(bucketProgram);
    clReleaseKernel(bucketcountKernel);
    clReleaseKernel(bucketprefixKernel);
    clReleaseKernel(bucketsortKernel);
    clReleaseCommandQueue(bucketCommands);
    clReleaseContext(bucketContext);
	free(pivotPoints);
	free(h_offsets);
	free(historesult);
}

void histogramInit(int listsize) {
    cl_uint num = 0;
    err = clGetPlatformIDs(0, NULL, &num);
    if (num == 0)
    {
        printf("Error: Failed to get platforms!\n");
        exit(1);
    }

    cl_platform_id platformID[num];
    err = clGetPlatformIDs(num, platformID, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get platforms!\n");
        exit(1);
    }

    err = clGetDeviceIDs(platformID[0], CL_DEVICE_TYPE_ALL,0,NULL,&num);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device IDs!\n");
        exit(1);
    }
    
    cl_device_id devices[num];
    err = clGetDeviceIDs(platformID[0],CL_DEVICE_TYPE_ALL,num,devices,NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        exit(1);
    }

    char name[128];
    //    int gpu = 1;
    //    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_ALL, 2, &device_id, NULL);
    
    
    clGetDeviceInfo(devices[0],CL_DEVICE_NAME,128,name,NULL);
    
    printf("Using device: %s \n", name);
    
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformID[0],
        0
    };
    
    histoContext = clCreateContext(contextProperties, 1, &devices[0], NULL, NULL, &err);
    if (err != CL_SUCCESS || histoContext == NULL)
    {
        printf("Error: Failed to create context! err: %i\n", err);
        exit(1);
    }

    
    histoCommands = clCreateCommandQueue(histoContext, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || histoCommands == NULL)
    {
        printf("Error: Failed to create cmd queue! err: %i\n", err);
        exit(1);
    }
    histoInput = clCreateBuffer(histoContext,  CL_MEM_READ_ONLY,  listsize*(sizeof(float)), NULL, NULL);
    histoOutput = clCreateBuffer(histoContext, CL_MEM_READ_WRITE, 1024 * sizeof(unsigned int), NULL, NULL);
    if (histoInput == NULL|| histoOutput == NULL)
    {
        printf("Error: Failed to create buffers! err: %i\n", err);
        exit(1);
    }

    FILE *fp;
    const char fileName[]="./histogram1024.cl";
    size_t source_size;
    char *source_str;
    
    fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
    
    
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
    printf("source size: %zu\n", source_size);

    histoProgram = clCreateProgramWithSource(histoContext, 1, (const char **) &source_str, (const size_t*)&source_size, &err);
    if (err != CL_SUCCESS || histoProgram == NULL)
    {
        printf("Error: Failed to create compute program! err: %i\n", err);
        exit(1);
    }

    err = clBuildProgram(histoProgram, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(histoProgram, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    histoKernel = clCreateKernel(histoProgram, "histogram1024Kernel", &err);
    if (!histoKernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    
}
void histogram1024GPU(unsigned int *h_Result, float *d_Data, float minimum, float maximum,int listsize){
    err = clEnqueueWriteBuffer(histoCommands, histoInput, CL_TRUE, 0, listsize*sizeof(float), d_Data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(histoCommands, histoOutput, CL_TRUE, 0, DIVISIONS*sizeof(unsigned int), h_Result, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    err = 0;
    err  = clSetKernelArg(histoKernel, 0, sizeof(cl_mem), &histoOutput);
    err  = clSetKernelArg(histoKernel, 1, sizeof(cl_mem), &histoInput);
    err  = clSetKernelArg(histoKernel, 2, sizeof(float), &minimum);
    err  = clSetKernelArg(histoKernel, 3, sizeof(float), &maximum);
    err  = clSetKernelArg(histoKernel, 4, sizeof(int), &listsize);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    size_t global = 6144;
    size_t local;
#ifdef HISTO_WG_SIZE_0
    local = HISTO_WG_SIZE_0;
#else
    local = 96;
#endif
    err = clEnqueueNDRangeKernel(histoCommands, histoKernel, 1, NULL, &global, &local, 0, NULL, &histoEvent);
    if (err)
    {
        printf("Error: Failed to execute histogram kernel!\n");
        exit(1);
    }
    clWaitForEvents(1 , &histoEvent);
    clFinish(histoCommands);
    err = clEnqueueReadBuffer( histoCommands, histoOutput, CL_TRUE, 0, 1024 * sizeof(unsigned int), h_Result, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read histo output array! %d\n", err);
        exit(1);
    }
    clFinish(histoCommands);
}

void finish_histogram() {
    clReleaseProgram(histoProgram);
    clReleaseKernel(histoKernel);
    clReleaseCommandQueue(histoCommands);
    clReleaseContext(histoContext);
    clReleaseMemObject((histoInput));
    clReleaseMemObject((histoOutput));
}
////////////////////////////////////////////////////////////////////////////////
// Given the input array of floats and the min and max of the distribution,
// sort the elements into float4 aligned buckets of roughly equal size
////////////////////////////////////////////////////////////////////////////////
void bucketSort(float *d_input, float *d_output, int listsize,
				int *sizes, int *nullElements, float minimum, float maximum,
				unsigned int *origOffsets)
{
//	////////////////////////////////////////////////////////////////////////////
//	// First pass - Create 1024 bin histogram
//	////////////////////////////////////////////////////////////////////////////
    histogramInit(listsize);
    histogram1024GPU(h_offsets, d_input, minimum, maximum, listsize);
    finish_histogram();
    for(int i=0; i<histosize; i++) historesult[i] = (float)h_offsets[i];

//	///////////////////////////////////////////////////////////////////////////
//	// Calculate pivot points (CPU algorithm)
//	///////////////////////////////////////////////////////////////////////////
	calcPivotPoints(historesult, histosize, listsize, DIVISIONS,
                    minimum, maximum, pivotPoints,
                    (maximum - minimum)/(float)histosize);
//
//	///////////////////////////////////////////////////////////////////////////
//	// Count the bucket sizes in new divisions
//	///////////////////////////////////////////////////////////////////////////
    
    bucketcountKernel = clCreateKernel(bucketProgram, "bucketcount", &err);
    if (!bucketcountKernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create bucketsort compute kernel!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(bucketCommands, l_pivotpoints_buff, CL_TRUE, 0, DIVISIONS*sizeof(float), pivotPoints, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to l_pivotpoints source array!\n");
        exit(1);
    }
    
    d_input_buff = clCreateBuffer(bucketContext,CL_MEM_READ_WRITE, (listsize + (DIVISIONS*4))*sizeof(float),NULL,NULL);
    
    err = clEnqueueWriteBuffer(bucketCommands, d_input_buff, CL_TRUE, 0, (listsize + (DIVISIONS*4))*sizeof(float), d_input, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to d_input_buff source array!\n");
        exit(1);
    }
    err = 0;
    err  = clSetKernelArg(bucketcountKernel, 0, sizeof(cl_mem), &d_input_buff);
    err  = clSetKernelArg(bucketcountKernel, 1, sizeof(cl_mem), &d_indice_buff);
    err  = clSetKernelArg(bucketcountKernel, 2, sizeof(cl_mem), &d_prefixoffsets_buff);
    err  = clSetKernelArg(bucketcountKernel, 3, sizeof(cl_int), &listsize);
    err  = clSetKernelArg(bucketcountKernel, 4, sizeof(cl_mem), &l_pivotpoints_buff);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    int blocks =((listsize -1) / (BUCKET_THREAD_N*BUCKET_BAND)) + 1;
    size_t global[] = {blocks*BUCKET_THREAD_N,1,1};
    size_t local[] = {BUCKET_THREAD_N,1,1};
    
    err = clEnqueueNDRangeKernel(bucketCommands, bucketcountKernel, 3, NULL, global, local, 0, NULL, &bucketCountEvent);
    if (err)
    {
        printf("Error: Failed to execute bucket count kernel!\n");
        exit(1);
    }
    clWaitForEvents(1 , &bucketCountEvent);
    clFinish(bucketCommands);
    err = clEnqueueReadBuffer( bucketCommands, d_prefixoffsets_buff, CL_TRUE, 0, blocks * BUCKET_BLOCK_MEMORY * sizeof(unsigned int), d_prefixoffsets, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read prefix output array! %d\n", err);
        exit(1);
    }
    err = clEnqueueReadBuffer( bucketCommands, d_indice_buff, CL_TRUE, 0, listsize * sizeof(int), d_indice, 0, NULL, NULL );
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read indice output array! %d\n", err);
        exit(1);
    }
    clFinish(bucketCommands);

//
//	///////////////////////////////////////////////////////////////////////////
//	// Prefix scan offsets and align each division to float4 (required by
//	// mergesort)
//	///////////////////////////////////////////////////////////////////////////
#ifdef BUCKET_WG_SIZE_0
    size_t localpre[] = {BUCKET_WG_SIZE_0,1,1};
#else
    size_t localpre[] = {128,1,1};
#endif
    size_t globalpre[] = {(DIVISIONS),1,1};
    
    bucketprefixKernel = clCreateKernel(bucketProgram, "bucketprefixoffset", &err);
    if (!bucketprefixKernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create bucket prefix compute kernel!\n");
        exit(1);
    }


    err = clEnqueueWriteBuffer(bucketCommands, d_prefixoffsets_buff, CL_TRUE, 0, blocks * BUCKET_BLOCK_MEMORY * sizeof(int), d_prefixoffsets, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to prefix offsets source array!\n");
        exit(1);
    }
    err = 0;
    err  = clSetKernelArg(bucketprefixKernel, 0, sizeof(cl_mem), &d_prefixoffsets_buff);
    err  = clSetKernelArg(bucketprefixKernel, 1, sizeof(cl_mem), &d_offsets_buff);
    err  = clSetKernelArg(bucketprefixKernel, 2, sizeof(cl_int), &blocks);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    err = clEnqueueNDRangeKernel(bucketCommands, bucketprefixKernel, 3, NULL, globalpre, localpre, 0, NULL, &bucketPrefixEvent);
    if (err)
    {
        printf("%d Error: Failed to execute bucket prefix kernel!\n", err);
        exit(1);
    }
    clWaitForEvents(1 , &bucketPrefixEvent);
    clFinish(bucketCommands);
    err = clEnqueueReadBuffer( bucketCommands, d_offsets_buff, CL_TRUE, 0, DIVISIONS * sizeof(unsigned int), h_offsets, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read d_offsets output array! %d\n", err);
        exit(1);
    }
    err = clEnqueueReadBuffer( bucketCommands, d_prefixoffsets_buff, CL_TRUE, 0, blocks * BUCKET_BLOCK_MEMORY * sizeof(int), d_prefixoffsets_altered, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read d_offsets output array! %d\n", err);
        exit(1);
    }
    clFinish(bucketCommands);

//	// copy the sizes from device to host
	origOffsets[0] = 0;
	for(int i=0; i<DIVISIONS; i++){
		origOffsets[i+1] = h_offsets[i] + origOffsets[i];
		if((h_offsets[i] % 4) != 0){
			nullElements[i] = (h_offsets[i] & ~3) + 4 - h_offsets[i];
		}
		else nullElements[i] = 0;
	}
	for(int i=0; i<DIVISIONS; i++) sizes[i] = (h_offsets[i] + nullElements[i])/4;
	for(int i=0; i<DIVISIONS; i++) {
		if((h_offsets[i] % 4) != 0)	h_offsets[i] = (h_offsets[i] & ~3) + 4;
	}
	for(int i=1; i<DIVISIONS; i++) h_offsets[i] = h_offsets[i-1] + h_offsets[i];
	for(int i=DIVISIONS - 1; i>0; i--) h_offsets[i] = h_offsets[i-1];
	h_offsets[0] = 0;
    


//	///////////////////////////////////////////////////////////////////////////
//	// Finally, sort the lot
//	///////////////////////////////////////////////////////////////////////////
    bucketsortKernel = clCreateKernel(bucketProgram, "bucketsort", &err);
    if (!bucketsortKernel|| err != CL_SUCCESS)
    {
        printf("Error: Failed to create bucketsort compute kernel!\n");
        exit(1);
    }

    err = clEnqueueWriteBuffer(bucketCommands, l_offsets_buff, CL_TRUE, 0, DIVISIONS * sizeof(unsigned int), h_offsets, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to  l_offsets source array!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(bucketCommands, d_input_buff, CL_TRUE, 0, (listsize + (DIVISIONS*4))*sizeof(float), d_input, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to d_input_buff source array!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(bucketCommands, d_indice_input_buff, CL_TRUE, 0, listsize*sizeof(int), d_indice, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to d_input_buff source array!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(bucketCommands, d_prefixoffsets_input_buff, CL_TRUE, 0, blocks * BUCKET_BLOCK_MEMORY * sizeof(int), d_prefixoffsets_altered, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to prefix offsets source array!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(bucketCommands, bucketOutput, CL_TRUE, 0, (listsize + (DIVISIONS*4))*sizeof(float), d_output, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    size_t localfinal[] = {BUCKET_THREAD_N,1,1};
    blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
    size_t globalfinal[] = {blocks*BUCKET_THREAD_N,1,1};
    err = 0;
    err  = clSetKernelArg(bucketsortKernel, 0, sizeof(cl_mem), &d_input_buff);
    err  = clSetKernelArg(bucketsortKernel, 1, sizeof(cl_mem), &d_indice_input_buff);
    err  = clSetKernelArg(bucketsortKernel, 2, sizeof(cl_mem), &bucketOutput);
    err  = clSetKernelArg(bucketsortKernel, 3, sizeof(cl_int), &listsize);
    err  = clSetKernelArg(bucketsortKernel, 4, sizeof(cl_mem), &d_prefixoffsets_input_buff);
    err  = clSetKernelArg(bucketsortKernel, 5, sizeof(cl_mem), &l_offsets_buff);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    err = clEnqueueNDRangeKernel(bucketCommands, bucketsortKernel, 3, NULL, globalfinal, localfinal, 0, NULL, &bucketSortEvent);
    if (err)
    {
        printf("%d Error: Failed to execute bucketsort kernel!\n", err);
    }
    err = clEnqueueReadBuffer( bucketCommands, bucketOutput, CL_TRUE, 0, (listsize + (DIVISIONS*4))*sizeof(float), d_output, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read d_output array! %d\n", err);
    }
    clFinish(bucketCommands);

}
double getBucketTime() {
  return sum;
}
////////////////////////////////////////////////////////////////////////////////
// Given a histogram of the list, figure out suitable pivotpoints that divide
// the list into approximately listsize/divisions elements each
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
					 int divisions, float min, float max, float *pivotPoints, float histo_width)
{
	float elemsPerSlice = listsize/(float)divisions;
	float startsAt = min;
	float endsAt = min + histo_width;
	float we_need = elemsPerSlice;
	int p_idx = 0;
	for(int i=0; i<histosize; i++)
	{
		if(i == histosize - 1){
			if(!(p_idx < divisions)){
				pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
			}
			break;
		}
		while(histogram[i] > we_need){
			if(!(p_idx < divisions)){
				printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions);
				exit(0);
			}
			pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
			startsAt += (we_need/histogram[i]) * histo_width;
			histogram[i] -= we_need;
			we_need = elemsPerSlice;
		}
		// grab what we can from what remains of it
		we_need -= histogram[i];
        
		startsAt = endsAt;
		endsAt += histo_width;
	}
	while(p_idx < divisions){
		pivotPoints[p_idx] = pivotPoints[p_idx-1];
		p_idx++;
	}
}
