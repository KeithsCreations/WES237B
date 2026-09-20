#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include "img.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                              \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define BLUR_PATH "kernels/blur.cl"
#define SHARPEN_PATH "kernels/sharpen.cl"
#define H_EDGE_PATH "kernels/h_edge.cl"
#define V_EDGE_PATH "kernels/v_edge.cl"

void OpenCLImageProc(Matrix *input0, Matrix *result, int select)
{
    // Load external OpenCL kernel code
    char *kernel_source;
    if (select == 0) { kernel_source = OclLoadKernel(BLUR_PATH); }
    else if (select == 1) { kernel_source = OclLoadKernel(SHARPEN_PATH); }
    else if (select == 2) { kernel_source = OclLoadKernel(H_EDGE_PATH); }
    else if (select == 3) { kernel_source = OclLoadKernel(V_EDGE_PATH); }

    // char *kernel_source = OclLoadKernel(KERNEL_PATH);

    // Device input and output buffers
    cl_mem device_a, device_b;

    //size_t global_item_size, local_item_size;
    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
# if __APPLE__
    queue = clCreateCommandQueue(context, device_id, 0, &err);
#else
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
#endif
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
    char *buff_erro;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
                printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
                exit(-1);
            }

        buff_erro = malloc(build_log_len);
        if (!buff_erro) {
            printf("malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }

        fprintf(stderr,"Build log: \n%s\n", buff_erro); //Be careful with  the fprint
        free(buff_erro);
        fprintf(stderr,"clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "imgproc", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Allocate GPU memory here
    device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, input0->shape[0] * input0->shape[1] * sizeof(int), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_a");

    // device_b = clCreateBuffer(context, CL_MEM_READ_ONLY, input1->shape[0] * input1->shape[1] * sizeof(int), NULL, &err);
    // CHECK_ERR(err, "clCreateBuffer device_b");

    device_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, result->shape[0] * result->shape[1] * sizeof(int), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_c");

    //@@ Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, input0->shape[0] * input0->shape[1] * sizeof(int), input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer deivce_a");

    // err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, input1->shape[0] * input1->shape[1] * sizeof(int), input1->data, 0, NULL, NULL);
    // CHECK_ERR(err, "clEnqueueWriteBuffer deivce_b");

    //@@ define local and global work sizes
    // local_item_size = 1;
    // unsigned int size_a = input0->shape[0] * input0->shape[1]; 
    // global_item_size = size_a;

    // Set the arguments to our compute kernel
    // __global const int *A, __global const int *B, __global int *C,
    // const unsigned int numARows, const unsigned int numAColumns,
    // const unsigned int numBRows, const unsigned int numBColumns,
    // const unsigned int numCRows, const unsigned int numCColumns
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_b);
    CHECK_ERR(err, "clSetKernelArg 1");
    // err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_c);
    // CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &input0->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &input0->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 3");
    // err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &result->shape[0]);
    // CHECK_ERR(err, "clSetKernelArg 7");
    // err |= clSetKernelArg(kernel, 8, sizeof(unsigned int), &result->shape[1]);
    // CHECK_ERR(err, "clSetKernelArg 8");

    // const int threadSize = 4;
    // const size_t local_item_size[2] = {width, height};
    //unsigned int size_a = input0->shape[0] * input0->shape[1]; 
    const size_t global_item_size[2] = {input0->shape[0], input0->shape[1]};

    //@@ Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &global_item_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel kernel");
    clFinish(queue);

    //@@ Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_b, CL_TRUE, 0, result->shape[0] * result->shape[1] * sizeof(int), result->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer deivce_c");

    //@@ Free the GPU memory here
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Release Host Memory
    // free(kernel_source);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <effect>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];

    int select;
    long conv = strtol(argv[3], NULL, 10);

    select = conv;
    
    printf("select: %d\n", select);

    if (select < 0 || select > 3)
    {
        fprintf(stderr, "effect must be from 0 - 3 inclusive!\n");
        return -1;
    }
    // Host input and output vectors and sizes
    Matrix host_a, host_b;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadImg");

    // err = LoadImgRaw(input_file_b, &host_b);
    // CHECK_ERR(err, "LoadImg");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer matrix (why is this snippet still here??)
    rows = host_a.shape[0];
    cols = host_a.shape[1];

    // Allocate the memory for the target.
    host_b.shape[0] = rows;
    host_b.shape[1] = cols;
    host_b.data = (int *)calloc(host_b.shape[0] * host_b.shape[1], sizeof(int));

    // printf("Did I make it at line 197\n");
    // Call your matrix multiply.
    OpenCLImageProc(&host_a, &host_b, select);

    // // Call to print the matrix
    // PrintMatrix(&host_c);

    // Save the matrix
    SaveMatrix(input_file_b, &host_b);

    // printf(host_b.data[0]);

    // Check the result of the matrix multiply
    // CheckMatrix(&answer, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);

    return 0;
}
