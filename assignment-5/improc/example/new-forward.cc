#include <cmath>
#include <iostream>
#include <vector>
#include <CL/cl.h>
#include <clblast.h>

#include "kernel.h"
#include "device.h"

#include "opencl-new-forward.h"

#define CHECK_ERR(err, msg)                            \
    if (err != CL_SUCCESS)                             \
    {                                                  \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                            \
    }

void OpenCLInterface::conv_forward_gemm_opencl_prolog(const float *host_y, const float *host_x, const float *host_k,
                                                      cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, cl_mem *device_x_unroll,
                                                      const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;

    //@@ Allocate GPU memory here (don't forget batch sizes!)
    *device_x = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B * C * H * W * sizeof(float), (void*)host_x, &err);
    CHECK_ERR(err, "clCreateBuffer device_x");

    *device_k = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * K * K * C * sizeof(float), (void*)host_k, &err);
    CHECK_ERR(err, "clCreateBuffer device_k");

    *device_y = clCreateBuffer(this->opencl->context, CL_MEM_WRITE_ONLY, B * M * (H - K + 1) * (W - K + 1) * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_y");

    *device_x_unroll = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE, B * C * K * K * (H - K + 1) * (W - K + 1) * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_x_unroll");
    
    //@@ Copy memory to the GPU here
    // err = clEnqueueWriteBuffer(queue, *device_y, CL_TRUE, 0, input0->shape[0] * input0->shape[1] * sizeof(float), h_A, 0, NULL, NULL);
    // CHECK_ERR(err, "clEnqueueWriteBuffer input0");

    // err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, input1->shape[0] * input1->shape[1] * sizeof(float), h_B, 0, NULL, NULL);
    // CHECK_ERR(err, "clEnqueueWriteBuffer input1");

    // err = clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, result->shape[0] * result->shape[1] * sizeof(float), h_C, 0, NULL, NULL);
    // CHECK_ERR(err, "clEnqueueWriteBuffer result");
}

void OpenCLInterface::conv_forward_gemm_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const cl_mem device_x_unroll, 
                                               const int B, const int M, const int C, const int H, const int W, const int K)
{
    //@@ ====== Start im2col =====

    // @@ define local and global work sizes
    size_t global_size[3] = { (size_t)W, (size_t)H, (size_t)(B*C) };

    cl_int err;

    //@@ Launch the im2col kernel here
    cl_kernel kernel = clCreateKernel(this->opencl->program, "im2col", &err);
    CHECK_ERR(err, "clCreateKernel im2col");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_x_unroll);
    CHECK_ERR(err, "clSetKernelArg device_x_unroll");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg device_x");
    err = clSetKernelArg(kernel, 2, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg B");
    err = clSetKernelArg(kernel, 3, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg C");
    err = clSetKernelArg(kernel, 4, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg H");
    err = clSetKernelArg(kernel, 5, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg W");
    err = clSetKernelArg(kernel, 6, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg K");

    err = clEnqueueNDRangeKernel(this->opencl->queue, kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel im2col");

    //@@ ====== End im2col =====

    //@@ ====== Start gemm =====

    // GEMM sizes
    const size_t n = (H - K + 1) * (W - K + 1);
    const size_t m = M;
    const size_t k = C * K * K;

    std::vector<size_t> a_offsets = std::vector<size_t>(B,0);
    std::vector<size_t> b_offsets = std::vector<size_t>(B);
    std::vector<size_t> c_offsets = std::vector<size_t>(B);

    std::vector<float> alphas = std::vector<float>(B,1);
    std::vector<float> betas = std::vector<float>(B,0); // C is not accumulated into, so scale it by 0

    // Account for each batch of inputs occupying k*n elements
    for (size_t bn = 0; bn < B; bn++) 
    {
        b_offsets[bn] = bn * (k * n);  
        c_offsets[bn] = bn * (m * n);
    }

    // @@ Call clblast::GemmBatched here
    clblast::StatusCode clblast_err = clblast::GemmBatched(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
                                                            m, n, k,
                                                            alphas.data(),
                                                            device_k, a_offsets.data(), k,
                                                            device_x_unroll, b_offsets.data(), n,
                                                            betas.data(),
                                                            device_y, c_offsets.data(), n,
                                                            B,
                                                            &(this->opencl->queue), nullptr);
    CHECK_ERR((cl_int)clblast_err, "clblast::GemmBatched");

    clReleaseKernel(kernel);
    //@@ ====== End gemm =====
}

void OpenCLInterface::conv_forward_gemm_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, cl_mem device_x_unroll,
                                                      const int B, const int M, const int C, const int H, const int W, const int K)
{
    //@@ Copy the output back to host
    cl_int err;
    err = clEnqueueReadBuffer(this->opencl->queue, device_y, CL_TRUE, 0, B * M * (H - K + 1) * (W - K + 1) * sizeof(float), host_y, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device_y");

    //@@ Free the GPU memory here
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
    clReleaseMemObject(device_x_unroll);
    // clReleaseCommandQueue(this->opencl->queue);
    // clReleaseContext(this->opencl->context);

    clblast::ClearCache();
}
