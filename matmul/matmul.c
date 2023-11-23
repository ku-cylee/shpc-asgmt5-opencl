#define _GNU_SOURCE
#include "matmul.h"
#include "util.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);              \
    exit(EXIT_FAILURE);                                                        \
  }

#define GROUP_WIDTH   64
#define VECTOR_SIZE   16

static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;
static cl_mem a_d, b_d, c_d;

float *A_padded, *B_padded, *C_padded;

// rows, cols: size of original matrix
void apply_zero_padding(const float *src, float *dst, int rows, int cols) {
  int cols_padded = (cols + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;

  for (int r = 0; r < rows; r++) {
    memcpy(dst + r * cols_padded, src + r * cols, cols * sizeof(float));
  }
}

void remove_zero_padding(const float *src, float *dst, int rows, int cols) {
  int cols_padded = (cols + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;

  for (int r = 0; r < rows; r++) {
    memcpy(dst + r * cols, src + r * cols_padded, cols * sizeof(float));
  }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  int M_padded = (M + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;
  int K_padded = (K + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;
  int N_padded = (N + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;

  int A_needs_padding = !(M == M_padded && K == K_padded);
  int B_needs_padding = !(K == K_padded && N == N_padded);
  int C_needs_padding = !(M == M_padded && N == N_padded);

  if (A_needs_padding) apply_zero_padding(A, A_padded, M, K);
  if (B_needs_padding) apply_zero_padding(B, B_padded, K, N);

  err = clEnqueueWriteBuffer(
    queue, a_d, CL_TRUE,
    0, M_padded * K_padded * sizeof(float), A_needs_padding ? A_padded : A,
    0, NULL, NULL);
  CHECK_ERROR(err);

  err = clEnqueueWriteBuffer(
    queue, b_d, CL_TRUE,
    0, K_padded * N_padded * sizeof(float), B_needs_padding ? B_padded : B,
    0, NULL, NULL);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &M_padded);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &N_padded);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 5, sizeof(int), &K_padded);
  CHECK_ERROR(err);

  size_t global_work_size[2] = { M_padded, N_padded / VECTOR_SIZE };
  size_t local_work_size[2] = { GROUP_WIDTH, GROUP_WIDTH / VECTOR_SIZE };

  err = clEnqueueNDRangeKernel(
    queue, kernel, 2,
    NULL, global_work_size, local_work_size,
    0, NULL, NULL);
  CHECK_ERROR(err);

  err = clFinish(queue);
  CHECK_ERROR(err);

  err = clEnqueueReadBuffer(
    queue, c_d, CL_TRUE,
    0, M_padded * N_padded * sizeof(float), C_needs_padding ? C_padded : C,
    0, NULL, NULL);
  CHECK_ERROR(err);

  if (C_needs_padding) remove_zero_padding(C_padded, C, M, N);
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

static cl_program create_and_build_program_with_source(cl_context context,
                                                       cl_device_id device,
                                                       const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char *)malloc(source_size + 1);
  size_t ntotal = 0;
  while (ntotal < source_size) {
    int nread = fread(source_code, sizeof(char), source_size, file);
    ntotal += nread;
  }
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                      NULL, &log_size));
    char *log = (char *)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

void matmul_initialize(int M, int N, int K) {
  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  print_platform_info(platform);

  // Get OpenCL device (only 1)
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_source(context, device, "kernel.cl");

  // Extract kernel from compiled program
  kernel = clCreateKernel(program, "sgemm", &err);
  CHECK_ERROR(err);

  int M_padded = (M + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;
  int K_padded = (K + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;
  int N_padded = (N + GROUP_WIDTH - 1) / GROUP_WIDTH * GROUP_WIDTH;

  // Create GPU buffers
  a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M_padded * K_padded * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K_padded * N_padded * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M_padded * N_padded * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);

  A_padded = (float *)calloc(M_padded * K_padded, sizeof(float));
  B_padded = (float *)calloc(K_padded * N_padded, sizeof(float));
  C_padded = (float *)calloc(M_padded * N_padded, sizeof(float));
}

void matmul_finalize() {
  clReleaseMemObject(a_d);
  clReleaseMemObject(b_d);
  clReleaseMemObject(c_d);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}
