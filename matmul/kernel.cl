#define GROUP_WIDTH   64
#define VECTOR_SIZE   16
#define VECTOR        float16

__kernel void sgemm(
  __global VECTOR *A,
  __global VECTOR *B,
  __global VECTOR *C,
  int M, int N, int K) {

  int k, t, v;

  int local_row = get_local_id(0);
  int local_col = get_local_id(1);
  int global_row = GROUP_WIDTH * get_group_id(0) + local_row;
  int global_col = (GROUP_WIDTH / VECTOR_SIZE) * get_group_id(1) + local_col;

  __local VECTOR A_tile[GROUP_WIDTH][GROUP_WIDTH / VECTOR_SIZE];
  __local VECTOR B_tile[GROUP_WIDTH][GROUP_WIDTH / VECTOR_SIZE];

  VECTOR sum_vec = (VECTOR) 0.0f;

  for (k = 0; k < K; k += GROUP_WIDTH) {
    int tile_row = k + local_row;
    int tile_col = k / VECTOR_SIZE + local_col;

    A_tile[local_row][local_col] = A[global_row * (K / VECTOR_SIZE) + tile_col];
    B_tile[local_row][local_col] = B[tile_row * (N / VECTOR_SIZE) + global_col];

    barrier(CLK_LOCAL_MEM_FENCE);

    float *A_vec_arr;
    VECTOR A_vec, B_vec;
    for (t = 0; t < GROUP_WIDTH / VECTOR_SIZE; t++) {
      A_vec = A_tile[local_row][t];
      A_vec_arr = (float *)&A_vec;
      for (v = 0; v < VECTOR_SIZE; v++) {
        B_vec = B_tile[VECTOR_SIZE * t + v][local_col];
        sum_vec += A_vec_arr[v] * B_vec;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[global_row * (N / VECTOR_SIZE) + global_col] = sum_vec;
}
