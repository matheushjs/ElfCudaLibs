#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer
{
  cudaEvent_t d_start;
  cudaEvent_t d_stop;

  GpuTimer()
  {
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(d_start);
    cudaEventDestroy(d_stop);
  }

  void start()
  {
    cudaEventRecord(d_start, 0);
  }

  void stop()
  {
    cudaEventRecord(d_stop, 0);
  }

  float elapsed()
  {
    float d_elapsed;
    cudaEventSynchronize(d_stop);
    cudaEventElapsedTime(&d_elapsed, d_start, d_stop);
    return d_elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
