#include <stdio.h>
#include <stdlib.h>
// For random number generator
#include <time.h>
// For square root
#include <math.h>

#include "helperfuncs.h"
#include "picture.h"
#include "floatvolume.h"

void setDiffVolumeSerial(struct FloatVolume *fv, struct Picture *picture1,
  struct Picture *picture2) {
  unsigned i, j, k;
  float *p1c, *p2c;

  // If pictures have differing dimensions, then quit
  if(picture1->width != picture2->width ||
    picture2->height != picture2->height) {
    printf("Pictures have different dimensions. Exiting setDiffVolmeSerial\n");
    return;
  }

  fv->height = picture1->height;
  fv->width = picture1->width;
  fv->depth = picture1->width;

  fv->contents = (float *) malloc(sizeof(float) * fv->height * fv->width *
    fv->depth);

  for(i = 0; i < fv->height; i++) {
    for(j = 0; j < fv->width; j++) {
      for(k = 0; k < fv->depth; k++) {
        // Get the index of the pixel of the first picture
        p1c = picture1->colors + toIndex2D(i, j, picture1->width) * 4;
        // Get the index of the pixel of the second picture
        p2c = picture2->colors + toIndex2D(i, k, picture2->width) * 4;

        // Insert the distance between these two colors into the float volume
        *(fv->contents + toIndex3D(i, j, fv->width, k, fv->depth)) =
          diffColor(p1c, p2c);

        /* *(fv->contents + toIndex3D(i, j, fv->width, k, fv->depth)) =
          toIndex3D(i, j, fv->width, k, fv->depth); */
      }
    }
  }
}

__global__ void setDiffVolumeKernel(float *d_fv, float *d_picture1,
  float *d_picture2, unsigned picWidth, unsigned picHeight) {
  __shared__ float p1_section[10 * 10 * 4];
  __shared__ float p2_section[10 * 10 * 4];
  unsigned i;

  // This thread's position in its block's subsection of the float volume
  unsigned sx, sy, sz;
  // Dimensions of the grid
  unsigned gx, gy, gz;
  // Position of this thread's block
  unsigned bx, by, bz;
  // This thread's position in the entire float volume
  unsigned vx, vy, vz;
  // The location of the colors that this thread will be comparing
  unsigned c1, c2;

  // Get the position of this thread in its subsection
  sz = threadIdx.x % 10;
  sy = threadIdx.x / 100;
  sx = (threadIdx.x % 100) / 10;

  // Get the dimensions of the grid
  gz = picWidth / 10 + (picWidth % 10);
  gy = picHeight / 10 + (picHeight % 10);
  gx = picWidth / 10 + (picWidth % 10);

  // Get the position of this thread's block
  bz = blockIdx.x % gz;
  by = blockIdx.x / (gx * gz);
  bx = (blockIdx.x % (gx * gz)) / gz;

  // Get the position of this thread in entire float volume
  vx = sx + 10 * bx;
  vy = sy + 10 * by;
  vz = sz + 10 * bz;

  // Copy subpicture to shared memory

  // See if this thread needs to copy from picture 1
  // picture 1 covers width * height

  // If the float volume z of this thread is zero,
  // then it needs to copy from picture 1
  if(sz == 0) {
    // Check if this thread will get a pixel not in the picture
    if(vx < picWidth && vy < picHeight) {
      for(i = 0; i < 4; i++) {
        p1_section[(sx + sy * 10) * 4 + i] =
          d_picture1[(vx + vy * picWidth) * 4 + i];
      }
    }
  }

  // See if this thread needs to copy from picture 2
  // picture 2 covers depth * height

  // If the float volume x of this thread is zero,
  // then it needs to copy from picture 2
  if(sx == 0) {
    // Check if this thread will get a pixel not in the picture
    if(vz < picWidth && vy < picHeight) {
      for(i = 0; i < 4; i++) {
        p2_section[(sz + sy * 10) * 4 + i] =
          d_picture2[(vz + vy * picWidth) * 4 + i];
      }
    }
  }

  __syncthreads();
  // Now each of d_picture1 and d_picture2 are properly filled out

  // Write difference into float volume
  if(vx < picWidth && vy < picHeight && vz < picWidth) {
    c1 = (sx + sy * 10) * 4;
    c2 = (sz + sy * 10) * 4;
    d_fv[vz + vx * picWidth + vy * picWidth * picWidth] =
      sqrtf(
        powf(p1_section[c1 + 0] - p2_section[c2 + 0], 2.f) +
        powf(p1_section[c1 + 1] - p2_section[c2 + 1], 2.f) +
        powf(p1_section[c1 + 2] - p2_section[c2 + 2], 2.f) +
        powf(p1_section[c1 + 3] - p2_section[c2 + 3], 2.f)
      );
  }
}

// TODO : make as parameters - max amount of memory, max number of blocks
void setDiffVolumeParallel(struct FloatVolume *fv, struct Picture *picture1,
  struct Picture *picture2) {
  // Memory locations of float arrays on the GPU
  float *d_fv, *d_picture1, *d_picture2;
  int fvDataLen;
  unsigned num_blocks;
  // Maximum dimensions of what subpictures can be held in the GPU
  // Ideally each of the dimensions should be multiples of 10
  unsigned max_height_gpu, max_width_gpu;
  // Maximum amount of floats that can be held in the float volume
  unsigned max_volume_gpu;
  // Maximum number of blocks that can be used per iteration
  unsigned max_blocks;
  // Based on the maximum that can be held in the GPU, get iterations
  unsigned num_iterations;

  // If pictures have differing dimensions, then quit
  if(picture1->width != picture2->width ||
    picture2->height != picture2->height) {
    printf(
      "Pictures have different dimensions. Exiting setDiffVolumeParallel\n");
    return;
  }

  // Create the FloatVolume
  fv->height = picture1->height;
  fv->width = picture1->width;
  fv->depth = picture1->width;

  fvDataLen = fv->height * fv->width * fv->depth;

  fv->contents = (float *) malloc(sizeof(float) * fvDataLen);

  /* // Set limitations
  max_height_gpu = 300;
  max_width_gpu = 300;
  max_volume_gpu = max_height_gpu * max_width_gpu;
  max_blocks = 27000;
  num_iterations = 1; */

  // Allocate space on the GPU
  cudaMalloc((void **) &d_fv, fvDataLen * sizeof(float));
  cudaMalloc((void **) &d_picture1, picture1->width * picture1->height *
    sizeof(float) * 4);
  cudaMalloc((void **) &d_picture2, picture1->width * picture1->height *
    sizeof(float) * 4);

  /* cudaMalloc((void **) &d_fv, max_volume_gpu * sizeof(float));
  cudaMalloc((void **) &d_picture1, max_height_gpu * max_width_gpu *
    sizeof(float) * 4); // 4 for RGBA
  cudaMalloc((void **) &d_picture2, max_height_gpu * max_width_gpu *
    sizeof(float) * 4); */

  // Give the pictures to the GPU
  // Params: destination, source, size of data to be copied, operation
  cudaMemcpy(d_picture1, picture1->colors,
    picture1->width * picture1->height * 4 * sizeof(float),
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_picture2, picture2->colors,
    picture1->width * picture1->height * 4 * sizeof(float),
    cudaMemcpyHostToDevice);

  // Kernel stuff
  // 1000 threads per block
  // So get 10 x 10 subset of each picture, with 4 colors each

  // Get the number of blocks this program will use
  // TODO : Assume that the maximum number of blocks that can run
  //   at the same time is unlimited
  num_blocks = (fv->height / 10 + (fv->height % 10)) *
    (fv->width / 10 + (fv->width % 10)) *
    (fv->depth / 10 + (fv->depth % 10));

  printf("num_blocks: %d\n", num_blocks);

  dim3 dimGrid(num_blocks);
  dim3 dimBlock(1000);

  // Do it
  setDiffVolumeKernel<<<dimGrid, dimBlock>>>(d_fv, d_picture1, d_picture2,
    picture1->width, picture1->height);

  // Copy the float volume back into host memory
  cudaMemcpy(fv->contents, d_fv, fvDataLen * sizeof(float),
    cudaMemcpyDeviceToHost);

  // Clear memory
  cudaFree(d_fv);
  cudaFree(d_picture1);
  cudaFree(d_picture2);
}

// Function too thicc
void pathVolumeInit(struct FloatVolume *pv, struct FloatVolume *dv) {
  int fvDataLen;
  unsigned i, j;
  float candidates2D[3], minCandidate;

  // Create the FloatVolume
  pv->height = dv->height;
  pv->width = dv->width;
  pv->depth = dv->depth;

  fvDataLen = pv->height * pv->width * pv->depth;

  pv->contents = (float *) malloc(sizeof(float) * fvDataLen);

  // TESTING : Set all cells in fv2 to 0
  for(i = 0; i < pv->depth * pv->width * pv->height; i++) {
    *(pv->contents + i) = 0.f;
  }

  // Set the first cell
  *(pv->contents + 0) = *(dv->contents + 0);

  // Fill cells where x = 0 and y = 0
  for(i = 1; i < pv->depth; i++) {
    *(pv->contents + toIndex3D(0, 0, pv->width, i, pv->depth)) =
      *(dv->contents + toIndex3D(0, 0, pv->width, i, pv->depth)) +
      *(pv->contents + toIndex3D(0, 0, pv->width, i - 1, pv->depth));
  }

  // Fill cells where z = 0 and y = 0
  for(i = 1; i < pv->width; i++) {
    *(pv->contents + toIndex3D(0, i, pv->width, 0, pv->depth)) =
      *(dv->contents + toIndex3D(0, i, pv->width, 0, pv->depth)) +
      *(pv->contents + toIndex3D(0, i - 1, pv->width, 0, pv->depth));
  }

  // Fill cells where z = 0 and x = 0
  for(i = 1; i < pv->height; i++) {
    *(pv->contents + toIndex3D(i, 0, pv->width, 0, pv->depth)) =
      *(dv->contents + toIndex3D(i, 0, pv->width, 0, pv->depth)) +
      *(pv->contents + toIndex3D(i - 1, 0, pv->width, 0, pv->depth));
  }

  // Fill cells where x = 0
  for(i = 1; i < pv->height; i++) {
    for(j = 1; j < pv->depth; j++) {
      candidates2D[0] =
        *(pv->contents + toIndex3D(i, 0, pv->width, j - 1, pv->depth));
      candidates2D[1] =
        *(pv->contents + toIndex3D(i - 1, 0, pv->width, j - 1, pv->depth));
      candidates2D[2] =
        *(pv->contents + toIndex3D(i - 1, 0, pv->width, j, pv->depth));

      minCandidate = candidates2D[0];
      if(candidates2D[1] < minCandidate)
        minCandidate = candidates2D[1];
      if(candidates2D[2] < minCandidate)
        minCandidate = candidates2D[2];

      *(pv->contents + toIndex3D(i, 0, pv->width, j, pv->depth)) =
        *(dv->contents + toIndex3D(i, 0, pv->width, j, pv->depth)) +
        minCandidate;
    }
  }

  // Fill cells where y = 0
  for(i = 1; i < pv->width; i++) {
    for(j = 1; j < pv->depth; j++) {
      candidates2D[0] =
        *(pv->contents + toIndex3D(0, i, pv->width, j - 1, pv->depth));
      candidates2D[1] =
        *(pv->contents + toIndex3D(0, i - 1, pv->width, j - 1, pv->depth));
      candidates2D[2] =
        *(pv->contents + toIndex3D(0, i - 1, pv->width, j, pv->depth));

      minCandidate = candidates2D[0];
      if(candidates2D[1] < minCandidate)
        minCandidate = candidates2D[1];
      if(candidates2D[2] < minCandidate)
        minCandidate = candidates2D[2];

      *(pv->contents + toIndex3D(0, i, pv->width, j, pv->depth)) =
        *(dv->contents + toIndex3D(0, i, pv->width, j, pv->depth)) +
        minCandidate;
    }
  }

  // Fill cells where z = 0
  for(i = 1; i < pv->height; i++) {
    for(j = 1; j < pv->width; j++) {
      candidates2D[0] =
        *(pv->contents + toIndex3D(i, j - 1, pv->width, 0, pv->depth));
      candidates2D[1] =
        *(pv->contents + toIndex3D(i - 1, j - 1, pv->width, 0, pv->depth));
      candidates2D[2] =
        *(pv->contents + toIndex3D(i - 1, j, pv->width, 0, pv->depth));

      minCandidate = candidates2D[0];
      if(candidates2D[1] < minCandidate)
        minCandidate = candidates2D[1];
      if(candidates2D[2] < minCandidate)
        minCandidate = candidates2D[2];

      *(pv->contents + toIndex3D(i, j, pv->width, 0, pv->depth)) =
        *(dv->contents + toIndex3D(i, j, pv->width, 0, pv->depth)) +
        minCandidate;
    }
  }
}

void setPathVolumeSerial(struct FloatVolume *pv, struct FloatVolume *dv) {
  unsigned i, j, k, l;
  float candidates3D[7], minCandidate;

  pathVolumeInit(pv, dv);

  // Finally fill in the remaining ones
  for(i = 1; i < pv->height; i++) {
    for(j = 1; j < pv->width; j++) {
      for(k = 1; k < pv->depth; k++) {
        /* candidates3D[0] = *(pv->contents +
          toIndex3D(i, j, pv->width, k - 1, pv->depth));
        candidates3D[1] = *(pv->contents +
          toIndex3D(i, j - 1, pv->width, k, pv->depth));
        candidates3D[2] = *(pv->contents +
          toIndex3D(i, j - 1, pv->width, k - 1, pv->depth));
        candidates3D[3] = *(pv->contents +
          toIndex3D(i - 1, j, pv->width, k, pv->depth));
        candidates3D[4] = *(pv->contents +
          toIndex3D(i - 1, j, pv->width, k - 1, pv->depth));
        candidates3D[5] = *(pv->contents +
          toIndex3D(i - 1, j - 1, pv->width, k, pv->depth));
        candidates3D[6] = *(pv->contents +
          toIndex3D(i - 1, j - 1, pv->width, k - 1, pv->depth));

        minCandidate = candidates3D[0];
        for(l = 1; l < 7; l++) {
          if(candidates3D[l] < minCandidate)
            minCandidate = candidates3D[l];
        }

        *(pv->contents + toIndex3D(i, j, pv->width, k, pv->depth)) =
          *(dv->contents + toIndex3D(i, j, pv->width, k, pv->depth)) +
          minCandidate; */

        *(pv->contents + toIndex3D(i, j, pv->width, k, pv->depth)) = 11.f;
      }
    }
  }
}

// Height, width, and height refer to the dimensions of the float volume
__global__ void setPathVolumeKernel(float *d_pv, float *d_dv, unsigned height,
  unsigned width, unsigned depth) {
  // The subvolume
  __shared__ float sv[11 * 11 * 11];
  float candidates3D[7], minCandidate;
  unsigned i, j;

  // This thread's position in its block's subsection of the float volume
  unsigned sx, sy, sz;
  // Dimensions of the grid
  unsigned gx, gy, gz;
  // Position of this thread's block
  unsigned bx, by, bz;
  // This thread's position in the entire float volume
  unsigned vx, vy, vz;

  // Get the position of this thread in its subsection
  sz = threadIdx.x % 10;
  sy = threadIdx.x / 100;
  sx = (threadIdx.x % 100) / 10;

  // Get the dimensions of the grid
  gz = (width - 1) / 10 + ((width - 1) % 10);
  gy = (height - 1) / 10 + ((height - 1) % 10);
  gx = (width - 1) / 10 + ((width - 1) % 10);

  // Get the position of this thread's block
  bz = blockIdx.x % gz;
  by = blockIdx.x / (gx * gz);
  bx = (blockIdx.x % (gx * gz)) / gz;

  // Get the position of this thread in entire float volume
  vx = sx + 10 * bx + 1;
  vy = sy + 10 * by + 1;
  vz = sz + 10 * bz + 1;

  // ez brute force... for demo purposes

  // Make each thread do work over and over until subvolume is filled
  /* for(i = 0; i < 10 + (10 - 1) + (10 - 1); i++) {
    if(vy < height && vx < width && vz < depth) {
      candidates3D[0] = d_pv[vy * width * depth + vx * depth + (vz - 1)];
      candidates3D[1] = d_pv[vy * width * depth + (vx - 1) * depth + vz];
      candidates3D[2] = d_pv[vy * width * depth + (vx - 1) * depth + (vz - 1)];
      candidates3D[3] = d_pv[(vy - 1) * width * depth + vx * depth + vz];
      candidates3D[4] = d_pv[(vy - 1) * width * depth + vx * depth + (vz - 1)];
      candidates3D[5] = d_pv[(vy - 1) * width * depth + (vx - 1) * depth + vz];
      candidates3D[6] =
        d_pv[(vy - 1) * width * depth + (vx - 1) * depth + (vz - 1)];

      minCandidate = candidates3D[0];
      for(j = 1; j < 7; j++) {
        if(candidates3D[j] < minCandidate)
          minCandidate = candidates3D[j];
      }

      d_pv[vy * width * depth + vx * depth + vz] = minCandidate +
        d_dv[vy * width * depth + vx * depth + vz];
    }
    __syncthreads();
  } */

  if(vy < height && vx < width && vz < depth) {
    d_pv[vy * width * depth + vx * depth + vz] = 11.f;
  }
}

// TODO : Currently the easy implementation
void setPathVolumeParallel(struct FloatVolume *pv, struct FloatVolume *dv) {
  // Memory locations of float volumes on the GPU
  float *d_pv, *d_dv;
  int fvDataLen, i;
  // y, x, z
  unsigned gdim[3];
  unsigned num_blocks, num_iter;

  // Serial implementation
  pathVolumeInit(pv, dv);
  fvDataLen = pv->height * pv->width * pv->depth;

  // Allocate space on the GPU
  cudaMalloc((void **) &d_pv, fvDataLen * sizeof(float));
  cudaMalloc((void **) &d_dv, fvDataLen * sizeof(float));

  // Give the diff volume to the GPU
  cudaMemcpy(d_dv, dv->contents, fvDataLen * sizeof(float),
    cudaMemcpyHostToDevice);
  // Give incomplete path volume to the GPU
  cudaMemcpy(d_pv, pv->contents, fvDataLen * sizeof(float),
    cudaMemcpyHostToDevice);

  // Kernel stuff
  // 1000 threads per block
  // Each block will get 11 x 11 x 11 subset of dv

  // Get the number of blocks this program will use
  // TODO : Assume that the maximum number of lbocks that can run
  //   at the same time is unlimited
  gdim[0] = ((pv->height - 1) / 10 + ((pv->height - 1) % 10));
  gdim[1] = ((pv->width - 1) / 10 + ((pv->width - 1) % 10));
  gdim[2] = ((pv->depth - 1) / 10 + ((pv->depth - 1) % 10));

  num_blocks = gdim[0] * gdim[1] * gdim[2];
  // Houdini stuff
  num_iter = gdim[0] + (gdim[1] - 1) + (gdim[2] - 1);

  dim3 dimGrid(num_blocks);
  dim3 dimBlock(1000);

  // for(i = 0; i < num_iter; i++) {
    // Each block will work on its own 10 x 10 x 10 portion
    // Will need info from the previous, so will need 11 x 11 x 11 portion

    // Dewit
    setPathVolumeKernel<<<dimGrid, dimBlock>>>(d_pv, d_dv, pv->height,
      pv->width, pv->depth);
  // }

  // Copy the path volume back into host memory
  cudaMemcpy(pv->contents, d_pv, fvDataLen * sizeof(float),
    cudaMemcpyDeviceToHost);

  // Clear memory
  cudaFree(d_dv);
  cudaFree(d_pv);

  /* num_blocks = ((fv->height - 1) / 10 + ((fv->height - 1) % 10)) *
    ((fv->width - 1) / 10 + ((fv->width - 1) % 10)) *
    ((fv->depth - 1) / 10 + ((fv->depth - 1) % 10)); */
}

int main() {
  struct Picture picture1, picture2;
  struct FloatVolume dvs, dvp;
  struct FloatVolume pvs, pvp;

  srand(time(NULL));

  // --- PICTURE CREATION SECTION ---------------------------------------------

  setRandomPicture(&picture1, 300, 300);
  setRandomPicture(&picture2, 300, 300);

  printf("--- picture1 ---\n");
  /* printPicture(&picture1);
  printf("\n"); */

  printf("--- picture2 ---\n");
  /* printPicture(&picture2);
  printf("\n"); */

  // --- DIFF VOLUME SECTION --------------------------------------------------

  setDiffVolumeSerial(&dvs, &picture1, &picture2);

  printf("--- diff volume serial ---\n");
  /* printFloatVolume(&dvs);
  printf("\n"); */

  setDiffVolumeParallel(&dvp, &picture1, &picture2);

  printf("--- diff volume parallel ---\n");
  /* printFloatVolume(&dvp);
  printf("\n"); */

  printf("--- diff volume comparison ---\n");
  printf("%d\n", compareFloatVolumes(&dvs, &dvp));
  printf("\n");

  // --- PATH VOLUME SECTION --------------------------------------------------

  /* setPathVolumeSerial(&pvs, &dvs);

  printf("--- path volume serial ---\n");
  printFloatVolume(&pvs);
  printf("\n");

  setPathVolumeParallel(&pvp, &dvp);

  printf("--- path volume parallel ---\n");
  printFloatVolume(&pvp);
  printf("\n");

  printf("--- path volume comparison ---\n");
  printf("%d\n", compareFloatVolumes(&pvs, &pvp));
  printf("\n"); */
}
