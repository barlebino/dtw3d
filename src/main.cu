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
    picture1->height != picture2->height) {
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
  gz = picWidth / 10;
  if(picWidth % 10) gz++;
  gy = picHeight / 10;
  if(picHeight % 10) gy++;
  gx = picWidth / 10;
  if(picWidth % 10) gx++;

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
  unsigned num_blocks, gdim[3];

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

  // Allocate space on the GPU
  cudaMalloc((void **) &d_fv, fvDataLen * sizeof(float));
  cudaMalloc((void **) &d_picture1, picture1->width * picture1->height *
    sizeof(float) * 4);
  cudaMalloc((void **) &d_picture2, picture1->width * picture1->height *
    sizeof(float) * 4);

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

  // y, then x, then z
  gdim[0] = fv->height / 10;
  if(fv->height % 10)
    gdim[0] = gdim[0] + 1;
  gdim[1] = fv->width / 10;
  if(fv->width % 10)
    gdim[1] = gdim[1] + 1;
  gdim[2] = fv->depth / 10;
  if(fv->depth % 10)
    gdim[2] = gdim[2] + 1;
  num_blocks = gdim[0] * gdim[1] * gdim[2];
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

// Inefficient but working
void setBigDiffVolumeParallel(struct FloatVolume *fv, struct Picture *picture1,
  struct Picture *picture2, unsigned subpicture_height) {
  unsigned numIterations, i;
  struct FloatVolume subfloatvolume;
  struct Picture subpicture1, subpicture2;
  unsigned subpicture_size, subfloatvolume_size;
  unsigned lastIterHeight, last_subpicture_size, last_subfloatvolume_size;
  int fvDataLen;

  // If pictures have differing dimensions, then quit
  if(picture1->width != picture2->width ||
    picture1->height != picture2->height) {
    printf(
      "Pictures have different dimensions. Exiting setBigDiffVolumeParallel");
    return;
  }

  // Check if bad subpicture_height value
  if(subpicture_height > picture1->height) {
    printf("subpicture_height > picture1->height\n");
    return;
  }

  // Allocate space for the final float volume
  fv->height = picture1->height;
  fv->width = picture1->width;
  fv->depth = picture1->width;
  fvDataLen = fv->height * fv->width * fv->depth;
  fv->contents = (float *) malloc(sizeof(float) * fvDataLen);

  // Clear float volume for testing
  for(i = 0; i < fvDataLen; i++) {
    *(fv->contents + i) = 0.f;
  }

  // Allocate space for each of the subpictures
  // (subfloatvolume will be allocated in setDiffVolumeParallel)
  subpicture1.width = picture1->width;
  subpicture1.height = subpicture_height;
  subpicture1.colors = (float *) malloc(sizeof(float) * subpicture1.width *
    subpicture1.height * 4); // RGBA
  // Note: dimensions of picture1 and picture2 are the same
  subpicture2.width = picture2->width;
  subpicture2.height = subpicture_height;
  subpicture2.colors = (float *) malloc(sizeof(float) * subpicture2.width *
    subpicture2.height * 4); // RGBA

  numIterations = picture1->height / subpicture_height +
    ((picture1->height % subpicture_height) > 0);
  // How many 32 bit floats are in one subpicture
  subpicture_size = subpicture1.height * subpicture1.width * 4; // RGBA
  // How many 32 bit floats are in one subfloatvolume
  subfloatvolume_size = subpicture1.height * subpicture1.width *
    subpicture2.width;

  /* printf("--- Info ---\n");
  printf("subpicture1.dim: %u, %u\n", subpicture1.width, subpicture1.height);
  printf("subpicture2.dim: %u, %u\n", subpicture2.width, subpicture2.height);
  printf("numIterations: %u\n", numIterations);
  printf("subpicture_size: %u\n", subpicture_size);
  printf("subfloatvolume_size: %u\n", subfloatvolume_size); */

  printf("numIterations: %u\n", numIterations);
  // numIterations - 1 because last iteration is a special case
  for(i = 0; i < numIterations - 1; i++) {
    // Load the subpictures
    memcpy(subpicture1.colors, picture1->colors + subpicture_size * i,
      subpicture_size * sizeof(float));
    memcpy(subpicture2.colors, picture2->colors + subpicture_size * i,
      subpicture_size * sizeof(float));

    // Call the normal diff volume function
    setDiffVolumeParallel(&subfloatvolume, &subpicture1, &subpicture2);

    // Copy the results of the subvolume into the final float volume
    // Float volume is allocated inside of this function
    memcpy(fv->contents + subfloatvolume_size * i, subfloatvolume.contents,
      subfloatvolume_size * sizeof(float));

    // Deallocate the subvolume
    free(subfloatvolume.contents);
  }
  printf("Entering last iteration...\n");

  free(subpicture1.colors);
  free(subpicture2.colors);

  // Take care of case where last iteration must process subpictures with
  //   smaller heights

  // Find out heights of subpictures
  if(picture1->height % subpicture_height) {
    lastIterHeight = picture1->height % subpicture_height;
    // subpicture1.height = picture1->height % subpicture_height;
    subpicture1.height = lastIterHeight;
    subpicture2.height = lastIterHeight;
  }

  // Reallocate subpictures
  subpicture1.colors = (float *) malloc(sizeof(float) * subpicture1.width *
    subpicture1.height * 4); // RGBA
  subpicture2.colors = (float *) malloc(sizeof(float) * subpicture2.width *
    subpicture2.height * 4); // RGBA

  // Recalculate sizes
  last_subpicture_size = subpicture1.height * subpicture1.width * 4; // RGBA
  last_subfloatvolume_size = subpicture1.height * subpicture1.width *
    subpicture2.width;

  // Load the subpictures
  memcpy(subpicture1.colors, picture1->colors + subpicture_size * i,
    last_subpicture_size * sizeof(float));
  memcpy(subpicture2.colors, picture2->colors + subpicture_size * i,
    last_subpicture_size * sizeof(float));

  // Call the normal diff volume Function
  setDiffVolumeParallel(&subfloatvolume, &subpicture1, &subpicture2);

  // Copy the results of the subvolume into the final float volume
  memcpy(fv->contents + subfloatvolume_size * i, subfloatvolume.contents,
    last_subfloatvolume_size * sizeof(float));
}

// Fill cells of path volume where x = 0 and y = 0
void setX0Y0(struct FloatVolume *pv, struct FloatVolume *dv) {
  unsigned i;
  for(i = 1; i < pv->depth; i++) {
    *(pv->contents + toIndex3D(0, 0, pv->width, i, pv->depth)) =
      *(dv->contents + toIndex3D(0, 0, pv->width, i, pv->depth)) +
      *(pv->contents + toIndex3D(0, 0, pv->width, i - 1, pv->depth));
  }
}

// Fill cells where z = 0 and y = 0
void setZ0Y0(struct FloatVolume *pv, struct FloatVolume *dv) {
  unsigned i;
  for(i = 1; i < pv->width; i++) {
    *(pv->contents + toIndex3D(0, i, pv->width, 0, pv->depth)) =
      *(dv->contents + toIndex3D(0, i, pv->width, 0, pv->depth)) +
      *(pv->contents + toIndex3D(0, i - 1, pv->width, 0, pv->depth));
  }
}

// Fill cells where z = 0 and x = 0
void setZ0X0(struct FloatVolume *pv, struct FloatVolume *dv) {
  unsigned i;
  for(i = 1; i < pv->height; i++) {
    *(pv->contents + toIndex3D(i, 0, pv->width, 0, pv->depth)) =
      *(dv->contents + toIndex3D(i, 0, pv->width, 0, pv->depth)) +
      *(pv->contents + toIndex3D(i - 1, 0, pv->width, 0, pv->depth));
  }
}

// Fill cells where x = 0, assuming Z0X0 and Z0Y0 are filled
void setX0(struct FloatVolume *pv, struct FloatVolume *dv) {
  float candidates2D[3], minCandidate;
  unsigned i, j;
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
}

// Fill cells where y = 0, assuming Z0Y0 and X0Y0 are filled
void setY0(struct FloatVolume *pv, struct FloatVolume *dv) {
  float candidates2D[3], minCandidate;
  unsigned i, j;

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
}

// Fill cells where z = 0, assuming Z0Y0 and Z0X0 are filled
void setZ0(struct FloatVolume *pv, struct FloatVolume *dv) {
  float candidates2D[3], minCandidate;
  unsigned i, j;

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
  setX0Y0(pv, dv);
  // Fill cells where z = 0 and y = 0
  setZ0Y0(pv, dv);
  // Fill cells where z = 0 and x = 0
  setZ0X0(pv, dv);
  // Fill cells where x = 0
  setX0(pv, dv);
  // Fill cells where y = 0
  setY0(pv, dv);
  // Fill cells where z = 0
  setZ0(pv, dv);
}

void setPathVolumeSerial(struct FloatVolume *pv, struct FloatVolume *dv) {
  unsigned i, j, k, l;
  float candidates3D[7], minCandidate;

  pathVolumeInit(pv, dv);

  // Finally fill in the remaining ones
  for(i = 1; i < pv->height; i++) {
    for(j = 1; j < pv->width; j++) {
      for(k = 1; k < pv->depth; k++) {
        candidates3D[0] = *(pv->contents +
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
          minCandidate;

        //*(pv->contents + toIndex3D(i, j, pv->width, k, pv->depth)) = 11.f;
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
  /* gz = (width - 1) / 10 + ((width - 1) % 10);
  gy = (height - 1) / 10 + ((height - 1) % 10);
  gx = (width - 1) / 10 + ((width - 1) % 10); */
  gz = (width - 1) / 10;
  if((width - 1) % 10) gz++;
  gy = (height - 1) / 10;
  if((height - 1) % 10) gy++;
  gx = (width - 1) / 10;
  if((width - 1) % 10) gx++;

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
  for(i = 0; i < 10 + (10 - 1) + (10 - 1); i++) {
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
  }

  /* if(vy < height && vx < width && vz < depth) {
    d_pv[vy * width * depth + vx * depth + vz] = 11.f;
  } */
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
  // TODO : Breaks when one of the pictures' dimensions is not more than or
  //   equal to 2
  /* gdim[0] = ((pv->height - 1) / 10 + ((pv->height - 1) % 10));
  gdim[1] = ((pv->width - 1) / 10 + ((pv->width - 1) % 10));
  gdim[2] = ((pv->depth - 1) / 10 + ((pv->depth - 1) % 10)); */
  gdim[0] = (pv->height - 1) / 10;
  if((pv->height - 1) % 10)
    gdim[0] = gdim[0] + 1;
  gdim[1] = (pv->width - 1) / 10;
  if((pv->width - 1) % 10)
    gdim[1] = gdim[1] + 1;
  gdim[2] = (pv->depth - 1) / 10;
  if((pv->depth - 1) % 10)
    gdim[2] = gdim[2] + 1;

  num_blocks = gdim[0] * gdim[1] * gdim[2];
  // Houdini stuff (Manhattan distance + 1)
  num_iter = gdim[0] + (gdim[1] - 1) + (gdim[2] - 1);

  dim3 dimGrid(num_blocks);
  dim3 dimBlock(1000);

  /* printf("fvDataLen: %u\n", fvDataLen);
  printf("gdim: [%u, %u, %u]\n", gdim[0], gdim[1], gdim[2]);
  printf("num_blocks: %u\n", num_blocks);
  printf("num_iter: %u\n", num_iter); */

  for(i = 0; i < num_iter; i++) {
    // Each block will work on its own 10 x 10 x 10 portion
    // Will need info from the previous, so will need 11 x 11 x 11 portion

    // Dewit
    setPathVolumeKernel<<<dimGrid, dimBlock>>>(d_pv, d_dv, pv->height,
      pv->width, pv->depth);
  }

  // Copy the path volume back into host memory
  cudaMemcpy(pv->contents, d_pv, fvDataLen * sizeof(float),
    cudaMemcpyDeviceToHost);

  // Clear memory
  cudaFree(d_dv);
  cudaFree(d_pv);
}

// Given a float volume with a complete y = 0, construct rest of float volume
void setSmallPathVolumeParallel(struct FloatVolume *pv,
  struct FloatVolume *dv) {
  // Memory locations of float volumes on the GPU
  float *d_pv, *d_dv;
  int fvDataLen, i;
  // y, x, z
  unsigned gdim[3];
  unsigned num_blocks, num_iter;

  // Fill cells where x = 0 and z = 0
  setZ0X0(pv, dv);
  // Fill cells where x = 0
  setX0(pv, dv);
  // Fill cells where z = 0
  setZ0(pv, dv);

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
  // TODO : Breaks when one of the pictures' dimensions is not more than or
  //   equal to 2
  /* gdim[0] = ((pv->height - 1) / 10 + ((pv->height - 1) % 10));
  gdim[1] = ((pv->width - 1) / 10 + ((pv->width - 1) % 10));
  gdim[2] = ((pv->depth - 1) / 10 + ((pv->depth - 1) % 10)); */
  gdim[0] = (pv->height - 1) / 10;
  if((pv->height - 1) % 10)
    gdim[0] = gdim[0] + 1;
  gdim[1] = (pv->width - 1) / 10;
  if((pv->width - 1) % 10)
    gdim[1] = gdim[1] + 1;
  gdim[2] = (pv->depth - 1) / 10;
  if((pv->depth - 1) % 10)
    gdim[2] = gdim[2] + 1;

  num_blocks = gdim[0] * gdim[1] * gdim[2];
  // Houdini stuff (Manhattan distance + 1)
  num_iter = gdim[0] + (gdim[1] - 1) + (gdim[2] - 1);

  dim3 dimGrid(num_blocks);
  dim3 dimBlock(1000);

  /* printf("fvDataLen: %u\n", fvDataLen);
  printf("gdim: [%u, %u, %u]\n", gdim[0], gdim[1], gdim[2]);
  printf("num_blocks: %u\n", num_blocks);
  printf("num_iter: %u\n", num_iter); */

  for(i = 0; i < num_iter; i++) {
    // Each block will work on its own 10 x 10 x 10 portion
    // Will need info from the previous, so will need 11 x 11 x 11 portion

    // Dewit
    setPathVolumeKernel<<<dimGrid, dimBlock>>>(d_pv, d_dv, pv->height,
      pv->width, pv->depth);
  }

  // Copy the path volume back into host memory
  cudaMemcpy(pv->contents, d_pv, fvDataLen * sizeof(float),
    cudaMemcpyDeviceToHost);

  // Clear memory
  cudaFree(d_dv);
  cudaFree(d_pv);
}

void setBigPathVolumeParallel(struct FloatVolume *pv, struct FloatVolume *dv,
  unsigned subvolume_height) {
  struct FloatVolume spv, sdv;
  unsigned subvolume_size, numIterations, i, lastIterHeight;
  float *y0buf;

  // Initialize the empty sub-pathvolume
  setEmptyFloatVolume(&spv, subvolume_height, dv->width, dv->depth);
  // Initialize the empty sub-diffvolume
  setEmptyFloatVolume(&sdv, subvolume_height, dv->width, dv->depth);
  // Initialize the empty final path volume
  setEmptyFloatVolume(pv, dv->height, dv->width, dv->depth);

  // Buffer holding the last y = 0 data
  y0buf = (float *) malloc(sizeof(float) * spv.width * spv.depth);

  // Creating the path volume will be done increments of sub-pathvolumes
  subvolume_size = subvolume_height * dv->width * dv->depth;
  // Find out how many sub-pathvolumes we will need to calculate
  // Every subvolume will calculate subvolume_size - 1 portion of the
  //   total subvolume, since the y = 0 of the subvolume is already in the
  //   total subvolume
  printf("subvolume_height: %u\n", subvolume_height);
  printf("pv->height - 1: %u\n", pv->height - 1);
  printf("spv.height - 1: %u\n", spv.height - 1);
  numIterations = (pv->height - 1) / (spv.height - 1);
  if((pv->height - 1) % (spv.height - 1)) numIterations++;

  // Set the very first cell in the final path volume
  *(pv->contents + 0) = *(dv->contents + 0);

  // Complete x = 0, y = 0
  setX0Y0(pv, dv);
  // Complete z = 0, y = 0
  setZ0Y0(pv, dv);
  // Complete y = 0
  setY0(pv, dv);

  for(i = 0; i < numIterations; i++) {
    // Resize the subvolumes if it is the last iteration
    if(i == numIterations - 1) {
      // Get the height of the volume of the last iteration
      lastIterHeight = ((pv->height - 1) % (spv.height - 1)) + 1;

      // Change dimensions of subvolumes
      spv.height = lastIterHeight;
      sdv.height = lastIterHeight;

      // Reallocate each of the subvolumes
      free(spv.contents);
      free(sdv.contents);
      spv.contents = (float *) malloc(sizeof(float) * spv.height * spv.width *
        spv.depth);
      sdv.contents = (float *) malloc(sizeof(float) * sdv.height * sdv.width *
        sdv.depth);
    }

    // Set the contents of the subvolumes

    /* // The path subvolume
    if(i == 0) {
      // Copy y = 0 to sub-pathvolume
      memcpy(spv.contents, pv->contents, sizeof(float) * spv.width * spv.depth);
    } else {
      // Copy y = spv.height - 1 from sub-pathvolume to y = 0 from
      // sub-pathvolume
      memcpy(spv.contents, y0buf, sizeof(float) * spv.width * spv.depth);
    } */

    // The sub diffvolume
    //memcpy(sdv.contents, dv->contents + subvolume_size, sizeof(float) *
    //  subvolume_size);
    memcpy(sdv.contents, dv->contents + (sdv.height - 1) * sdv.width *
      sdv.depth * i, sizeof(float) * sdv.height * sdv.width * sdv.depth);

    printf("--- sub-diffvolume %u ---\n", i);
    printFloatVolume(&sdv);

    /* // Complete path subvolume
    setSmallPathVolumeParallel(&spv, &sdv);

    // Copy the contents of the path subvolume to the total volume
    memcpy(pv->contents + pv->width * pv->depth + i *
      (spv.height - 1) * spv.width * spv.depth,
      spv.contents + spv.width * spv.depth,
      sizeof(float) * (spv.height - 1) * spv.width * spv.depth);

    // Copy the contents at y = max y to y = 0 within the path subvolume
    memcpy(y0buf, spv.contents + (spv.height - 1) * spv.width * spv.height,
      sizeof(float) * spv.width * spv.height); */
  }

  /* printf("--- State check ---\n");
  printf("subvolume_size: %u\n", subvolume_size);
  printf("numIterations: %u\n", numIterations);
  printf("-- sub-pathvolume --\n");
  printFloatVolume(&spv);
  printf("-- sub-diffvolume --\n");
  printFloatVolume(&sdv); */

  // Deallocation
  free(y0buf);
  free(spv.contents);
  free(sdv.contents);
}

int main() {
  struct Picture picture1, picture2;
  struct FloatVolume dvs, dvp;
  struct FloatVolume pvs, pvp;
  unsigned i, j, res;

  srand(time(NULL));

  for(i = 4; i < 25; i++) {
  for(j = 4; j < 25; j++) {

  printf("(%u, %u)\n", i , j);

  // --- PICTURE CREATION SECTION ---------------------------------------------

  /*setRandomPicture(&picture1, 6, 6);
  setRandomPicture(&picture2, 6, 6);*/
  setRandomPicture(&picture1, i, j);
  setRandomPicture(&picture2, i, j);

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

  // setDiffVolumeParallel(&dvp, &picture1, &picture2);
  setBigDiffVolumeParallel(&dvp, &picture1, &picture2, 2);

  printf("--- diff volume parallel ---\n");
  /* printFloatVolume(&dvp);
  printf("\n"); */

  printf("--- diff volume comparison ---\n");
  res = compareFloatVolumes(&dvs, &dvp);
  printf("%d\n", res);
  if(res != 0)
    exit(1);
  //printf("%d\n", compareFloatVolumes(&dvs, &dvp));
  printf("\n");

  // --- PATH VOLUME SECTION --------------------------------------------------

  setPathVolumeSerial(&pvs, &dvs);

  printf("--- path volume serial ---\n");
  /* printFloatVolume(&pvs);
  printf("\n"); */

  //setBigPathVolumeParallel(&pvp, &dvp, 3);
  // Print test volume
  /* printf("-- test volume --\n");
  printFloatVolume(&pvp); */
  // Deallocate test volume
  // free(pvp.contents);

  setPathVolumeParallel(&pvp, &dvp);

  printf("--- path volume parallel ---\n");
  /* printFloatVolume(&pvp);
  printf("\n"); */

  printf("--- path volume comparison ---\n");
  res = compareFloatVolumes(&pvs, &pvp);
  printf("%d\n", res);
  if(res != 0)
    exit(1);
  //printf("%d\n", compareFloatVolumes(&pvs, &pvp));
  printf("\n");

  // --- DEALLOCATION ---------------------------------------------------------

  free(picture1.colors);
  free(picture2.colors);
  free(dvs.contents);
  free(dvp.contents);
  free(pvs.contents);
  free(pvp.contents);

  }
  }
}
