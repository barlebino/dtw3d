#include <stdio.h>
#include <stdlib.h>
// For random number generator
#include <time.h>
// For square root
#include <math.h>

// Color at index 0 is top left, at index 0 is top left + 1 to the right
// All colors are in [0, 255]
struct Picture {
  unsigned width, height;
  float *colors;
};

// index = y * width * depth + x * depth + z
// given a cube with one face towards you,
// index 0 is at top left, closest to you
struct FloatVolume {
  unsigned width, height, depth;
  float *contents;
};

// y-coordinate = a, x-coordinate = b
unsigned toIndex2D(unsigned a, unsigned b, unsigned blen) {
  return a * blen + b;
}

// y-coordinate = a, x-coordinate = b, z-cooridnate = c
unsigned toIndex3D(unsigned a, unsigned b, unsigned blen, unsigned c,
  unsigned clen) {
  return a * blen * clen + b * clen + c;
}

// Sets the picture at location "picture" into a random picture of dimensions
// width and height
void setRandomPicture(struct Picture *picture, unsigned width,
  unsigned height) {
  unsigned i, j;
  float *currentColor;

  picture->width = width;
  picture->height = height;

  picture->colors = (float *) malloc(sizeof(float) *
    picture->width * picture->height * 4);

  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++) {
      currentColor = picture->colors + (i * width + j) * 4;

      *(currentColor + 0) = (float) (rand() % 256);
      *(currentColor + 1) = (float) (rand() % 256);
      *(currentColor + 2) = (float) (rand() % 256);
      *(currentColor + 3) = (float) (rand() % 256);
    }
  }
}

void printPicture(struct Picture *picture) {
  unsigned i, j;
  float *currentColor;

  for(i = 0; i < picture->height; i++) {
    for(j = 0; j < picture->width; j++) {
      currentColor = picture->colors + (i * picture->width + j) * 4;

      printf("(%u, %u): [%f, %f, %f, %f]\n", i, j,
        *(currentColor + 0),
        *(currentColor + 1),
        *(currentColor + 2),
        *(currentColor + 3)
      );
    }
  }
}

void setEmptyFloatVolume(struct FloatVolume *fv, unsigned width,
  unsigned height, unsigned depth) {
  unsigned i, j, k;

  fv->width = width;
  fv->height = height;
  fv->depth = depth;

  fv->contents = (float *) malloc(sizeof(float) * fv->width * fv->height *
    fv->depth);

  for(i = 0; i < fv->height; i++) {
    for(j = 0; j < fv->width; j++) {
      for(k = 0; k < fv->depth; k++) {
        *(fv->contents + toIndex3D(i, j, fv->width, k, fv->depth)) = 0.f;
      }
    }
  }
}

void printFloatVolume(struct FloatVolume *fv) {
  unsigned i, j, k;

  printf("Entering printFloatVolume...\n");

  for(i = 0; i < fv->height; i++) {
    for(j = 0; j < fv->width; j++) {
      for(k = 0; k < fv->depth; k++) {
        printf("(%u, %u, %u): %f\n", i, j, k,
           *(fv->contents + toIndex3D(i, j, fv->width, k, fv->depth)));
      }
    }
  }
}

// Given two colors, determine differentce
float diffColor(float *c1, float *c2) {
  return sqrtf(
    powf(*(c1 + 0) - *(c2 + 0), 2.f) +
    powf(*(c1 + 1) - *(c2 + 1), 2.f) +
    powf(*(c1 + 2) - *(c2 + 2), 2.f) +
    powf(*(c1 + 3) - *(c2 + 3), 2.f)
  );
}

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
  num_blocks = (fv->height / 10 + (fv->height % 10)) *
    (fv->width / 10 + (fv->width % 10)) *
    (fv->depth / 10 + (fv->depth % 10));

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

// Return 1 if difference, 0 if none
int compareFloatVolumes(struct FloatVolume *fv1, struct FloatVolume *fv2) {
  unsigned i;

  // Compare the dimensions
  if(fv1->width != fv2->width ||
    fv1->height != fv2->height ||
    fv1->depth != fv2->depth) {
    printf("Dimensions don't match\n");
    return 1;
  }

  // Compare the contents
  for(i = 0; i < fv1->width * fv1->height * fv1->depth; i++) {
    if(*(fv1->contents + i) - *(fv2->contents + i) > .001f) {
      printf("Contents don't match\n");
      return 1;
    }
  }

  return 0;
}

void setPathVolumeSerial(struct FloatVolume *pvs, struct FloatVolume *dv) {
  unsigned i, j, k, l;
  float candidates2D[3], candidates3D[7], minCandidate;

  // Set up the float volume
  pvs->depth = dv->depth;
  pvs->width = dv->width;
  pvs->height = dv->height;

  pvs->contents = (float *) malloc(sizeof(float) * pvs->depth * pvs->width *
    pvs->height);

  // TESTING : Set all cells in fv2 to 0
  for(i = 0; i < pvs->depth * pvs->width * pvs->height; i++) {
    *(pvs->contents + i) = 0.f;
  }

  // Set the first cells
  *(pvs->contents + 0) = *(dv->contents + 0);

  // Fill cells where x = 0 and y = 0
  for(i = 1; i < pvs->depth; i++) {
    // *(pvs->contents + toIndex3D(0, 0, pvs->width, i, pvs->depth)) = 1.f;
    *(pvs->contents + toIndex3D(0, 0, pvs->width, i, pvs->depth)) =
      *(dv->contents + toIndex3D(0, 0, pvs->width, i, pvs->depth)) +
      *(pvs->contents + toIndex3D(0, 0, pvs->width, i - 1, pvs->depth));
  }

  // Fill cells where z = 0 and y = 0
  for(i = 1; i < pvs->width; i++) {
    // *(pvs->contents + toIndex3D(0, i, pvs->width, 0, pvs->depth)) = 2.f;
    *(pvs->contents + toIndex3D(0, i, pvs->width, 0, pvs->depth)) =
      *(dv->contents + toIndex3D(0, i, pvs->width, 0, pvs->depth)) +
      *(pvs->contents + toIndex3D(0, i - 1, pvs->width, 0, pvs->depth));
  }

  // Fill cells where z = 0 and x = 0
  for(i = 1; i < pvs->height; i++) {
    // *(pvs->contents + toIndex3D(i, 0, pvs->width, 0, pvs->depth)) = 3.f;
    *(pvs->contents + toIndex3D(i, 0, pvs->width, 0, pvs->depth)) =
      *(dv->contents + toIndex3D(i, 0, pvs->width, 0, pvs->depth)) +
      *(pvs->contents + toIndex3D(i - 1, 0, pvs->width, 0, pvs->depth));
  }

  // Fill cells where x = 0
  for(i = 1; i < pvs->height; i++) {
    for(j = 1; j < pvs->depth; j++) {
      candidates2D[0] =
        *(pvs->contents + toIndex3D(i, 0, pvs->width, j - 1, pvs->depth));
      candidates2D[1] =
        *(pvs->contents + toIndex3D(i - 1, 0, pvs->width, j - 1, pvs->depth));
      candidates2D[2] =
        *(pvs->contents + toIndex3D(i - 1, 0, pvs->width, j, pvs->depth));

      minCandidate = candidates2D[0];
      if(candidates2D[1] < minCandidate)
        minCandidate = candidates2D[1];
      if(candidates2D[2] < minCandidate)
        minCandidate = candidates2D[2];

      *(pvs->contents + toIndex3D(i, 0, pvs->width, j, pvs->depth)) =
        *(dv->contents + toIndex3D(i, 0, pvs->width, j, pvs->depth)) +
        minCandidate;
    }
  }
 
  // Fill cells where y = 0
  for(i = 1; i < pvs->width; i++) {
    for(j = 1; j < pvs->depth; j++) {
      candidates2D[0] =
        *(pvs->contents + toIndex3D(0, i, pvs->width, j - 1, pvs->depth));
      candidates2D[1] =
        *(pvs->contents + toIndex3D(0, i - 1, pvs->width, j - 1, pvs->depth));
      candidates2D[2] =
        *(pvs->contents + toIndex3D(0, i - 1, pvs->width, j, pvs->depth));

      minCandidate = candidates2D[0];
      if(candidates2D[1] < minCandidate)
        minCandidate = candidates2D[1];
      if(candidates2D[2] < minCandidate)
        minCandidate = candidates2D[2];

      *(pvs->contents + toIndex3D(0, i, pvs->width, j, pvs->depth)) =
        *(dv->contents + toIndex3D(0, i, pvs->width, j, pvs->depth)) +
        minCandidate;
    }
  }

  // Fill cells where z = 0
  for(i = 1; i < pvs->height; i++) {
    for(j = 1; j < pvs->width; j++) {
      candidates2D[0] =
        *(pvs->contents + toIndex3D(i, j - 1, pvs->width, 0, pvs->depth));
      candidates2D[1] =
        *(pvs->contents + toIndex3D(i - 1, j - 1, pvs->width, 0, pvs->depth));
      candidates2D[2] =
        *(pvs->contents + toIndex3D(i - 1, j, pvs->width, 0, pvs->depth));

      minCandidate = candidates2D[0];
      if(candidates2D[1] < minCandidate)
        minCandidate = candidates2D[1];
      if(candidates2D[2] < minCandidate)
        minCandidate = candidates2D[2];

      *(pvs->contents + toIndex3D(i, j, pvs->width, 0, pvs->depth)) =
        *(dv->contents + toIndex3D(i, j, pvs->width, 0, pvs->depth)) +
        minCandidate;
    }
  }

  // Finally fill in the remaining ones
  for(i = 1; i < pvs->height; i++) {
    for(j = 1; j < pvs->width; j++) {
      for(k = 1; k < pvs->depth; k++) {
        candidates3D[0] = *(pvs->contents +
          toIndex3D(i, j, pvs->width, k - 1, pvs->depth));
        candidates3D[1] = *(pvs->contents +
          toIndex3D(i, j - 1, pvs->width, k, pvs->depth));
        candidates3D[2] = *(pvs->contents +
          toIndex3D(i, j - 1, pvs->width, k - 1, pvs->depth));
        candidates3D[3] = *(pvs->contents +
          toIndex3D(i - 1, j, pvs->width, k, pvs->depth));
        candidates3D[4] = *(pvs->contents +
          toIndex3D(i - 1, j, pvs->width, k - 1, pvs->depth));
        candidates3D[5] = *(pvs->contents +
          toIndex3D(i - 1, j - 1, pvs->width, k, pvs->depth));
        candidates3D[6] = *(pvs->contents +
          toIndex3D(i - 1, j - 1, pvs->width, k - 1, pvs->depth));

        minCandidate = candidates3D[0];
        for(l = 1; l < 7; l++) {
          if(candidates3D[l] < minCandidate)
            minCandidate = candidates3D[l];
        }
        
        *(pvs->contents + toIndex3D(i, j, pvs->width, k, pvs->depth)) =
          *(dv->contents + toIndex3D(i, j, pvs->width, k, pvs->depth)) +
          minCandidate;
      }
    }
  }
}

int main() {
  struct Picture picture1, picture2;
  struct FloatVolume dvs, dvp;
  struct FloatVolume pvs, pvp;

  srand(time(NULL));

  // --- PICTURE CREATION SECTION ---------------------------------------------

  setRandomPicture(&picture1, 3, 3);
  setRandomPicture(&picture2, 3, 3);

  printf("--- picture1 ---\n");
  /*printPicture(&picture1);
  printf("\n");*/

  printf("--- picture2 ---\n");
  /*printPicture(&picture2);
  printf("\n");*/

  // --- DIFF VOLUME SECTION --------------------------------------------------

  setDiffVolumeSerial(&dvs, &picture1, &picture2);

  printf("--- diff volume serial ---\n");
  /* printFloatVolume(&dvs);
  printf("\n"); */

  setDiffVolumeParallel(&dvp, &picture1, &picture2);

  printf("--- diff volume parallel ---\n");
  printFloatVolume(&dvp);
  printf("\n");

  printf("--- diff volume comparison ---\n");
  printf("%d\n", compareFloatVolumes(&dvs, &dvp));
  printf("\n");

  // --- PATH VOLUME SECTION --------------------------------------------------

  setPathVolumeSerial(&pvs, &dvs);
  printFloatVolume(&pvs);
  printf("\n");

  // setPathVolumeParallel(&pvp, &dvp);
}
