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

unsigned toIndex2D(unsigned a, unsigned b, unsigned blen) {
  return a * blen + b;
}

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

  // Get the position of this thread in its subsection
  sz = threadIdx.x % 10;
  sy = threadIdx.x / 100;
  sx = (threadIdx.x % 100) / 10;

  // Get the dimensions of the grid
  gx = picWidth / 10 + (picWidth % 10);
  gy = picHeight / 10 + (picHeight % 10);
  gz = picWidth / 10 + (picWidth % 10);

  // Get the position of this thread's block
  bz = blockIdx.x % gz;
  by = blockIdx.x / (gx * gz);
  bx = (blockIdx.x % (gx * gz)) / gz;

  // Get the position of this thread in entire float volume
  vx = sx + 10 * bx;
  vy = sy + 10 * by;
  vz = sz + 10 * bz;

  /* // Copy subpicture to shared memory

  // See if this thread needs to copy from picture 1
  // picture 1 covers width * height
  
  // If the float volume z of this thread is zero, 
  // then it needs to copy from picture 1
  if(sz == 0) {
    // Check if this thread will get a pixel not in the picture
    if(vx < picWidth && vy < picHeight) {
      p1_section[sx + sy * 10 + 0] = d_picture1[vx + vy * picWidth + 0];
      p1_section[sx + sy * 10 + 1] = d_picture1[vx + vy * picWidth + 1];
      p1_section[sx + sy * 10 + 2] = d_picture1[vx + vy * picWidth + 2];
      p1_section[sx + sy * 10 + 3] = d_picture1[vx + vy * picWidth + 3];
    }
  }

  // See if this thread needs to copy from picture 2
  // picture 2 covers depth * height
  
  // If the float volume x of this thread is zero,
  // then it needs to copy from picture 2
  if(sx == 0) {
    // Check if this thread will get a pixel not in the picture
    if(vz < picWidth && vy < picHeight) {
      p2_section[sx + sy * 10 + 0] = d_picture2[vx + vy * picWidth + 0];
      p2_section[sx + sy * 10 + 1] = d_picture2[vx + vy * picWidth + 1];
      p2_section[sx + sy * 10 + 2] = d_picture2[vx + vy * picWidth + 2];
      p2_section[sx + sy * 10 + 3] = d_picture2[vx + vy * picWidth + 3];
    }
  } */

  // Write into float volume
  if(vx < picWidth && vy < picHeight && vz < picWidth) {
    d_fv[vz + vx * picWidth + vy * picWidth * picHeight] =
      vz + vx * picWidth + vy * picWidth * picHeight;
  }
}

void setDiffVolumeParallel(struct FloatVolume *fv, struct Picture *picture1,
  struct Picture *picture2) {
  // Memory locations of float arrays on the GPU
  float *d_fv, *d_picture1, *d_picture2;
  int fvDataLen;
  unsigned num_blocks;
  
  // If pictures have differing dimensions, then quit
  if(picture1->width != picture2->width ||
    picture2->height != picture2->height) {
    printf("Pictures have different dimensions. Exiting setDiffVolmeSerial\n");
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
  // TODO : Assume that the maximum number of blocks in unlimited
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
    return 1;
  }

  // Compare the contents
  for(i = 0; i < fv1->width * fv1->height * fv1->depth; i++) {
    if(*(fv1->contents + i) != *(fv2->contents + i)) {
      return 1;
    }
  }

  return 0;
}

int main() {
  struct Picture picture1, picture2;
  struct FloatVolume dvs, dvp;

  srand(time(NULL));

  setRandomPicture(&picture1, 11, 11);
  setRandomPicture(&picture2, 11, 11);

  printf("--- picture1 ---\n");
  printPicture(&picture1);
  printf("\n");

  printf("--- picture2 ---\n");
  printPicture(&picture2);
  printf("\n");

  setDiffVolumeSerial(&dvs, &picture1, &picture2);
  setDiffVolumeParallel(&dvp, &picture1, &picture2);

  printf("%d\n", compareFloatVolumes(&dvs, &dvp));

  /* printf("--- diff volume serial ---\n");
  printFloatVolume(&dvs);
  printf("\n");

  printf("--- diff volume parallel ---\n");
  printFloatVolume(&dvp);
  printf("\n"); */
}

