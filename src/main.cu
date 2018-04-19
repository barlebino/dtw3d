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

/* void printColor(struct Color *color) {
  printf("[%f, %f, %f, %f]\n", color->r, color->g, color->b, color->a);
} */

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

/* // Returns 0 if equal pictures, 1 otherwise
int comparePictures(struct Picture *picture1, struct Picture *picture2) {
  unsigned i, j;
  struct Color *currentColor1, *currentColor2;

  // Dimension comparison
  if(picture1->width != picture2->width ||
    picture1->height != picture2->height) {
    return 1;
  }

  // Color comparison
  for(i = 0; i < picture1->height; i++) {
    for(j = 0; j < picture1->width; j++) {
      currentColor1 = picture1->colors + i * picture1->width + j;
      currentColor2 = picture2->colors + i * picture1->width + j;

      if(currentColor1->r != currentColor2->r ||
        currentColor1->g != currentColor2->g ||
        currentColor1->b != currentColor2->b ||
        currentColor1->a != currentColor2->a) {
        return 1;
      }
    }
  }
  
  return 0;
} */

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

void setDiffVolumeParallel(struct FloatVolume *fv, struct Picture *picture1,
  struct Picture *picture2) {
  // Memory locations of float arrays on the GPU
  float *d_fv, *d_picture1, *d_picture2;
  int fvDataLen;
  
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
    cudaMemcpyHostToDevice
  );
  cudaMemcpy(d_picture2, picture2->colors,
    picture1->width * picture1->height * 4 * sizeof(float),
    cudaMemcpyHostToDevice
  );

  // Kernel stuff
  

  // Clear memory
  cudaFree(d_fv);
  cudaFree(d_picture1);
  cudaFree(d_picture2);
}

int main() {
  struct Picture picture1, picture2;
  struct FloatVolume dvs, dvp;

  srand(time(NULL));

  setRandomPicture(&picture1, 2, 2);
  setRandomPicture(&picture2, 2, 2);

  printf("--- picture1 ---\n");
  printPicture(&picture1);
  printf("\n");

  printf("--- picture2 ---\n");
  printPicture(&picture2);
  printf("\n");

  setDiffVolumeSerial(&dvs, &picture1, &picture2);
  setDiffVolumeParallel(&dvp, &picture1, &picture2);

  printf("--- diff volume serial ---\n");
  printFloatVolume(&dvs);
  printf("\n");

  printf("--- diff volume parallel ---\n");
  printFloatVolume(&dvp);
  printf("\n");
}

