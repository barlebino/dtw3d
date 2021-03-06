#include <stdio.h>
#include <math.h>

#include "lodepng.h"

// Color at index 0 is top left, at index 0 is top left + 1 to the right
// All colors are in [0, 255]
struct Picture {
  unsigned width, height;
  unsigned char *colors;
};

__global__ void turnPictureKernel(unsigned char *d_in, unsigned char *d_out,
  unsigned picWidth, unsigned picHeight, double sn, double cs) {
  int x, y, index;
  double rotateX, rotateY;
  int tempX, tempY;
  unsigned i;

  double startVecX, startVecY;
  double endVecX, endVecY;

  // Samples of colors
  unsigned char c0[4], c1[4], c2[4], c3[4];

  // Get the coordinates of this thread
  index = blockIdx.x * 1024 + threadIdx.x;
  y = index / picWidth;
  x = index % picWidth;

  // Get vector from center of picture to pixel
  startVecX = (double) (x - (int) picWidth / 2);
  startVecY = (double) (y - (int) picHeight / 2);

  // Get new vector from center of picture to pixel
  endVecX = startVecX * cs - startVecY * sn;
  endVecY = startVecY * cs + startVecX * sn;

  // Get new coordinates from the end vector
  rotateX = endVecX + (double) picWidth / 2;
  rotateY = endVecY + (double) picHeight / 2;

  // Sample one set of coordinates from rotateX
  tempX = (int) floor(rotateX);
  tempY = (int) floor(rotateY);

  // Set the color
  if((tempX >= 0) && (tempX < picWidth) && (tempY >= 0) &&
    (tempY < picHeight)) {
    d_out[index * 4 + 0] = d_in[(tempX + tempY * picWidth) * 4 + 0];
    d_out[index * 4 + 1] = d_in[(tempX + tempY * picWidth) * 4 + 1];
    d_out[index * 4 + 2] = d_in[(tempX + tempY * picWidth) * 4 + 2];
    d_out[index * 4 + 3] = d_in[(tempX + tempY * picWidth) * 4 + 3];
  } else {
    d_out[index * 4 + 0] = 0;
    d_out[index * 4 + 1] = 0;
    d_out[index * 4 + 2] = 0;
    d_out[index * 4 + 3] = 0;
  }
}

// NOTE: Allocates the out picture
void turnPictureParallel(struct Picture *in, struct Picture *out,
  double radians) {
  // Memory locations of float arrays on the GPU
  unsigned char *d_in, *d_out;
  // Kernel stuff
  unsigned num_blocks;
  // Pass sine and cosine to the kernel so that it doesn't do more stuff
  double sn, cs;

  // Sine and cosine
  sn = sin(radians);
  cs = cos(radians);

  // Create the out picture
  out->width = in->width;
  out->height = in->height;
  out->colors = (unsigned char *) malloc(sizeof(unsigned char) * out->width *
    out->height * 4); // RGBA

  // Allocate space for both pictures on the GPU
  cudaMalloc((void **) &d_in, sizeof(unsigned char) * in->width *
    in->height * 4); // RGBA
  cudaMalloc((void **) &d_out, sizeof(unsigned char) * out->width *
    out->height * 4); // RGBA

  // Give the input picture to the GPU
  cudaMemcpy(d_in, in->colors, in->width * in->height *
    sizeof(unsigned char) * 4, cudaMemcpyHostToDevice); // RGBA

  // Kernel stuff
  // 1024 threads per block
  // TODO: Magic number
  num_blocks = (in->width * in->height) / 1024;
  if((in->width * in->height) % 1024)
    num_blocks = num_blocks + 1;

  dim3 dimGrid(num_blocks);
  dim3 dimBlock(1024);

  // Dewit
  turnPictureKernel<<<dimGrid, dimBlock>>>(d_in, d_out, in->width, in->height,
    sn, cs);

  // Copy the picture back into host memory
  cudaMemcpy(out->colors, d_out, out->width * out->height *
    sizeof(unsigned char) * 4, cudaMemcpyDeviceToHost);

  // Clear memory
  cudaFree(d_in);
  cudaFree(d_out);
}

int main() {
  struct Picture outPic, inPic;

  // Output of reading file
  unsigned char *out, *in;
  unsigned w, h;

  // Temporary variables
  unsigned i, j;
  unsigned pixelIndex;

  lodepng_decode32_file(&(inPic.colors), &(inPic.width), &(inPic.height),
    "anime.png");

  printf("Begin turn\n");
  turnPictureParallel(&inPic, &outPic, 3.14159f / 4.f);

  /*lodepng_decode32_file(&out, &w, &h, "anime.png");

  printf("(w, h): (%u, %u)\n", w, h);

  for(i = 0; i < h; i++) {
    for(j = 0; j < w; j++) {
      pixelIndex = (i * w + j) * 4; // 4 for RGBA

      *(out + pixelIndex + 0) = 255 - *(out + pixelIndex + 0);
      *(out + pixelIndex + 1) = 255 - *(out + pixelIndex + 1);
      *(out + pixelIndex + 2) = 255 - *(out + pixelIndex + 2);
    }
  }*/

  //lodepng_encode32_file("greenery.png", inPic.colors, inPic.width,
  //  inPic.height);
  lodepng_encode32_file("greenery.png", outPic.colors, outPic.width,
    outPic.height);

  // lodepng_encode32_file("anime-inverse.png", out, w, h);
}
