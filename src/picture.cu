#include <stdio.h>

#include "picture.h"

// Sets the picture at location "picture" into a random picture of dimensions
// width and height
/*void setRandomPicture(struct Picture *picture, unsigned width,
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
}*/
void setRandomPicture(struct Picture *picture, unsigned width,
  unsigned height) {
  unsigned i, j;
  unsigned char *currentColor;

  picture->width = width;
  picture->height = height;

  picture->colors = (unsigned char *) malloc(sizeof(unsigned char) *
    picture->width * picture->height * 4);

  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++) {
      currentColor = picture->colors + (i * width + j) * 4;

      *(currentColor + 0) = (unsigned char) (rand() % 256);
      *(currentColor + 1) = (unsigned char) (rand() % 256);
      *(currentColor + 2) = (unsigned char) (rand() % 256);
      *(currentColor + 3) = (unsigned char) (rand() % 256);
    }
  }
}

/*void printPicture(struct Picture *picture) {
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
}*/
void printPicture(struct Picture *picture) {
  unsigned i, j;
  unsigned char *currentColor;

  for(i = 0; i < picture->height; i++) {
    for(j = 0; j < picture->width; j++) {
      currentColor = picture->colors + (i * picture->width + j) * 4;

      printf("(%u, %u): [%u, %u, %u, %u]\n", i, j,
        (unsigned) *(currentColor + 0),
        (unsigned) *(currentColor + 1),
        (unsigned) *(currentColor + 2),
        (unsigned) *(currentColor + 3)
      );
    }
  }
}
