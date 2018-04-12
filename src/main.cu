#include <stdio.h>
#include <stdlib.h>
// For random number generator
#include <time.h>
// For square root
#include <math.h>

// All members are in [0, 255]
struct Color {
  float r, g, b, a;
};

// Color at index 0 is top left, at index 0 is top left + 1 to the right
struct Picture {
  unsigned width, height;
  struct Color *colors;
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

float diffColor(struct Color *color1, struct Color *color2) {
  return sqrt(powf(color1->r - color2->r, 2.f) +
    powf(color1->g - color2->g, 2.f) +
    powf(color1->b - color2->b, 2.f) +
    powf(color1->a - color2->a, 2.f));
}

void setRandomColor(struct Color *color) {
  color->r = (float) (rand() % 256);
  color->g = (float) (rand() % 256);
  color->b = (float) (rand() % 256);
  color->a = (float) (rand() % 256);
}

// Sets the picture at location "picture" into a random picture of dimensions
// width and height
void setRandomPicture(struct Picture *picture, unsigned width,
  unsigned height) {
  unsigned i, j;
  struct Color *currentColor;

  picture->width = width;
  picture->height = height;
  
  picture->colors = (struct Color *) malloc(sizeof(struct Color) *
    picture->width * picture->height);

  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++) {
      currentColor = picture->colors + i * width + j;
      setRandomColor(currentColor);
    }
  }
}

void printColor(struct Color *color) {
  printf("[%f, %f, %f, %f]\n", color->r, color->g, color->b, color->a);
}

void printPicture(struct Picture *picture) {
  unsigned i, j;
  struct Color *currentColor;

  for(i = 0; i < picture->height; i++) {
    for(j = 0; j < picture->width; j++) {
      currentColor = picture->colors + i * picture->width + j;

      printf("(%u, %u): [%f, %f, %f, %f]\n",
        j, i, currentColor->r, currentColor->g, currentColor->b,
        currentColor-> a);
    }
  }
}

// Returns 0 if equal pictures, 1 otherwise
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

int main() {
  struct Picture picture1, picture2, picture3;
  struct Color color1, color2;
  struct FloatVolume fv;

  srand(time(NULL));

  setRandomPicture(&picture1, 3, 3);
  setRandomPicture(&picture2, 3, 3);
  setRandomPicture(&picture3, 2, 3);

  printf("--- picture1 ---\n");
  printPicture(&picture1);
  printf("\n");

  printf("--- picture2 ---\n");
  printPicture(&picture2);
  printf("\n");

  printf("--- picture3 ---\n");
  printPicture(&picture3);
  printf("\n");

  printf("--- picture1 vs picture1 ---\n");
  printf("%d\n\n", comparePictures(&picture1, &picture1));

  printf("--- picture1 vs picture2 ---\n");
  printf("%d\n\n", comparePictures(&picture1, &picture2));

  printf("--- picture1 vs picture3 ---\n");
  printf("%d\n\n", comparePictures(&picture1, &picture3));

  setRandomColor(&color1);
  setRandomColor(&color2);

  printf("--- color1 ---\n");
  printColor(&color1);
  printf("\n");

  printf("--- color2 ---\n");
  printColor(&color2);
  printf("\n");

  printf("--- color1 vs color2 ---\n");
  printf("%f\n", diffColor(&color1, &color2));
  printf("\n");

  setEmptyFloatVolume(&fv, 3, 3, 2);

  printf("--- float volume ---\n");
  printFloatVolume(&fv);
  printf("\n");

  printf("Hello world!\n");
}

