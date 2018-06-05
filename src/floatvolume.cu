#include <stdio.h>

#include "floatvolume.h"
#include "helperfuncs.h"

// NOTE: All values here are garbage,
void setEmptyFloatVolume(struct FloatVolume *fv, unsigned height,
  unsigned width, unsigned depth) {
  unsigned i;

  fv->width = width;
  fv->height = height;
  fv->depth = depth;

  fv->contents = (float *) malloc(sizeof(float) * fv->width * fv->height *
    fv->depth);
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
