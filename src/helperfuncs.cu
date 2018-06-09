#include <stdio.h>

#include "helperfuncs.h"
#include "floatvolume.h"

// y-coordinate = a, x-coordinate = b
unsigned toIndex2D(unsigned a, unsigned b, unsigned blen) {
  return a * blen + b;
}

// y-coordinate = a, x-coordinate = b, z-cooridnate = c
unsigned toIndex3D(unsigned a, unsigned b, unsigned blen, unsigned c,
  unsigned clen) {
  return a * blen * clen + b * clen + c;
}

// Given two colors, determine differentce
float diffColor(unsigned char *c1, unsigned char *c2) {
  return sqrtf(
    powf((float) *(c1 + 0) - (float) *(c2 + 0), 2.f) +
    powf((float) *(c1 + 1) - (float) *(c2 + 1), 2.f) +
    powf((float) *(c1 + 2) - (float) *(c2 + 2), 2.f) +
    powf((float) *(c1 + 3) - (float) *(c2 + 3), 2.f)
  );
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
    if(abs(*(fv1->contents + i) - *(fv2->contents + i)) > .1f) {
      printf("Contents don't match: abs(%f - %f) > .1f\n",
        *(fv1->contents + i), *(fv2->contents + i));
      return 1;
    }
  }

  return 0;
}
