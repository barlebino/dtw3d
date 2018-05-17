#include "pathvolume.h"
#include "floatvolume.h"
#include "helperfuncs.h"

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
