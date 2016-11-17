/*
 * cholinv.c - Inversion of a real, symmetric positive definite matrix
 * given by its lower- or upper-triangular Cholesky factor.
 *
 * Uses LAPACK call [D|S]POTRI.
 *
 * TODO: add support for Hermitian positive definite matrices.
 *
 * See also CHOLSOLVE.
 */

#include "string.h"
#include "mex.h"
#include "matrix.h"

void dpotri_(const char*, const ptrdiff_t*, double*, const ptrdiff_t*,
  ptrdiff_t*);
void spotri_(const char*, const ptrdiff_t*, float*, const ptrdiff_t*,
  ptrdiff_t*);

void errorcheck(const ptrdiff_t info) {
  /*
   * Check lapack's info flag. Print a mex error message on error.
   */
  if (info < 0) {
    mexErrMsgIdAndTxt("cholinv:*potri:illegalvalue",
      "*potri error: illegal value.");
  } else if (info > 0) {
    mexErrMsgIdAndTxt("cholinv:*potri:singular",
      "*potri error: the matrix is singular.");
  }
}

void dtricpy(double *A, const ptrdiff_t *n, const char *uplo) {
  /*
   * Copy strictly lower or upper elements of matrix A to the other half.
   */
  ptrdiff_t i, j; /* element indices */

  if (!strcmp(uplo, "U")) {
    for (i = 0; i < *n; i++) {
      for (j = i + 1; j < *n; j++) {
        A[i*(*n) + j] = A[j*(*n) + i];
      }
    }
  } else {
    for (i = 0; i < *n; i++) {
      for (j = i + 1; j < *n; j++) {
        A[j*(*n) + i] = A[i*(*n) + j];
      }
    }
  }
}

void stricpy(float *A, const ptrdiff_t *n, const char *uplo) {
  /*
   * Copy strictly lower or upper elements of matrix A to the other half.
   */
  ptrdiff_t i, j; /* element indices */

  if (!strcmp(uplo, "U")) {
    for (i = 0; i < *n; i++) {
      for (j = i + 1; j < *n; j++) {
        A[i*(*n) + j] = A[j*(*n) + i];
      }
    }
  } else {
    for (i = 0; i < *n; i++) {
      for (j = i + 1; j < *n; j++) {
        A[j*(*n) + i] = A[i*(*n) + j];
      }
    }
  }
}

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
  mxClassID type; /* input matrix type */
  const char *uplo; /* upper / lower indicator */
  const char *inpbuf2; /* store input upper / lower string */
  ptrdiff_t info; /* lapack return flag */
  ptrdiff_t n; /* the number of columns / rows */
  mxArray *Ainv; /* a working matrix */

  if (nrhs < 1 || nrhs > 2) {
    mexErrMsgIdAndTxt("cholinv:nargin",
      "Wrong number of input arguments.");
  }

  if (nrhs == 2 && !mxIsChar(prhs[1])) {
    mexErrMsgIdAndTxt("cholinv:notstring",
      "Second input must be a string.");
  }

  if (nrhs == 2) {
    inpbuf2 = mxArrayToString(prhs[1]);

    if (!strcmp(inpbuf2, "lower")) {
      uplo = "L";
    } else if (!strcmp(inpbuf2, "upper")) {
      uplo = "U";
    } else {
      mexErrMsgIdAndTxt("cholinv:uplo", "Unknown value.");
    }
  } else { /* nrhs == 1 */
    uplo = "U";
  }

  n = (ptrdiff_t) mxGetM(prhs[0]);

  if (n != (ptrdiff_t) mxGetN(prhs[0])) {
    mexErrMsgIdAndTxt("cholinv:notsquare",
      "Input must be a square matrix.");
  }

  if (mxIsComplex(prhs[0])) {
    mexErrMsgIdAndTxt("cholinv:iscomplex",
      "Input matrix must be real.");
  }

  if (n == 0) {
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    return;
  }

  Ainv = mxDuplicateArray(prhs[0]);

  type = mxGetClassID(prhs[0]);

  if (type == mxDOUBLE_CLASS) {
    double *B = (double*) mxGetData(Ainv);

    /* compute the inverse */
    dpotri_(uplo, &n, B, &n, &info);

    /* check errors */
    errorcheck(info);

    /* mirror the matrix along the diagonal */
    dtricpy(B, &n, uplo);
  } else if (type == mxSINGLE_CLASS) {
    float *B = (float*) mxGetData(Ainv);

    /* compute the inverse */
    spotri_(uplo, &n, B, &n, &info);

    /* check errors */
    errorcheck(info);

    /* mirror the matrix along the diagonal */
    stricpy(B, &n, uplo);
  } else {
    mexErrMsgIdAndTxt("cholinv:type",
      "Unrecognized type of the input matrix.");
  }

  plhs[0] = Ainv;
}
