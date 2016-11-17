/*
 * cholsolve.c - Solve a system of linear equations AX = B for X,
 * where A is a symmetric positive definite matrix given by its
 * lower- or upper-triangular Cholesky factor.
 *
 * Uses LAPACK call [D|S]POTRF.
 *
 * TODO: add support for Hermitian positive definite matrices.
 *
 * See also CHOLINV.
 */

#include "string.h"
#include "mex.h"
#include "matrix.h"

void dpotrs_(const char*, const ptrdiff_t*, const ptrdiff_t*, const double*,
  const ptrdiff_t*, double*, const ptrdiff_t*, mwSignedIndex*);
void spotrs_(const char*, const ptrdiff_t*, const ptrdiff_t*, const float*,
  const ptrdiff_t*, float*, const ptrdiff_t*, mwSignedIndex*);

void errorcheck(const ptrdiff_t info) {
  /*
   * Check lapack's info flag. Print a mex error message on error.
   */
  if (info < 0) {
    mexErrMsgIdAndTxt("cholsolve:*potrs:illegalvalue",
      "*potri error: illegal value.");
  } else if (info > 0) {
    mexErrMsgIdAndTxt("cholsolve:*potrs:unknownerror",
      "Unexpected error.");
  }
}

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
  mxClassID type; /* input matrices type */
  const char *uplo = "U"; /* upper / lower indicator */
  const char *inpbuf2 = ""; /* buffer for the third input */
  ptrdiff_t info; /* lapack return flag */
  ptrdiff_t mA; /* the number of rows of A */
  ptrdiff_t nA; /* the number of columns of A */
  ptrdiff_t mB; /* the number of rows of B */
  ptrdiff_t nB; /* the number of columns of B */
  mxArray *A, *B; /* working matrices */

  if (nrhs < 2 || nrhs > 3) {
    mexErrMsgIdAndTxt("cholsolve:nargin",
      "Wrong number of input arguments.");
  }

  if (nrhs >= 3 && !mxIsChar(prhs[2])) {
    mexErrMsgIdAndTxt("cholsolve:notstring",
      "Third input must be a string.");
  } else if (nrhs >= 3) {
    inpbuf2 = mxArrayToString(prhs[2]);
  } else {
    inpbuf2 = "";
  }

  if (!strcmp(inpbuf2, "lower")) {
    uplo = "L";
  } else if (!strcmp(inpbuf2, "upper")) {
    uplo = "U";
  } else if (nrhs >= 3) {
    mexErrMsgIdAndTxt("cholsolve:uplo", "Unknown value.");
  } else {
    uplo = "U";
  }

  mA = (ptrdiff_t) mxGetM(prhs[0]);
  nA = (ptrdiff_t) mxGetN(prhs[0]);

  mB = (ptrdiff_t) mxGetM(prhs[1]);
  nB = (ptrdiff_t) mxGetN(prhs[1]);

  if (nA != mA) {
    mexErrMsgIdAndTxt("cholsolve:notsquare",
      "Input must be a square matrix.");
  }

  if (nA != mB) {
    mexErrMsgIdAndTxt("cholsolve:dimensionmismatch",
      "Inner dimensions must match.");
  }

  if (mA == 0) {
    /* return an empty matrix */
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    return;
  }

  if (nB == 0) {
    /* return an empty mB x 0 matrix */
    plhs[0] = mxCreateDoubleMatrix(mB, 0, mxREAL);
    return;
  }

  type = mxGetClassID(prhs[0]);

  if (mxIsComplex(prhs[0]) || mxIsComplex(prhs[1])) {
    mexErrMsgIdAndTxt("cholsolve:iscomplex",
      "Input matrix must be real.");
  }

  if (type != mxGetClassID(prhs[1])) {
    mexErrMsgIdAndTxt("cholsolve:typemismatch",
      "The left and right hand side types must match.");
  }

  A = mxDuplicateArray(prhs[0]);
  B = mxDuplicateArray(prhs[1]);

  if (type == mxDOUBLE_CLASS) {
    double *Z = (double*) mxGetData(A);
    double *X = (double*) mxGetData(B);

    /* solve the linear equations */
    dpotrs_(uplo, &mA, &nB, Z, &mA, X, &mB, &info);

    /* check errors */
    errorcheck(info);
  } else if (type == mxSINGLE_CLASS) {
    float *Z = (float*) mxGetData(A);
    float *X = (float*) mxGetData(B);

    /* solve the linear equations */
    spotrs_(uplo, &mA, &nB, Z, &mA, X, &mB, &info);

    /* check errors */
    errorcheck(info);
  } else {
    mexErrMsgIdAndTxt("cholsolve:type",
      "Unrecognized type of the input matrix.");
  }

  /* B has been modified with the result */
  plhs[0] = B;
}
