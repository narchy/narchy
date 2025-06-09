package jcog.math;

import jcog.data.bit.MetalBitSet;

public enum MatrixDeterminant { ;

    private static final int BLOCK_SIZE = 32; // Tune based on (L1?) cache size
    private static final double epsilon = 1e-12;

    public static double determinantWithout(double[][] k, int row) {
        int n = k.length;
        if (row >= n) throw new IllegalArgumentException("Row index out of bounds");

        /*
         * Fast determinant calculation for small matrices (n ≤ 3)
         * For small matrices (n ≤ 3), use direct formulas for better performance
         */
        return switch (n) {
            case 1 ->  determinantWithout1(k, row);// 1x1 matrix
            case 2 ->  determinantWithout2(k, row); // 2x2 matrix
            case 3 ->  determinantWithout3(k, row); // 3x3 matrix
            default -> determinantWithoutN(k, row);
        };
    }

    private static double determinantWithout1(double[][] k, int row) {
        return row == 0 ? 1.0 : k[0][0];
    }

    private static double determinantWithout2(double[][] k, int row) {
        return (row == -1) ?
                    ((k[0][0] * k[1][1]) - (k[0][1] * k[1][0])) :
                    ((row == 0) ? k[1][1] : k[0][0]);
    }

    private static double determinantWithout3(double[][] k, int row) {
        if (row == -1)
            return k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1])
                 - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0])
                 + k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]);

        // 3x3 with row removal - compute 2x2 determinant efficiently
        double[][] sub = new double[2][2];
        int si = 0;
        for (int i = 0; i < 3; i++) {
            if (i != row) {
                int sj = 0;
                for (int j = 0; j < 3; j++) {
                    if (j != row)
                        sub[si][sj++] = k[i][j];
                }
                si++;
            }
        }
        return sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0];
    }

    /**
     * Optimized in-place LU decomposition for larger matrices.
     * Uses blocked algorithm for better cache performance.
     *
     * Avoids creating submatrix by tracking skipped row/col in-place
     */
    private static double determinantWithoutN(double[][] k, int skipRow) {
        int n = k.length;
        int size = skipRow == -1 ? n : n - 1;

        // Reuse arrays to minimize allocations
        double[] tempRow = new double[n];
        int[] pivots = new int[size];
        MetalBitSet used = MetalBitSet.bits(n);
        int sign = 1;

        // Initialize pivot array and mark skipped row
        if (skipRow != -1) used.set(skipRow);

        double det = 1;

        for (int start = 0; start < size; start += BLOCK_SIZE) {
            int end = Math.min(start + BLOCK_SIZE, size);

            // Process each block
            for (int p = start; p < end; p++) {
                // Find pivot, skipping used rows
                int pivot = -1;
                double maxVal = 0;
                for (int i = 0; i < n; i++) {
                    if (used.test(i)) continue;
                    double absVal = Math.abs(k[i][pivots[p]]);
                    if (absVal > maxVal) {
                        maxVal = absVal;
                        pivot = i;
                    }
                }

                // Check for singularity
                if (maxVal < epsilon)
                    return 0.0;

                used.set(pivot);
                pivots[p] = pivot;

                // Update determinant
                det *= k[pivot][p];
                if (pivot != p) sign = -sign;

                // Cache pivot row for better memory access
                System.arraycopy(k[pivot], 0, tempRow, 0, n);

                // Update trailing submatrix
                for (int i = 0; i < n; i++) {
                    if (!used.test(i) && i != pivot) {
                        double alpha = k[i][p] / tempRow[p];
                        for (int j = p + 1; j < n; j++)
                            k[i][j] -= alpha * tempRow[j];
                    }
                }
            }
        }

        var value = sign * det;
        if (value!=value)
            throw new IllegalStateException();
        return value;
    }


    /**
     * Computes log-determinant with optional row/column exclusion.
     * Specifically designed for positive definite covariance matrices.
     */
    public static double logDeterminantWithout(double[][] k, int row) {
        if (k == null || k.length == 0 || k[0] == null) {
            throw new IllegalArgumentException("Invalid matrix");
        }
        int n = k.length;
        if (row >= n) throw new IllegalArgumentException("Row index out of bounds");

        // For very small matrices, use direct computation
        if (n <= 2) return logDeterminantSmall(k, row);

        return row == -1 ?
                logDeterminantCholesky(k)
                : logDeterminantCholeskyWithout(k, row);
    }

    public static double informationGain(double[][] k) {
        return informationGain(k, -1);
    }

    /**
     * Compute information gain using numerically stable log-determinants.
     * Avoids explicit computation of determinant ratios.
     * if withoutIdx==-1, none are removed
     */
    public static double informationGain(double[][] k, int withoutIdx) {
        if (withoutIdx < -1 || withoutIdx >= k.length) {
            throw new IllegalArgumentException("Invalid index");
        }

        double logDetFull = logDeterminantWithout(k, -1);
        double logDetReduced = logDeterminantWithout(k, withoutIdx);

        // Information gain is the difference of log-determinants
        var y = logDetFull - logDetReduced;
        if (y!=y)
            throw new IllegalStateException();
        return y;
    }

    /**
     * Compute log-determinant using Cholesky decomposition.
     * Handles positive definite matrices stably.
     */
    private static double logDeterminantCholesky(double[][] matrix) {
        int n = matrix.length;
        double[][] L = new double[n][n];

        double logDet = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = matrix[i][j];

                for (int k = 0; k < j; k++) {
                    sum -= L[i][k] * L[j][k];
                }

                if (i == j) {
                    if (sum <= 0.0) {
                        // Handle non-positive definite case
                        if (sum > -epsilon) sum = epsilon;
                        else throw new IllegalArgumentException("Matrix is not positive definite");
                    }
                    L[i][j] = Math.sqrt(sum);
                    logDet += Math.log(L[i][j]);
                } else {
                    L[i][j] = sum / L[j][j];
                }
            }
        }

        return 2.0 * logDet;  // log(det(A)) = 2*log(det(L))
    }

    /**
     * Compute log-determinant excluding specified row/column.
     * Uses stable Cholesky update/downdate.
     */
    private static double logDeterminantCholeskyWithout(double[][] matrix, int skipRow) {
        int n = matrix.length;
        int newSize = n - 1;
        double[][] reduced = new double[newSize][newSize];

        // Create reduced matrix excluding row/column
        int ri = 0;
        for (int i = 0; i < n; i++) {
            if (i == skipRow) continue;
            int rj = 0;
            for (int j = 0; j < n; j++) {
                if (j == skipRow) continue;
                reduced[ri][rj] = matrix[i][j];
                rj++;
            }
            ri++;
        }

        return logDeterminantCholesky(reduced);
    }

    /**
     * Direct computation for small matrices (n ≤ 2).
     * Uses specialized formulas for numerical stability.
     */
    private static double logDeterminantSmall(double[][] k, int row) {
        if (k.length == 1) {
            return row == 0 ? 0.0 : Math.log(Math.max(k[0][0], epsilon));
        }

        // 2x2 matrix
        if (row == -1) {
            double a = k[0][0], b = k[0][1], c = k[1][0], d = k[1][1];
            double det = a * d - b * c;
            if (det <= 0.0) {
                if (det > -epsilon) det = epsilon;
                else throw new IllegalArgumentException("Matrix is not positive definite");
            }
            return Math.log(det);
        }

        return row == 0 ? Math.log(Math.max(k[1][1], epsilon))
                : Math.log(Math.max(k[0][0], epsilon));
    }
}
