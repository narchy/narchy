package jcog.tensor.util;

import jdk.incubator.vector.DoubleVector;
import org.ejml.UtilEjml;
import org.ejml.data.DMatrix1Row;

import static jcog.Util.fma;
import static jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED;
import static jdk.incubator.vector.DoubleVector.fromArray;
import static jdk.incubator.vector.VectorOperators.ADD;

public enum MatUtil_CPU { ;

    // Optimized block sizes based on common L1/L2 cache sizes
    private static final int L1_BLOCK_SIZE = 32;  // Increased from 8 for better L1 cache utilization
    private static final int L2_BLOCK_SIZE = 256; // Added L2 blocking for hierarchical blocking
    private static final int VECTOR_SIZE = SPECIES_PREFERRED.length();
    private static final int UNROLL_FACTOR = 4;   // Manual loop unrolling factor

    // Threshold for switching to simple multiplication for small matrices
    private static final int SMALL_MATRIX_THRESHOLD = 32;


    /**
     * Optimized matrix multiplication with transposed A
     * assumes C is zero
     */
    public static void multTransA(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        UtilEjml.assertTrue(A != C && B != C, "Neither 'A' or 'B' can be the same matrix as 'C'");
        final int ar = A.numRows;
        UtilEjml.assertShape(ar, B.numRows, "The 'A' and 'B' matrices do not have compatible dimensions");
        final int ac = A.numCols, bc = B.numCols;

        // Early exit for empty matrices
        if (ac == 0 || ar == 0) {
            //CommonOps_DDRM.fill(C, 0);
            return;
        }

        // Use simple multiplication for small matrices
        if (ac * ar * bc < SMALL_MATRIX_THRESHOLD * SMALL_MATRIX_THRESHOLD * SMALL_MATRIX_THRESHOLD) {
            multTransASimple(A, B, C);
            return;
        }

        C.reshape(ac, bc);

        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;

        //CommonOps_DDRM.fill(C, 0); // Initialize C to zero for accumulation

        // Hierarchical blocking for L2 and L1 cache
        for (int i2 = 0; i2 < ac; i2 += L2_BLOCK_SIZE) {
            final int i2Max = Math.min(i2 + L2_BLOCK_SIZE, ac);
            for (int k2 = 0; k2 < ar; k2 += L2_BLOCK_SIZE) {
                final int k2Max = Math.min(k2 + L2_BLOCK_SIZE, ar);
                for (int j2 = 0; j2 < bc; j2 += L2_BLOCK_SIZE) {
                    final int j2Max = Math.min(j2 + L2_BLOCK_SIZE, bc);

                    // L1 cache blocking
                    for (int i = i2; i < i2Max; i += L1_BLOCK_SIZE) {
                        final int iMax = Math.min(i + L1_BLOCK_SIZE, i2Max);
                        for (int k = k2; k < k2Max; k += L1_BLOCK_SIZE) {
                            final int kMax = Math.min(k + L1_BLOCK_SIZE, k2Max);
                            for (int j = j2; j < j2Max; j += L1_BLOCK_SIZE) {
                                multTransABlockOptimized(aData, bData, cData, i, iMax, k, kMax, j, Math.min(j + L1_BLOCK_SIZE, j2Max), ac, bc);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void multTransABlockOptimized(double[] aData, double[] bData, double[] cData,
                                                 int iStart, int iMax, int kStart, int kMax,
                                                 int jStart, int jMax, int ac, int bc) {
        for (int i = iStart; i < iMax; i++) {
            final int indexC_start = i * bc;

            // Vectorized accumulation with manual loop unrolling
            for (int k = kStart; k < kMax; k++) {
                final double valA = aData[k * ac + i];
                final var vValA = DoubleVector.broadcast(SPECIES_PREFERRED, valA);

                int j = jStart;
                final int vectorLimit = jMax - VECTOR_SIZE * UNROLL_FACTOR;

                // Unrolled vectorized multiplication
                for (; j <= vectorLimit; j += VECTOR_SIZE * UNROLL_FACTOR) {
                    int idx = indexC_start + j;

                    var vB1 = fromArray(SPECIES_PREFERRED, bData, k * bc + j);
                    var vC1 = fromArray(SPECIES_PREFERRED, cData, idx);
                    vB1.fma(vValA, vC1).intoArray(cData, idx);

                    var vB2 = fromArray(SPECIES_PREFERRED, bData, k * bc + j + VECTOR_SIZE);
                    var vC2 = fromArray(SPECIES_PREFERRED, cData, idx + VECTOR_SIZE);
                    vB2.fma(vValA, vC2).intoArray(cData, idx + VECTOR_SIZE);

                    var vB3 = fromArray(SPECIES_PREFERRED, bData, k * bc + j + VECTOR_SIZE * 2);
                    var vC3 = fromArray(SPECIES_PREFERRED, cData, idx + VECTOR_SIZE * 2);
                    vB3.fma(vValA, vC3).intoArray(cData, idx + VECTOR_SIZE * 2);

                    var vB4 = fromArray(SPECIES_PREFERRED, bData, k * bc + j + VECTOR_SIZE * 3);
                    var vC4 = fromArray(SPECIES_PREFERRED, cData, idx + VECTOR_SIZE * 3);
                    vB4.fma(vValA, vC4).intoArray(cData, idx + VECTOR_SIZE * 3);
                }

                // Handle remaining vectors
                for (; j <= jMax - VECTOR_SIZE; j += VECTOR_SIZE) {
                    int idx = indexC_start + j;
                    var vB = fromArray(SPECIES_PREFERRED, bData, k * bc + j);
                    var vC = fromArray(SPECIES_PREFERRED, cData, idx);
                    vB.fma(vValA, vC).intoArray(cData, idx);
                }

                // Handle remaining scalars
                for (; j < jMax; j++)
                    cData[indexC_start + j] = fma(valA, bData[k * bc + j], cData[indexC_start + j]);
            }
        }
    }

    private static void multTransASimple(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        final int ar = A.numRows;
        final int ac = A.numCols;
        final int bc = B.numCols;
        C.reshape(ac, bc);

        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;

        for (int i = 0; i < ac; i++) {
            for (int j = 0; j < bc; j++) {
                double sum = 0;
                for (int k = 0; k < ar; k++)
                    sum = fma(aData[k * ac + i], bData[k * bc + j], sum);

                cData[i * bc + j] = sum;
            }
        }
    }

    /**
     * Optimized matrix multiplication with transposed B
     * assumes C is zero
     */
    public static void multTransB(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        UtilEjml.assertTrue(A != C && B != C, "Neither 'A' or 'B' can be the same matrix as 'C'");
        final int bc = B.numCols;
        UtilEjml.assertShape(A.numCols, bc, "The 'A' and 'B' matrices do not have compatible dimensions");

//        // Special case for row vector multiplication
//        if (B.numRows == 1) {
//            mult(A, B, C);
//            return;
//        }

        final int ar = A.numRows;
        final int br = B.numRows;

        // Early exit for empty matrices
        if (bc == 0 || ar == 0) {
            //CommonOps_DDRM.fill(C, 0);
            return;
        }

        // Use simple multiplication for small matrices
        if (ar * bc * br < SMALL_MATRIX_THRESHOLD * SMALL_MATRIX_THRESHOLD * SMALL_MATRIX_THRESHOLD) {
            multTransBSimple(A, B, C);
            return;
        }

        C.reshape(ar, br);

        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;

        // Initialize C to zero for accumulation
        //CommonOps_DDRM.fill(C, 0);

        // Hierarchical blocking for L2 and L1 cache
        for (int i2 = 0; i2 < ar; i2 += L2_BLOCK_SIZE) {
            final int i2Max = Math.min(i2 + L2_BLOCK_SIZE, ar);
            for (int j2 = 0; j2 < br; j2 += L2_BLOCK_SIZE) {
                final int j2Max = Math.min(j2 + L2_BLOCK_SIZE, br);
                for (int k2 = 0; k2 < bc; k2 += L2_BLOCK_SIZE) {
                    final int k2Max = Math.min(k2 + L2_BLOCK_SIZE, bc);

                    // L1 cache blocking
                    for (int i = i2; i < i2Max; i += L1_BLOCK_SIZE) {
                        final int iMax = Math.min(i + L1_BLOCK_SIZE, i2Max);
                        for (int j = j2; j < j2Max; j += L1_BLOCK_SIZE) {
                            final int jMax = Math.min(j + L1_BLOCK_SIZE, j2Max);
                            for (int k = k2; k < k2Max; k += L1_BLOCK_SIZE) {
                                final int kMax = Math.min(k + L1_BLOCK_SIZE, k2Max);
                                multTransBBlockOptimized(aData, bData, cData, i, iMax, j, jMax, k, kMax, bc, br);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void multTransBBlockOptimized(double[] aData, double[] bData, double[] cData,
                                                 int iStart, int iMax, int jStart, int jMax,
                                                 int kStart, int kMax, int bc, int br) {
        for (int i = iStart; i < iMax; i++) {
            final int cIndex_base = i * br;
            final int aIndex_base = i * bc;

            for (int j = jStart; j < jMax; j++) {
                final int bIndex_base = j * bc;
                double sum = 0.0;

                // Vectorized accumulation with manual unrolling
                int k = kStart;
                final int vectorLimit = kMax - VECTOR_SIZE * UNROLL_FACTOR;

                if (k <= vectorLimit) {
                    var vSum1 = DoubleVector.zero(SPECIES_PREFERRED);
                    var vSum2 = DoubleVector.zero(SPECIES_PREFERRED);
                    var vSum3 = DoubleVector.zero(SPECIES_PREFERRED);
                    var vSum4 = DoubleVector.zero(SPECIES_PREFERRED);

                    for (; k <= vectorLimit; k += VECTOR_SIZE * UNROLL_FACTOR) {
                        // Unroll 1
                        var vA1 = fromArray(SPECIES_PREFERRED, aData, aIndex_base + k);
                        var vB1 = fromArray(SPECIES_PREFERRED, bData, bIndex_base + k);
                        vSum1 = vA1.fma(vB1, vSum1);

                        // Unroll 2
                        var vA2 = fromArray(SPECIES_PREFERRED, aData, aIndex_base + k + VECTOR_SIZE);
                        var vB2 = fromArray(SPECIES_PREFERRED, bData, bIndex_base + k + VECTOR_SIZE);
                        vSum2 = vA2.fma(vB2, vSum2);

                        // Unroll 3
                        var vA3 = fromArray(SPECIES_PREFERRED, aData, aIndex_base + k + VECTOR_SIZE * 2);
                        var vB3 = fromArray(SPECIES_PREFERRED, bData, bIndex_base + k + VECTOR_SIZE * 2);
                        vSum3 = vA3.fma(vB3, vSum3);

                        // Unroll 4
                        var vA4 = fromArray(SPECIES_PREFERRED, aData, aIndex_base + k + VECTOR_SIZE * 3);
                        var vB4 = fromArray(SPECIES_PREFERRED, bData, bIndex_base + k + VECTOR_SIZE * 3);
                        vSum4 = vA4.fma(vB4, vSum4);
                    }

                    // Combine unrolled sums
                    sum = vSum1.reduceLanes(ADD) +
                            vSum2.reduceLanes(ADD) +
                            vSum3.reduceLanes(ADD) +
                            vSum4.reduceLanes(ADD);
                }

                // Handle remaining vectors
                final int remainingVectorLimit = kMax - VECTOR_SIZE;
                for (; k <= remainingVectorLimit; k += VECTOR_SIZE) {
                    var vA = fromArray(SPECIES_PREFERRED, aData, aIndex_base + k);
                    var vB = fromArray(SPECIES_PREFERRED, bData, bIndex_base + k);
                    sum += vA.mul(vB).reduceLanes(ADD);
                }

                // Handle remaining scalars
                for (; k < kMax; k++)
                    sum = fma(aData[aIndex_base + k], bData[bIndex_base + k], sum);

                cData[cIndex_base + j] = sum;
            }
        }
    }

    private static void multTransBSimple(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        final int ar = A.numRows;
        final int bc = B.numCols;
        final int br = B.numRows;
        C.reshape(ar, br);

        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;

        for (int i = 0; i < ar; i++) {
            for (int j = 0; j < br; j++) {
                //TODO KahanSum?
                var ibc = i * bc;
                var jbc = j * bc;
                double sum = 0;
                for (int k = 0; k < bc; k++)
                    sum = fma(aData[ibc + k], bData[jbc + k], sum);

                cData[i * br + j] = sum;
            }
        }
    }

    /**
     * Optimized standard matrix multiplication
     * assumes C is zero
     */
    public static void mult(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        final int aRows = A.numRows, aCols = A.numCols;
        final int bCols = B.numCols;

        UtilEjml.assertTrue(A != C && B != C, "Neither 'A' or 'B' can be the same matrix as 'C'");
        UtilEjml.assertShape(aCols, B.numRows, "The 'A' and 'B' matrices do not have compatible dimensions");

        // Early exit for empty matrices
        if (aCols == 0 || aRows == 0) {
            //CommonOps_DDRM.fill(C, 0);
            return;
        }
        C.reshape(aRows, bCols);

        // Use simple multiplication for small matrices
        if (aRows * aCols * bCols < SMALL_MATRIX_THRESHOLD * SMALL_MATRIX_THRESHOLD * SMALL_MATRIX_THRESHOLD)
            multSimple(A, B, C);
        else
            multBlock(A, B, C, aRows, aCols, bCols);
    }

    /** assumes C is zero */
    private static void multBlock(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C, int aRows, int aCols, int bCols) {
        final double[] aa = A.data;
        final double[] bb = B.data;
        final double[] cc = C.data;

        // Initialize C to zero for accumulation
        //CommonOps_DDRM.fill(C, 0);

        // Hierarchical blocking for L2 and L1 cache
        for (int i2 = 0; i2 < aRows; i2 += L2_BLOCK_SIZE) {
            final int i2Max = Math.min(i2 + L2_BLOCK_SIZE, aRows);
            for (int k2 = 0; k2 < aCols; k2 += L2_BLOCK_SIZE) {
                final int k2Max = Math.min(k2 + L2_BLOCK_SIZE, aCols);
                for (int j2 = 0; j2 < bCols; j2 += L2_BLOCK_SIZE) {
                    final int j2Max = Math.min(j2 + L2_BLOCK_SIZE, bCols);

                    // L1 cache blocking
                    for (int i = i2; i < i2Max; i += L1_BLOCK_SIZE) {
                        int iMax = Math.min(i + L1_BLOCK_SIZE, i2Max);
                        for (int k = k2; k < k2Max; k += L1_BLOCK_SIZE) {
                            int kMax = Math.min(k + L1_BLOCK_SIZE, k2Max);
                            for (int j = j2; j < j2Max; j += L1_BLOCK_SIZE) {
                                multBlockOptimized(aa, bb, cc, i, iMax, j,
                                        Math.min(j + L1_BLOCK_SIZE, j2Max), k, kMax, aCols, bCols);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void multBlockOptimized(double[] aa, double[] bb, double[] cc,
                                           int iStart, int iMax, int jStart, int jMax,
                                           int kStart, int kMax, int aCols, int bCols) {
        // Process each row of the block
        for (int i = iStart; i < iMax; i++) {
            final int cIndex_base = i * bCols;
            final int aIndex_base = i * aCols;

            // Process each column of the result block
            for (int j = jStart; j < jMax; j++) {
                // Accumulator registers for manual loop unrolling
                double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;

                // Process the depth dimension with manual unrolling
                int k = kStart;
                final int unrollLimit = kMax - UNROLL_FACTOR;

                // Main loop with manual unrolling
                for (; k <= unrollLimit; k += UNROLL_FACTOR) {
                    final int bIndex = k * bCols + j;
                    final int aIndex = aIndex_base + k;

                    sum1 = fma(aa[aIndex], bb[bIndex], sum1);
                    sum2 = fma(aa[aIndex + 1], bb[bIndex + bCols], sum2);
                    sum3 = fma(aa[aIndex + 2], bb[bIndex + bCols * 2], sum3);
                    sum4 = fma(aa[aIndex + 3], bb[bIndex + bCols * 3], sum4);
                }

                // Handle remaining elements
                for (; k < kMax; k++)
                    sum1 = fma(aa[aIndex_base + k], bb[k * bCols + j], sum1);

                // Combine all partial sums
                cc[cIndex_base + j] += sum1 + sum2 + sum3 + sum4;
            }
        }
    }

    private static void multSimple(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        final int aRows = A.numRows;
        final int aCols = A.numCols;
        final int bCols = B.numCols;

        final double[] aa = A.data;
        final double[] bb = B.data;
        final double[] cc = C.data;

        for (int i = 0; i < aRows; i++) {
            final int aRowOffset = i * aCols;
            final int cRowOffset = i * bCols;
            for (int j = 0; j < bCols; j++) {
                double sum = 0;
                for (int k = 0; k < aCols; k++)
                    sum = fma(aa[aRowOffset + k], bb[k * bCols + j], sum);

                cc[cRowOffset + j] = sum;
            }
        }
    }

    public static void addEle(double[] a, double[] b, double[] result) {
        int i = 0;
        var SPECIES = SPECIES_PREFERRED;
        int upperBound = SPECIES.loopBound(a.length);
        var l = SPECIES.length();
        for (; i < upperBound; i += l)
            fromArray(SPECIES, a, i).add(fromArray(SPECIES, b, i)).intoArray(result, i);
        for (; i < a.length; i++)
            result[i] = a[i] + b[i];
    }
}