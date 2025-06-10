/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This is not the original file distributed by the Apache Software Foundation
 * It has been modified by the Hipparchus project
 */
package jcog.tensor.rl.pg.util;

import org.hipparchus.distribution.multivariate.AbstractMultivariateRealDistribution;
import org.hipparchus.exception.LocalizedCoreFormats;
import org.hipparchus.exception.MathIllegalArgumentException;
import org.hipparchus.linear.Array2DRowRealMatrix;
import org.hipparchus.linear.EigenDecompositionSymmetric;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.random.RandomGenerator;
import org.hipparchus.random.Well19937c;
import org.hipparchus.util.Precision;

import static jcog.util.ArrayUtil.copy;

/**
 * Implementation of the multivariate normal (Gaussian) distribution.
 *
 * @see <a href="http://en.wikipedia.org/wiki/Multivariate_normal_distribution">
 * Multivariate normal distribution (Wikipedia)</a>
 * @see <a href="http://mathworld.wolfram.com/MultivariateNormalDistribution.html">
 * Multivariate normal distribution (MathWorld)</a>
 */
public class MyMultivariateNormalDistribution extends AbstractMultivariateRealDistribution {
    /**
     * Default singular matrix tolerance check value
     **/
    private static final double DEFAULT_TOLERANCE = Precision.EPSILON;

    /**
     * Vector of means.
     */
    private final double[] means;
    /**
     * Covariance matrix.
     */
    private RealMatrix covarianceMatrix;
    /**
     * The matrix inverse of the covariance matrix.
     */
    private RealMatrix covarianceMatrixInverse;
    /**
     * The determinant of the covariance matrix.
     */
    private double covarianceMatrixDeterminant;
    /**
     * Matrix used in computation of samples.
     */
    private RealMatrix samplingMatrix;
    /**
     * Inverse singular check tolerance when testing if invertable
     **/
    private final double singularMatrixCheckTolerance;

    public MyMultivariateNormalDistribution(final double[] means, final double[][] covariances) throws MathIllegalArgumentException {
        this(means, covariances, DEFAULT_TOLERANCE);
    }

    public MyMultivariateNormalDistribution(final double[] means, final double[][] covariances, final double singularMatrixCheckTolerance) throws MathIllegalArgumentException {
        this(new Well19937c(), means, covariances, singularMatrixCheckTolerance);
    }

    public MyMultivariateNormalDistribution(RandomGenerator rng, final double[] means, final double[][] covariances) {
        this(rng, means, covariances, DEFAULT_TOLERANCE);
    }

    public MyMultivariateNormalDistribution(RandomGenerator rng, final double[] means, final double[][] covariances, final double singularMatrixCheckTolerance) throws MathIllegalArgumentException {
        super(rng, means.length);

        var dim = means.length;

        if (covariances.length != dim)
            throw new MathIllegalArgumentException(LocalizedCoreFormats.DIMENSIONS_MISMATCH, covariances.length, dim);

        for (int i = 0; i < dim; i++)
            if (dim != covariances[i].length)
                throw new MathIllegalArgumentException(LocalizedCoreFormats.DIMENSIONS_MISMATCH, covariances[i].length, dim);

        this.means = means;
        this.singularMatrixCheckTolerance = singularMatrixCheckTolerance;

        setCovariance(covariances);
    }

    public void setCovariance(double[][] covariances) {
        final RealMatrix samplingMatrix;
        covarianceMatrix = new Array2DRowRealMatrix(covariances);

        // Covariance matrix eigen decomposition.
        var covMatDec = new EigenDecompositionSymmetric(covarianceMatrix, singularMatrixCheckTolerance, true);

        // Compute and store the inverse.
        covarianceMatrixInverse = covMatDec.getSolver().getInverse();
        // Compute and store the determinant.
        covarianceMatrixDeterminant = covMatDec.getDeterminant();

        // Eigenvalues of the covariance matrix.
        var covMatEigenvalues = covMatDec.getEigenvalues();

        for (var covMatEigenvalue : covMatEigenvalues)
            if (covMatEigenvalue < 0)
                throw new MathIllegalArgumentException(LocalizedCoreFormats.NOT_POSITIVE_DEFINITE_MATRIX);

        int dim = getDimension();

        // Matrix where each column is an eigenvector of the covariance matrix.
        var covMatEigenvectors = new Array2DRowRealMatrix(dim, dim);
        for (int v = 0; v < dim; v++) covMatEigenvectors.setColumn(v, covMatDec.getEigenvector(v).toArray());

        var tmpMatrix = covMatEigenvectors.transpose();

        // Scale each eigenvector by the square root of its eigenvalue.
        for (int row = 0; row < dim; row++) {
            var factor = Math.sqrt(covMatEigenvalues[row]);
            for (int col = 0; col < dim; col++) tmpMatrix.multiplyEntry(row, col, factor);
        }

        samplingMatrix = covMatEigenvectors.multiply(tmpMatrix);
        this.samplingMatrix = samplingMatrix;
    }

    @Override
    public double density(final double[] vals) throws MathIllegalArgumentException {
        var dim = getDimension();
        if (vals.length != dim) throw new MathIllegalArgumentException(LocalizedCoreFormats.DIMENSIONS_MISMATCH, vals.length, dim);

        return Math.pow(2 * Math.PI, -0.5 * dim) * Math.pow(covarianceMatrixDeterminant, -0.5) * exponentTerm(vals);
    }

    public double[] stddev() {
        var dim = getDimension();
        var std = new double[dim];
        var s = covarianceMatrix.getData();
        for (int i = 0; i < dim; i++) std[i] = Math.sqrt(s[i][i]);
        return std;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] sample() {
        final int dim = getDimension();
        final double[] normalVals = new double[dim];

        for (int i = 0; i < dim; i++) normalVals[i] = random.nextGaussian();

        final double[] vals = samplingMatrix.operate(normalVals);

        for (int i = 0; i < dim; i++) vals[i] += means[i];

        return vals;
    }

    private double exponentTerm(final double[] values) {
        var centered = new double[values.length];
        for (int i = 0; i < centered.length; i++) centered[i] = values[i] - means[i];
        var p = covarianceMatrixInverse.preMultiply(centered);
        double sum = 0;
        for (int i = 0; i < p.length; i++) sum += p[i] * centered[i];
        return Math.exp(-0.5 * sum);
    }

    public void setMeans(double[] mean) {
        copy(mean, this.means);
    }

    public void setCovarianceMatrix(double[][] covarianceMatrix) {

    }
}
