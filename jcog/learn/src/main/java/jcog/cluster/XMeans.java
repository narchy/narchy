package jcog.cluster;

import jcog.TODO;
import jcog.data.DistanceFunction;
import jcog.data.list.Lst;
import jcog.random.XoRoShiRo128PlusRandom;

import java.util.List;
import java.util.random.RandomGenerator;

/** untested: recursively breaks down the centroids */
public abstract class XMeans<X> extends KMeansPlusPlus<X> {

    private final int maxK;
    private final double bic_ratio_threshold;

    protected XMeans(int minK, int maxK, int dims, DistanceFunction measure, RandomGenerator random, double bic_ratio_threshold) {
        super(minK, dims, measure, random);
        this.maxK = maxK;
        this.bic_ratio_threshold = bic_ratio_threshold;
    }

    @Override
    public void cluster(Lst<X> values, int maxIterations) {
        super.cluster(values, maxIterations);
        
        while (clusterCount() < maxK) {
            boolean improved = false;
            List<CentroidCluster<X>> newClusters = new Lst<>();

            for (CentroidCluster<X> cluster : clusters) {
                if (tryToSplitCluster(cluster, newClusters, maxIterations)) {
                    improved = true;
                } else {
                    newClusters.add(cluster);
                }
            }

            if (!improved) break;

            clusters.clear();
            clusters.addAll(newClusters);
        }
    }

    private boolean tryToSplitCluster(CentroidCluster<X> cluster, List<CentroidCluster<X>> newClusters, int maxIterations) {
        if (cluster.size() < 2) {
            newClusters.add(cluster);
            return false;
        }

        // Create a sub-KMeans++ for this cluster
        RandomGenerator random = new XoRoShiRo128PlusRandom();
        KMeansPlusPlus<X> subKMeans = new KMeansPlusPlus<>(2, dims, distance, random) {
            @Override
            public void coord(X x, double[] coords) {
                XMeans.this.coord(x, coords);
            }
        };

        Lst<X> subValues = cluster.valueList(this);
        subKMeans.cluster(subValues, maxIterations);

        double bicBefore = computeBIC(cluster);
        double bicAfter = computeBIC(subKMeans.clusters.get(0)) + computeBIC(subKMeans.clusters.get(1));

        if (bicAfter / bicBefore > bic_ratio_threshold) {
            newClusters.addAll(subKMeans.clusters);
            return true;
        } else {
            newClusters.add(cluster);
            return false;
        }
    }

    private double computeBIC(CentroidCluster<X> cluster) {
        int n = cluster.size();
        int d = dims;
        
        if (n <= 1) return 0;

        double logLikelihood = computeLogLikelihood(cluster);
        double numParameters = d * (clusterCount() + 1);  // d * (K + 1)

        return -2 * logLikelihood + numParameters * Math.log(n);
    }


    /** needs equivalent of this Standard Deviation class */
    private double computeLogLikelihood(CentroidCluster<X> cluster) {
        throw new TODO();
//        int n = cluster.size();
//        int d = dims;
//
//        if (n <= 1) return 0;
//
//        double[] variances = new double[d];
//        StandardDeviation stdDev = new StandardDeviation();
//
//        for (int i = 0; i < d; i++) {
//            final int dim = i;
//            double[] values = cluster.valueList(this).stream()
//                .mapToDouble(x -> {
//                    double[] coords = new double[dims];
//                    coord(x, coords);
//                    return coords[dim];
//                })
//                .toArray();
//            variances[i] = Math.pow(stdDev.evaluate(values), 2);
//        }
//
//        double logLikelihood = 0;
//        for (int i = 0; i < n; i++) {
//            double sum = 0;
//            for (int j = 0; j < d; j++) {
//                double diff = coord(cluster.valueList(this).get(i))[j] - cluster.center[j];
//                sum += FastMath.pow(diff, 2) / variances[j];
//            }
//            logLikelihood -= (d / 2.0) * FastMath.log(2 * Math.PI) +
//                             0.5 * sum +
//                             0.5 * FastMath.log(variances[0]); // Assuming all variances are equal
//        }
//
//        return logLikelihood;
    }
}