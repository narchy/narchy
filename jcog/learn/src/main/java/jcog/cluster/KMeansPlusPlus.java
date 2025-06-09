package jcog.cluster;

import jcog.Is;
import jcog.Util;
import jcog.data.DistanceFunction;
import jcog.data.bit.IntArrayNBitSet;
import jcog.data.bit.MetalBitSet;
import jcog.data.iterator.ArrayIterator;
import jcog.data.list.Lst;
import jcog.random.RandomBits;
import jcog.util.ArrayUtil;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.Random;
import java.util.function.ToDoubleFunction;
import java.util.random.RandomGenerator;

import static jcog.Str.n2;

/**
 * A K-means++ clustering algorithm implementation.
 *
 * The algorithm:
 * 1. Uses K-means++ initialization for initial centroids.
 * 2. Assigns each point to the nearest centroid.
 * 3. Recomputes centroids until convergence or maximum iterations reached.
 *
 * This implementation allows custom distance functions and an "empty cluster" strategy.
 */
@Is("Determining_the_number_of_clusters_in_a_data_set")
public abstract class KMeansPlusPlus<X> implements AutoCloseable {

    public final int clusterCountMax;
    public int clusterCount;

    protected final int dims;
    protected final DistanceFunction distance;


    public final Lst<CentroidCluster<X>> clusters = new Lst<>(CentroidCluster.EmptyCentroidClustersArray);
    private final Lst<CentroidCluster<X>> clustersNext = new Lst<>(CentroidCluster.EmptyCentroidClustersArray);

    public transient Lst<X> values;
    private transient double[][] coord = ArrayUtil.EMPTY_DOUBLE_DOUBLE;
    private transient int[] assignments = ArrayUtil.EMPTY_INT_ARRAY;
    private transient double[] minDist;

    /** Random generator for center initialization. */
    public RandomGenerator random;

    /**
     * Constructor.
     *
     * @param k the number of clusters
     * @param dims the dimension of each data point
     * @param measure the distance function
     * @param random random generator for initialization
     */
    protected KMeansPlusPlus(int k, int dims, DistanceFunction measure, RandomGenerator random) {
        if (k < 2)
            throw new UnsupportedOperationException("Number of clusters must be greater than 1");

        this.distance = measure;
        this.clusterCountMax = k;
        this.dims = dims;
        this.random = (random instanceof Random r) ? new RandomBits(r) : random;
    }

    /**
     * Clusters the given list of points.
     *
     * @param _values points to cluster
     * @param maxIterations maximum number of iterations
     */
    public void cluster(Lst<X> _values, int maxIterations)  {
//        this.values = new Lst<>(_values.size());
//        for (var v : _values) {
//            if (v != null) this.values.add(v);
//        }
        this.values = _values.clone();

        var n = this.values.size();
        if (n <= 0) return;

        realloc(n);
        initCenters();

        var clustersArr = clusters.array();
        var clustersNextArr = clustersNext.array();

        assign(clustersArr, assignments);

        for (var iter = 0; iter < maxIterations; iter++) {
            if (!cluster(clustersArr, clustersNextArr))
                break; // Converged, break early
        }
    }

    /** @return true if changed, false if converged */
    private boolean cluster(CentroidCluster<X>[] clustersArr, CentroidCluster<X>[] clustersNextArr) {
        if (updateClusters(clustersArr, clustersNextArr)) {
            // Swap clusters with clustersNext
            for (var i = 0; i < clusterCount; i++)
                clustersArr[i].set(clustersNextArr[i]);
            return true;
        }
        return false;
    }

    @Override
    public String toString() {
        if (coord.length == 0) return "empty";
        var sb = new StringBuilder(256);
        for (var c = 0; c < this.clusterCount; c++) {
            sb.append(c)
              .append('<').append(n2(this.coordCluster(c)))
              .append('>')
              .append('=')
              .append(this.values(c)).append(' ');
        }
        if (sb.length() > 0) sb.setLength(sb.length() - 1);
        return sb.toString();
    }

    /**
     * Updates cluster centroids and reassigns points.
     * Returns true if clusters changed, false if converged.
     */
    private boolean updateClusters(CentroidCluster<X>[] current, CentroidCluster<X>[] next) {
        var emptyStrat = emptyStrategy();

        // Compute new cluster centers
        for (var i = 0; i < clusterCount; i++) {
            var clusterValues = current[i].values;

            var cNext = next[i];
            cNext.clearValues();
            var c = cNext.center;
            if (clusterValues.isEmpty()) {
                // Empty cluster: re-initialize center according to chosen strategy
                coordCopy(emptyStrat.get(this), c);
            } else {
                computeCentroid(clusterValues, c);
            }
        }

        // Reassign points to updated clusters
        return assign(next, assignments) > 0;
    }

    /**
     * Determines the empty cluster strategy. Here simplified as always FARTHEST_POINT.
     */
    private EmptyClusterStrategy emptyStrategy() {
        //return EmptyClusterStrategy.FARTHEST_POINT;
        return EmptyClusterStrategy.MOST_POINTS;
        //return EmptyClusterStrategy.LARGEST_VARIANCE;
    }

    /**
     * Ensures internal arrays are large enough and (re)initialized.
     *
     * @param pointCount number of points
     */
    public void realloc(int pointCount) {
        if (pointCount <= 0)
            throw new UnsupportedOperationException("Number of points must be positive.");

        // Only reallocate if needed
        if (assignments.length < pointCount) {
            assignments = new int[pointCount];
            coord = new double[pointCount][dims];
            minDist = new double[pointCount];
        } else {
            Arrays.fill(minDist, 0, pointCount, 0);
        }

        for (var i = 0; i < pointCount; i++)
            Arrays.fill(coord[i], Double.NaN);
        Arrays.fill(assignments, 0, assignments.length, -1);

        clusterCount = Util.clamp(pointCount / 2 /* at least 2 per centroid */, 1, clusterCountMax);

        // Prepare cluster lists
        if (clusters.size() != clusterCount) {
            clusters.clear();
            clustersNext.clear();
            for (var i = 0; i < clusterCount; i++) {
                clusters.add(new CentroidCluster<>(dims, pointCount));
                clustersNext.add(new CentroidCluster<>(dims, pointCount));
            }
        } else {
            // Just clear values if already allocated
            for (var c : clusters) c.clear(pointCount);
            for (var c : clustersNext) c.clear(pointCount);
        }
    }

    /**
     * Sort clusters by their centroid coordinates.
     */
    public KMeansPlusPlus<X> sortClusters() {
        clusters.sort((X, Y) -> Arrays.compare(X.center, Y.center));
        return this;
    }

    /**
     * Assigns each point to the nearest cluster.
     *
     * @param clusters cluster array
     * @param assignments current assignment array
     * @return number of points that changed their cluster
     */
    private int assign(CentroidCluster<X>[] clusters, int[] assignments) {
        // Clear cluster membership before re-assignment
        for (var c : clusters) {
            if (c!=null)
                c.clearValues();
        }

        var changes = 0;
        final var P = values.size();

        for (var p = 0; p < P; p++) {
            var c = nearest(coord(p), clusters);
//            if (c == -1)
//                throw new UnsupportedOperationException(); //continue; //??
            if (c != assignments[p]) {
                assignments[p] = c;
                changes++;
            }
            clusters[c].add(p);
        }
        return changes;
    }

    public int nearestByEquality(X p) {
        var known = indexOf(p);
        return known >= 0 ? nearest(known, clusters.array()) : -1;
    }

    public int nearest(X p) {
        return nearest(coords(p, new double[this.dims]), clusters.array());
    }

    private int nearest(int p, CentroidCluster<X>[] clusters) {
        return nearest(coord(p), clusters);
    }
    /**
     * Find the nearest cluster for a given point index (already known coordinates).
     */
    public int nearest(double[] p, CentroidCluster<X>[] cc) {
        var minDist = Double.POSITIVE_INFINITY;
        var minCluster = -1;
        for (var i = 0; i < cc.length; i++) {
            var ci = cc[i];
            if (ci != null) {
                var d = distance.distance(p, ci.center);
                if (d < minDist) {
                    minDist = d;
                    minCluster = i;
                } else {
                    if (d == minDist && ci.size() < cc[minCluster].size())
                        minCluster = i; // Tie-break: choose cluster with fewer points
                }
            }
        }
        return minCluster;
    }

    /**
     * Computes initial cluster centers using K-means++ initialization.
     */
    private void initCenters() {
        final var n = values.size();
        var clusterCount = Math.min(n, this.clusterCount);

        var taken = MetalBitSet.bits(n);

        // Choose first center at random
        var firstIndex = random.nextInt(n);
        var firstCenter = coord(firstIndex);
        assignCenter(taken, firstIndex, firstCenter, 0);

        Arrays.fill(minDist, Double.POSITIVE_INFINITY);

        // Initialize minDist array with distances to the first center
        for (var i = 0; i < n; i++)
            if (i != firstIndex) updateMinDist(i, firstCenter);


        var assigned = 1;
        while (assigned < clusterCount) {
            var newCenterIdx = nextCenterDistProb(n, taken);
            //int newCenterIdx = nextCenterRandom(n, taken);
//            if (newCenterIdx < 0)
//                throw new UnsupportedOperationException(); //break;


            var newC = coord(newCenterIdx);
            assignCenter(taken, newCenterIdx, newC, assigned);
            assigned++;

            if (assigned < clusterCount) {
                // Update minDist
                for (var j = 0; j < n; j++)
                    if (!taken.test(j)) updateMinDist(j, newC);
            }
        }
    }

    private void updateMinDist(int i, double[] x) {
        double y = distance.distance(coord(i), x);
        minDist[i] = Math.min(minDist[i], y);
    }

    private int nextCenterRandom(int n, MetalBitSet taken) {
        int idx = random.nextInt(n);
        int remain = n;
        while (taken.test(idx)) {
            if (--remain <= 0)
                throw new UnsupportedOperationException();
            idx = (idx + 1) % n;
        }
        return idx;
    }

    private @Nullable int nextCenterDistProb(int n, MetalBitSet taken) {
        double distSum = 0;
        for (var i = 0; i < n; i++) {
            if (!taken.test(i))
                distSum += minDist[i];
        }
        if (distSum == 0) return nextCenterRandom(n, taken);

        // Roulette-select a new center with probability proportional to its minDist
        var r = random.nextDouble() * distSum;
        var newCenterIdx = -1;
        var lastAvailable = -1;

        double sum = 0;
        for (var i = 0; i < n; i++) {
            if (!taken.test(i)) {
                lastAvailable = i;
                sum += minDist[i];
                if (sum >= r)
                    return i;
            }
        }

        // In case of floating point issues, pick the last available point
        return lastAvailable-1;
    }

    private void assignCenter(MetalBitSet taken, int x, double[] xCoord, int centroid) {
        clusters.get(centroid).setCenter(xCoord);
        taken.set(x);
    }

    public final double[] coord(int x) {
        var c = coord[x];
        if (!computed(c))
            coords(values.get(x), c);
        return c;
    }

    public void coordCopy(int x, double[] target) {
        var c = coord(x);
        System.arraycopy(c, 0, target, 0, c.length);
    }

    private static boolean computed(double[] c) {
        var firstCoord = c[0];
        return firstCoord==firstCoord; /* not NaN */
    }

    public final double[] coords(X x, double[] y) {
        coord(x, y);
        return y;
    }

    /**
     * Implement this method to extract coordinates from your data point X into the array coords.
     */
    public abstract void coord(X x, double[] coords);

    /**
     * Compute the mean centroid for a set of points.
     */
    private void computeCentroid(IntArrayNBitSet points, double[] c) {
        Arrays.fill(c, 0);
        var count = 0;
        var it = points.iterator();
        while (it.hasNext()) {
            Util.addTo(c, coord(it.next()));
            count++;
        }
        if (count > 1) {
            for (var i = 0; i < dims; i++)
                c[i] /= count;
        }
    }

    /**
     * Clears all data structures.
     */
    public void clear() {
        if (values != null) values.clear();
        clusters.forEach(CentroidCluster::clear);
        coord = ArrayUtil.EMPTY_DOUBLE_DOUBLE;
        assignments = ArrayUtil.EMPTY_INT_ARRAY;
        minDist = ArrayUtil.EMPTY_DOUBLE_ARRAY;
    }

    public int cluster(X instance) {
        return cluster(indexOf(instance));
    }

    public int cluster(int instance) {
        if (instance < 0) return -1;
        var k = clusters.size();
        for (var i = 0; i < k; i++) {
            if (clusters.get(i).values.test(instance)) return i;
        }
        return -1;
    }

    private int indexOf(X p) {
        return (values != null) ? values.indexOf(p) : -1;
    }

    /**
     * Sort clusters by their average distance-to-center (variance).
     */
    public void sortClustersByVariance() {
        clusters.sortThisByFloat(c -> (float) c.meanDistanceToCenter(this), true);
    }

    public final Lst<X> values(int c) {
        return clusters.get(c).valueList(this);
    }

    public final double valueSum(int c, ToDoubleFunction<X> each) {
        return clusters.get(c).valueSum(each, this);
    }

    public double[] coordCluster(int c) {
        return clusters.get(c).center;
    }

    public final int valueCount(int c) {
        return clusters.get(c).size();
    }

    public Iterator<X> valueIterator(int cluster) {
        final var ii = clusterValues(cluster).iterator();
        final var kv = values.array();
        return new Iterator<>() {
            @Override public boolean hasNext() { return ii.hasNext(); }
            @Override public X next() { return kv[ii.next()]; }
        };
    }

    public Iterator<X> valueIteratorShuffled(int cluster, RandomGenerator rng) {
        var vv = clusterValues(cluster);
        var kv = values.array();
        var count = vv.cardinality();
        var subset = Arrays.copyOf(kv, count);
        var idxIter = vv.iterator();
        var idx = 0;
        while (idxIter.hasNext()) {
            subset[idx++] = kv[idxIter.next()];
        }
        ArrayUtil.shuffle(subset, rng);
        return ArrayIterator.iterate(subset);
    }

    private IntArrayNBitSet clusterValues(int cluster) {
        return clusters.get(cluster).values;
    }

    public void sortClustersRandom() {
        clusters.shuffleThis(random);
    }

    @Override
    public void close() {
        clusters.delete();
        clear();
    }

    public float clusterCount() {
        return clusters.size();
    }

    /**
     * Strategies to handle empty clusters.
     */
    public enum EmptyClusterStrategy {

        LARGEST_VARIANCE() {
            @Override
            public <X> int get(KMeansPlusPlus<X> k) {
                var maxVariance = Double.NEGATIVE_INFINITY;
                CentroidCluster<X> selected = null;
                for (var cluster : k.clusters) {
                    if (cluster.size() > 0) {
                        var variance = cluster.meanDistanceToCenter(k);
                        if (variance > maxVariance) {
                            maxVariance = variance;
                            selected = cluster;
                        }
                   }
                }
                //if (selected == null)
                    //throw new RuntimeException("No non-empty cluster found for LARGEST_VARIANCE strategy.");
                // Extract a random point
                return selected.clusterPop(k.random);
            }
        },

        MOST_POINTS() {
            @Override
            public <X> int get(KMeansPlusPlus<X> k) {
                var max = -1;
                CentroidCluster<X> biggest = null;
                var kc = k.clusters;
                var n = kc.size();
                // Random offset for fairness
                var rng = k.random;
                var start = rng.nextInt(n);
                for (var i = 0; i < n; i++) {
                    var cluster = kc.get((start + i) % n);
                    var cs = cluster.size();
                    if (cs > max) {
                        max = cs;
                        biggest = cluster;
                    }
                }
                return biggest.clusterPop(rng);
                //return (biggest != null) ? biggest.clusterPop(k.random) : -1;
            }
        },

        FARTHEST_POINT() {
            @Override
            public <X> int get(KMeansPlusPlus<X> k) {
                return new FarthestPointSelector<>(k).select();
            }

            /**
             * Helper class to select a farthest point from its cluster centroid.
             */
            class FarthestPointSelector<X> {
                private final KMeansPlusPlus<X> k;
                private double maxDistance = Double.NEGATIVE_INFINITY;
                private CentroidCluster<X> selectedCluster;
                private int selectedPoint = -1;

                FarthestPointSelector(KMeansPlusPlus<X> k) {
                    this.k = k;
                }

                int select() {
                    for (var c : k.clusters) {
                        if (c.size() == 0) continue;
                        apply(c);
                    }
                    if (selectedCluster != null) {
                        selectedCluster.remove(selectedPoint);
                        return selectedPoint;
                    } else {
                        throw new UnsupportedOperationException();
                    }
                }

                private void apply(CentroidCluster<X> c) {
                    var center = c.center;
                    var it = c.values.iterator();
                    while (it.hasNext()) {
                        var idx = it.next();
                        var d = k.distance.distance(k.coord(idx), center);
                        if (d > maxDistance) {
                            maxDistance = d;
                            selectedCluster = c;
                            selectedPoint = idx;
                        }
                    }
                }
            }
        };

        public abstract <X> int get(KMeansPlusPlus<X> c);
    }

    /**
     * Represents a cluster with a centroid and the indices of points that belong to it.
     */
    public static final class CentroidCluster<X> {
        private static final CentroidCluster[] EmptyCentroidClustersArray = new CentroidCluster[0];
        final IntArrayNBitSet values;
        public final double[] center;

        CentroidCluster(int dims, int items) {
            center = new double[dims];
            Arrays.fill(center, Double.NaN);
            values = new IntArrayNBitSet(items);
        }

        @Override public String toString() {
            return n2(center) + "=" + values.toString();
        }
        void add(int i) {
            values.set(i);
        }

        void remove(int i) {
            values.clear(i);
        }

        public void values(KMeansPlusPlus<X> k, Collection<X> result) {
            var it = values.iterator();
            var arr = k.values.array();
            while (it.hasNext()) {
                result.add(arr[it.next()]);
            }
        }

        Lst<X> valueList(KMeansPlusPlus<X> k) {
            var l = new Lst<X>(size());
            values(k, l);
            return l;
        }

        double valueSum(ToDoubleFunction<X> func, KMeansPlusPlus<X> k) {
            double sum = 0;
            var arr = k.values.array();
            var it = values.iterator();
            while (it.hasNext()) {
                sum += func.applyAsDouble(arr[it.next()]);
            }
            return sum;
        }

        double meanDistanceToCenter(KMeansPlusPlus<?> k) {
            var it = values.iterator();
            double distSum = 0;
            var count = 0;
            while (it.hasNext()) {
                distSum += k.distance.distance(k.coord(it.next()), center);
                count++;
            }
            return count == 0 ? Double.POSITIVE_INFINITY : distSum / count;
        }

        int size() {
            return values.cardinality();
        }

        void set(CentroidCluster<X> c) {
            System.arraycopy(c.center, 0, center, 0, center.length);
            values.set(c.values);
        }

        void setCenter(double[] c) {
            System.arraycopy(c, 0, center, 0, c.length);
        }

        void clear(int capacity) {
            clear();
            values.resize(capacity);
        }

        void clear() {
            values.clear();
            Arrays.fill(center, Double.NaN);
        }

        void clearValues() {
            values.clear();
        }

        /**
         * Extract a random point from the cluster and remove it.
         */
        public int clusterPop(RandomGenerator random) {
            var vn = values.cardinality();
            if (vn == 0)
                return -1; // empty, nothing to pop
            var which = nthValue(random.nextInt(vn));
            values.clear(which);
            return which;
        }

        private int nthValue(int i) {
            var v = values.iterator();
            var nth = 0;
            while (v.hasNext()) {
                var w = v.next();
                if (nth == i) return w;
                nth++;
            }
            return -1;
        }
    }

}
