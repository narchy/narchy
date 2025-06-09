package jcog.cluster;

import jcog.data.DistanceFunction;
import jcog.data.bit.IntArrayNBitSet;
import jcog.data.list.Lst;
import jcog.util.ArrayUtil;

import java.util.*;

/**
 * An implementation of the Torque Clustering algorithm, refactored to match the coding style
 * and library usage of KMeansPlusPlus.java.
 *
 * The algorithm:
 * 1. Initializes each data point as a cluster.
 * 2. Builds a dendrogram by iteratively merging nearest clusters based on mass and distance.
 * 3. Identifies abnormal connections using torque gap analysis.
 * 4. Reconstructs final clusters from non-abnormal connections.
 */
abstract public class TorqueClustering<X> {

    /** Input data points as a list of type X */
    private final Lst<X> values;

    /** Number of dimensions per data point */
    private final int dims;

    /** Distance function for measuring distances between points */
    protected final DistanceFunction distance;

    /** Coordinates of all data points, precomputed for efficiency */
    private double[][] coord;

    /** Current list of clusters during processing */
    private Lst<Cluster> clusters;

    /** All merge connections forming the dendrogram */
    private Lst<Connection> allConnections;

    /** Connections identified as abnormal (cuts) */
    private Set<Connection> abnormalConnections;

    /** Final clustering result: map from root to list of point indices
     * TODO IntObjectHashMap<IntArrayList> */
    private Map<Integer, List<Integer>> finalClusters;

    /**
     * Constructor.
     *
     * @param values List of data points to cluster
     * @param dims Dimension of each data point
     * @param distance Distance function for measuring distances
     */
    public TorqueClustering(Lst<X> values, int dims, DistanceFunction distance) {
        if (values.isEmpty())
            throw new IllegalArgumentException("Input data cannot be empty");
        this.values = values.clone();
        this.dims = dims;
        this.distance = distance;
        realloc(values.size());
    }

    /**
     * Abstract method to extract coordinates from a data point.
     * Must be implemented by subclasses to convert X to double[].
     */
    public abstract void coord(X x, double[] coords);

    /**
     * Clusters the data points using the Torque Clustering algorithm.
     */
    public void cluster() {
        initializeClusters();
        buildDendrogram();
        identifyAbnormalConnections();
        reconstructFinalClusters();
    }

    /**
     * Preallocates and initializes internal arrays.
     *
     * @param n Number of data points
     */
    private void realloc(int n) {
        this.coord = new double[n][dims];
        for (var i = 0; i < n; i++) {
            Arrays.fill(coord[i], Double.NaN);
            coord(values.get(i), coord[i]);
        }
    }

    /**
     * Initializes each data point as a separate cluster.
     */
    private void initializeClusters() {
        clusters = new Lst<>();
        var n = values.size();
        for (var i = 0; i < n; i++) {
            clusters.add(new Cluster(i, coord[i], i, n));
        }
    }

    /**
     * Builds the dendrogram by iteratively merging clusters.
     */
    private void buildDendrogram() {
        allConnections = new Lst<>();
        while (clusters.size() > 1) {
            var currentConnections = new Lst<Connection>();
            for (var c : clusters) {
                var neighbor = findNearestCluster(c);
                if (neighbor != null) {
                    var con = new Connection(c, neighbor, distance);
                    currentConnections.add(con);
                    allConnections.add(con);
                }
            }
            clusters = mergeClusters(currentConnections);
        }
    }

    /**
     * Finds the nearest cluster with greater mass, or equal mass with a higher ID.
     *
     * @param cluster The cluster to find a neighbor for
     * @return The nearest eligible neighbor, or null if none exists
     */
    private Cluster findNearestCluster(Cluster cluster) {
        Cluster bestNeighbor = null;
        var bestDistance = Double.POSITIVE_INFINITY;
        for (var candidate : clusters) {
            if (candidate.id == cluster.id) continue;
            if (candidate.mass > cluster.mass ||
                    (candidate.mass == cluster.mass && candidate.id > cluster.id)) {
                var dist = distance.distance(cluster.centroid, candidate.centroid);
                if (dist < bestDistance) {
                    bestDistance = dist;
                    bestNeighbor = candidate;
                }
            }
        }
        return bestNeighbor;
    }

    /**
     * Merges clusters based on the current connections.
     *
     * @param connections List of connections to process
     * @return Updated list of clusters
     */
    private Lst<Cluster> mergeClusters(Lst<Connection> connections) {
        var clusterMap = new HashMap<Integer, Cluster>();
        for (var cluster : clusters) {
            clusterMap.put(cluster.id, cluster);
        }

        var uf = new UnionFind(clusters.size());
        for (var con : connections) {
            uf.union(con.cluster1.id, con.cluster2.id);
        }

        var compMap = new HashMap<Integer, Lst<Cluster>>();
        for (var c : clusters) {
            compMap.computeIfAbsent(uf.find(c.id), k -> new Lst<>()).add(c);
        }

        var newClusters = new Lst<Cluster>();
        var newId = 0;
        for (var group : compMap.values()) {
            var merged = group.get(0);
            var newCluster = new Cluster(newId++, merged.centroid, firstIndex(merged.indices), values.size());
            newCluster.mass = 0;
            newCluster.indices.clear();
            for (var c : group) {
                newCluster.mass += c.mass;
                newCluster.indices.set(c.indices);
            }
            var newCentroid = new double[dims];
            var it = newCluster.indices.iterator();
            while (it.hasNext()) {
                var idx = it.next();
                for (var i = 0; i < dims; i++)
                    newCentroid[i] += coord[idx][i];
            }
            for (var i = 0; i < dims; i++) {
                newCentroid[i] /= newCluster.mass;
            }
            newCluster.centroid = newCentroid;
            newClusters.add(newCluster);
        }
        return newClusters;
    }

    /**
     * Identifies abnormal connections using torque gap analysis.
     */
    private void identifyAbnormalConnections() {
        var sorted = allConnections.clone();
        sorted.sort((a, b) -> Double.compare(b.tau, a.tau));
        var maxGap = 0.0;
        var cutIndex = 0;
        for (var i = 0; i < sorted.size() - 1; i++) {
            var gap = sorted.get(i).tau - sorted.get(i + 1).tau;
            if (gap > maxGap) {
                maxGap = gap;
                cutIndex = i;
            }
        }
        abnormalConnections = new HashSet<>(cutIndex + 1);
        for (var i = 0; i <= cutIndex; i++) {
            abnormalConnections.add(sorted.get(i));
        }
    }

    /**
     * Reconstructs final clusters from non-abnormal connections.
     */
    private void reconstructFinalClusters() {
        var n = values.size();
        var uf = new UnionFind(n);
        for (var con : allConnections) {
            if (!abnormalConnections.contains(con)) {
                uf.union(firstIndex(con.cluster1.indices), firstIndex(con.cluster2.indices));
            }
        }
        finalClusters = new HashMap<>();
        for (var i = 0; i < n; i++) {
            var root = uf.find(i);
            finalClusters.computeIfAbsent(root, k -> new Lst<>()).add(i);
        }
    }

    private int firstIndex(IntArrayNBitSet s) {
        return s.next(true, 0, s.capacity());
    }

    /**
     * Returns the final clustering result.
     *
     * @return Map from cluster root to list of point indices
     */
    public Map<Integer, List<Integer>> getFinalClusters() {
        return finalClusters;
    }

    /**
     * Clears all internal data structures.
     */
    public void clear() {
        if (clusters != null) clusters.clear();
        if (allConnections != null) allConnections.clear();
        if (abnormalConnections != null) abnormalConnections.clear();
        if (finalClusters != null) finalClusters.clear();
        coord = ArrayUtil.EMPTY_DOUBLE_DOUBLE;
    }

    /** Represents a cluster with a centroid and indices of points */
    public static class Cluster {
        public int id;
        public double mass;
        public double[] centroid;
        public final IntArrayNBitSet indices;

        public Cluster(int id, double[] point, int index, int capacity) {
            this.id = id;
            this.mass = 1.0;
            this.centroid = point.clone();
            this.indices = new IntArrayNBitSet(capacity);
            this.indices.set(index);
        }

        public void mergeWith(Cluster other) {
            var totalMass = this.mass + other.mass;
            int dims = centroid.length;
            var newCentroid = new double[dims];
            for (var i = 0; i < dims; i++) {
                newCentroid[i] = (this.centroid[i] * this.mass + other.centroid[i] * other.mass) / totalMass;
            }
            this.mass = totalMass;
            this.centroid = newCentroid;
            this.indices.set(other.indices);
        }

        @Override
        public String toString() {
            return "Cluster{id=" + id + ", mass=" + mass + ", centroid=" + Arrays.toString(centroid) + "}";
        }
    }

    /** Represents a connection between two clusters */
    public static class Connection {
        public final Cluster cluster1;
        public final Cluster cluster2;
        public final double massProduct;
        public final double distance;
        public final double tau;

        public Connection(Cluster c1, Cluster c2, DistanceFunction d) {
            this.cluster1 = c1;
            this.cluster2 = c2;
            this.massProduct = c1.mass * c2.mass;
            this.distance = d.distance(c1.centroid, c2.centroid);
            this.tau = massProduct * (this.distance * this.distance);
        }

        @Override
        public String toString() {
            return "Connection{c1=" + cluster1.id + ", c2=" + cluster2.id + ", tau=" + tau + "}";
        }
    }

    /** Union-Find structure for managing point indices */
    public static class UnionFind {
        private final int[] parent;

        public UnionFind(int n) {
            parent = new int[n];
            for (var i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        public int find(int x) {
            var px = parent[x];
            return px != x ? (parent[x] = find(px)) : px;
        }

        public void union(int x, int y) {
            var rootX = find(x);
            var rootY = find(y);
            if (rootX != rootY)
                parent[rootX] = rootY;
        }
    }

    /**
     * Example usage with double[][] data.
     */
    public static void main(String[] args) {
        var data = new double[][]{
                {1.0, 2.0}, {1.1, 2.1}, {5.0, 8.0}, {5.1, 7.9}, {9.0, 1.0}, {9.1, 1.1}
        };
        var values = new Lst<double[]>(data.length);
        for (var d : data) values.add(d);
        var clustering = new TorqueClustering<double[]>(values, 2, DistanceFunction::distanceCartesianSq) {
            @Override
            public void coord(double[] x, double[] coords) {
                System.arraycopy(x, 0, coords, 0, x.length);
            }
        };
        clustering.cluster();
        var clusters = clustering.getFinalClusters();
        System.out.println("Final Clusters:");
        for (var entry : clusters.entrySet()) {
            System.out.println("Cluster " + entry.getKey() + ": " + entry.getValue());
        }
    }
}