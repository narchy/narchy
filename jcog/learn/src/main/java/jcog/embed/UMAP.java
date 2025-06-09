package jcog.embed;

import jcog.random.XoRoShiRo128PlusRandom;

import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

/**
 * A simplified, enhanced UMAP-like class in Java with:
 *   - Efficient indexing
 *   - Pluggable distance function
 *   - Basic usage of minDist/spread
 *   - 2D Swing visualization example
 */
public class UMAP<X> {

    /* ==================== Configuration ==================== */

    /** Number of neighbors to consider. */
    private int nNeighbors = 15;

    /** UMAP's minDist parameter (controls how tightly points can be packed). */
    private double minDist = 0.1;

    /** UMAP's spread parameter (controls the scale of the neighborhoods). */
    private double spread = 1.0;

    /** Number of layout epochs. */
    private int nEpochs = 10;

    /** Initial embedding range (points placed randomly in a box of this size). */
    private double embeddingInitRadius = 0.01;

    /** Distance function. Defaults to Euclidean if none is supplied. */
    private final BiFunction<double[], double[], Double> distanceFn;

    /** Embedding dimension (2 for typical visualization, but can be 3, etc.). */
    private final int dimEmbed;

    /* ==================== Data Structures ==================== */

    /**
     * A list of points (including data, embedding, etc.).
     * The index in this list is the unique ID used internally.
     */
    private final List<Point<X>> points = new ArrayList<>();

    /**
     * Map from the user-provided key (X) to the integer index in the 'points' list.
     */
    private final Map<X,Integer> indexByKey = new ConcurrentHashMap<>();

    /**
     * Internal graph that stores edges (for attractiveness between neighbors).
     */
    private final Graph graph = new Graph();


    /* ==================== Constructors ==================== */

    public UMAP(int dimEmbed) {
        this(dimEmbed, UMAP::euclideanDistance);
    }

    public UMAP(int dimEmbed, BiFunction<double[], double[], Double> distanceFn) {
        if (dimEmbed <= 0) {
            throw new IllegalArgumentException("Embedding dimension must be >= 1");
        }
        this.dimEmbed = dimEmbed;
        this.distanceFn = distanceFn;
    }

    /* ==================== Public Configuration Methods ==================== */

    public void setNNeighbors(int nNeighbors) {
        this.nNeighbors = nNeighbors;
    }

    public void setMinDist(double minDist) {
        this.minDist = minDist;
    }

    public void setSpread(double spread) {
        this.spread = spread;
    }

    public void setNEpochs(int nEpochs) {
        this.nEpochs = nEpochs;
    }

    public void setEmbeddingInitRadius(double embeddingInitRadius) {
        this.embeddingInitRadius = embeddingInitRadius;
    }

    /* ==================== Public Data Handling ==================== */

    /**
     * Adds a data point with default weight=1.
     */
    public void put(X key, double[] data) {
        put(key, data, 1.0, NodeType.DATA, null);
    }

    /**
     * Adds a data point with the specified weight.
     */
    public void put(X key, double[] data, double weight) {
        put(key, data, weight, NodeType.DATA, null);
    }

    /**
     * Adds an anchor (fixed embedding). Does not move during layout.
     * Example usage: pin a certain reference point in place.
     */
    public void putAnchor(X key, double[] anchorEmbedding, double weight) {
        put(key, null, weight, NodeType.ANCHOR, anchorEmbedding);
    }

    /**
     * Adds a virtual node. Has no 'data' but is free to move in embedding space.
     */
    public void putVirtual(X key, double weight) {
        put(key, null, weight, NodeType.VIRTUAL, null);
    }

    /**
     * Removes a point from the dataset (if present).
     */
    public void remove(X key) {
        Integer idx = indexByKey.remove(key);
        if (idx != null && idx < points.size()) {
            points.set(idx, null); // "tombstone"
        }
    }

    /* ==================== Graph / Edges ==================== */

    /**
     * Adds an edge with default weight=1, attract=1.0
     */
    public void addEdge(X from, X to) {
        addEdge(from, to, 1.0, 1.0);
    }

    /**
     * Adds an edge with custom weight and attract factor.
     * 'weight' can be seen as how strongly these nodes are connected.
     * 'attract' is an additional factor that can scale the attractive force.
     */
    public void addEdge(X from, X to, double weight, double attract) {
        int idxFrom = getIndexOrThrow(from);
        int idxTo = getIndexOrThrow(to);
        if (idxFrom == idxTo) {
            throw new IllegalArgumentException("No self-loops allowed: " + from);
        }
        graph.addEdge(idxFrom, idxTo, weight, attract);
    }

    /* ==================== UMAP-Like Computation ==================== */

    /**
     * Recomputes neighbors, builds the graph edges, and optimizes the layout.
     */
    public void commit() {
        // Clean out any nulls from 'remove()'
        // Rebuild a compact array of points
        compactPointsList();

        int n = points.size();
        if (n == 0) return;

        nNeighbors = Math.min(nNeighbors, n - 1);

        // Rebuild the graph from scratch
        graph.clear();

        // 1) Compute distances NxN
        double[][] dists = computeDistances(points);

        // 2) Find k-nearest neighbors
        int[][] knnIndex = findKNN(dists, nNeighbors);

        // 3) Compute sigmas (simplified version)
        double[] sigmas = computeSigmas(dists);

        // 4) Build edges from knn
        buildEdgesFromKNN(knnIndex, dists, sigmas);

        // 5) Optimize layout with a simple epochs loop
        optimizeLayout();
    }

    /**
     * Retrieve a map of key -> final embedding array.
     */
    public Map<X, double[]> getEmbeddings() {
        Map<X, double[]> result = new HashMap<>();
        for (Point<X> p : points) {
            if (p != null && p.type != null) {
                result.put(p.id, p.embed);
            }
        }
        return result;
    }

    /* ==================== Swing Visualization (2D only) ==================== */

    /**
     * Quick helper to visualize the current 2D embedding in a Swing window.
     * Ignores points if dimEmbed != 2.
     */
    public void showSwing2D(String title) {
        if (dimEmbed != 2) {
            throw new IllegalStateException("Visualization is 2D only. dimEmbed=" + dimEmbed);
        }

        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Copy references to avoid concurrency issues
        List<Point<X>> snapshot = new ArrayList<>(points);

        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g;
                g2.setColor(Color.BLUE);

                // Simple bounding box find
                double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
                double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;

                for (Point<X> p : snapshot) {
                    if (p == null) continue;
                    double x = p.embed[0];
                    double y = p.embed[1];
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }

                double rangeX = maxX - minX;
                double rangeY = maxY - minY;
                if (rangeX < 1e-9) rangeX = 1; // avoid div0
                if (rangeY < 1e-9) rangeY = 1;

                int w = getWidth(), h = getHeight();

                for (Point<X> p : snapshot) {
                    if (p == null) continue;

                    double x = p.embed[0];
                    double y = p.embed[1];

                    // Normalize [min..max] -> [0..1], then multiply by pixel size
                    int px = (int) ((x - minX) / rangeX * (w - 10) + 5);
                    int py = (int) ((y - minY) / rangeY * (h - 10) + 5);

                    if (p.type == NodeType.ANCHOR) {
                        g2.setColor(Color.RED);
                        g2.fillOval(px - 5, py - 5, 10, 10);
                        g2.setColor(Color.BLUE);
                    } else {
                        g2.fillOval(px - 3, py - 3, 6, 6);
                    }
                }
            }
        };
        panel.setPreferredSize(new Dimension(600, 600));
        frame.add(panel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    /* ==================== Internal Implementation ==================== */

    /** Compact 'points' list to remove any nulls/tombstones. Rebuild indexByKey. */
    private void compactPointsList() {
        List<Point<X>> clean = new ArrayList<>();
        indexByKey.clear();
        for (Point<X> p : points) {
            if (p != null) {
                indexByKey.put(p.id, clean.size());
                clean.add(p);
            }
        }
        points.clear();
        points.addAll(clean);
    }

    /**
     * Main "add" method that handles anchor, virtual, data node.
     */
    private void put(X key, double[] data, double weight, NodeType type, double[] anchorEmbedding) {
        Integer existing = indexByKey.get(key);
        if (existing != null) {
            // overwrite?
            throw new IllegalArgumentException("Key already exists: " + key);
        }

        double[] embed;
        if (type == NodeType.ANCHOR && anchorEmbedding != null) {
            // use provided anchor
            if (anchorEmbedding.length != dimEmbed) {
                throw new IllegalArgumentException("Anchor embedding dimension mismatch");
            }
            embed = Arrays.copyOf(anchorEmbedding, dimEmbed);
        } else {
            // random init
            embed = randomEmbedding(dimEmbed, embeddingInitRadius);
        }

        Point<X> p = new Point<>(key, data, embed, type, weight);
        int idx = points.size();
        points.add(p);
        indexByKey.put(key, idx);
    }

    /** Retrieve integer index for an existing key or throw error if not found. */
    private int getIndexOrThrow(X key) {
        Integer idx = indexByKey.get(key);
        if (idx == null) {
            throw new NoSuchElementException("Key not found: " + key);
        }
        return idx;
    }

    /** NxN distance matrix. For large n, consider approximate methods. */
    private double[][] computeDistances(List<Point<X>> pts) {
        int n = pts.size();
        double[][] dist = new double[n][n];
        for (int i = 0; i < n; i++) {
            Point<X> pi = pts.get(i);
            for (int j = i; j < n; j++) {
                if (i == j) {
                    dist[i][j] = 0.0;
                } else {
                    Point<X> pj = pts.get(j);
                    // If either is anchor or data == null => skip?
                    if (pi.data == null && pj.data == null) {
                        // If both are "virtual" or "anchor" with no data, set big distance
                        dist[i][j] = dist[j][i] = Double.POSITIVE_INFINITY;
                    } else if (pi.data == null || pj.data == null) {
                        // One side is virtual/anchor with no data => infinite distance
                        dist[i][j] = dist[j][i] = Double.POSITIVE_INFINITY;
                    } else {
                        double d = distanceFn.apply(pi.data, pj.data);
                        dist[i][j] = dist[j][i] = d;
                    }
                }
            }
        }
        return dist;
    }

    /**
     * Finds k nearest neighbors for each row in dist[] using a naive partial sort.
     */
    private int[][] findKNN(double[][] dist, int k) {
        int n = dist.length;
        int[][] indices = new int[n][k];

        for (int i = 0; i < n; i++) {
            // We’ll create an array of (distance, index) so we can partial-sort
            double[] row = dist[i];
            Integer[] all = IntStream.range(0, n).boxed().toArray(Integer[]::new);

            // Sort by distance
            Arrays.sort(all, Comparator.comparingDouble(o -> row[o]));

            // Take the first k that aren't i itself
            int count = 0, idxPos = 0;
            while (count < k && idxPos < all.length) {
                int candidate = all[idxPos++];
                if (candidate == i) continue; // skip self
                if (Double.isInfinite(row[candidate])) continue; // skip infinite
                indices[i][count++] = candidate;
            }
            // If we couldn't find k valid neighbors, fill the rest with -1
            while (count < k) {
                indices[i][count++] = -1;
            }
        }
        return indices;
    }

    /**
     * Simplified approach to computing "sigma" for each row,
     * e.g. log of the mean distance. Real UMAP does a binary search
     * to match perplexity. This is just a placeholder.
     */
    private double[] computeSigmas(double[][] dist) {
        int n = dist.length;
        double[] sigmas = new double[n];
        for (int i = 0; i < n; i++) {
            double[] row = dist[i];
            // compute mean ignoring Infinity
            double sum = 0;
            int count = 0;
            for (double d : row) {
                if (!Double.isInfinite(d) && d > 0) {
                    sum += d;
                    count++;
                }
            }
            double mean = (count == 0) ? 1.0 : (sum / count);
            // simple approach
            sigmas[i] = Math.log1p(mean);
        }
        return sigmas;
    }

    /**
     * Build edges from kNN with minDist/spread logic.
     * weight = exp(-((dist-minDist) / (sigma)) ) if dist>minDist else ...
     */
    private void buildEdgesFromKNN(int[][] knnIndex, double[][] dist, double[] sigmas) {
        graph.clear();
        int n = knnIndex.length;
        for (int i = 0; i < n; i++) {
            int[] neighbors = knnIndex[i];
            double sigma = sigmas[i];

            for (int nn : neighbors) {
                if (nn < 0 || nn == i) continue;
                double d = dist[i][nn];

                // Basic minDist/spread approach:
                double effectiveDist = (d > minDist) ? (d - minDist) : 0.0;
                // scale by 'spread' as well:
                effectiveDist = effectiveDist / spread;

                // approximate weight function
                double w = Math.exp(-Math.max(0.0, effectiveDist) / sigma);

                // 1.0 is a fixed "attract" factor
                graph.addEdge(i, nn, w, 1.0);
            }
        }
    }

    /**
     * Simple layout optimization: for each epoch,
     *   - For each edge, compute the force, apply to the "from" node (if not anchor).
     *
     * Real UMAP would also do negative sampling, learning rate decay, etc.
     */
    private void optimizeLayout() {
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            for (Graph.Edge e : graph.getAllEdges()) {
                Point<X> pFrom = points.get(e.from());
                Point<X> pTo = points.get(e.to());
                if (pFrom.type == NodeType.ANCHOR) {
                    // anchored => skip
                    continue;
                }

                // compute force
                double[] vec = computeForce(pFrom.embed, pTo.embed, e.weight(), e.attract());
                // apply scaled by the node's weight
                addInPlace(pFrom.embed, vec, pFrom.weight);
            }
        }
    }

    /**
     * The "UMAP" force function: (a[i] - b[i]) / (1 + (diff^2)) scaled by weight * attract
     */
    private static double[] computeForce(double[] a, double[] b, double weight, double attract) {
        int n = a.length;
        double[] force = new double[n];
        for (int i = 0; i < n; i++) {
            double diff = a[i] - b[i];
            double denom = 1.0 + diff * diff; // naive
            force[i] = weight * attract * diff / denom;
        }
        return force;
    }

    /**
     * Adds `src` scaled by factor to `dest`.
     */
    private static void addInPlace(double[] dest, double[] src, double factor) {
        for (int i = 0; i < dest.length; i++) {
            dest[i] += src[i] * factor;
        }
    }

    /**
     * Returns a random embedding array of length 'dim',
     * each component in [0..radius).
     */
    private static double[] randomEmbedding(int dim, double radius) {
        double[] e = new double[dim];
        Random r = ThreadLocalRandom.current();
        for (int i = 0; i < dim; i++) {
            e[i] = r.nextDouble() * radius;
        }
        return e;
    }

    /**
     * Default Euclidean distance.
     */
    private static double euclideanDistance(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /* ==================== Nested Classes & Enums ==================== */

    /** Simple node types. */
    public enum NodeType {
        DATA, VIRTUAL, ANCHOR
    }

    /**
     * A single data point record: includes the user’s id (X),
     * the high-dimensional data, the current embedding, the node type, and a weight factor.
     */
    private static class Point<X> {
        final X id;
        final double[] data;
        final double[] embed;
        final NodeType type;
        final double weight;

        Point(X id, double[] data, double[] embed, NodeType type, double weight) {
            this.id = id;
            this.data = data;
            this.embed = embed;
            this.type = type;
            this.weight = weight;
        }

        @Override
        public String toString() {
            return id + ":" + type + " => " + Arrays.toString(embed);
        }
    }

    /**
     * Holds adjacency lists for each node. Each list is a set of edges from that node.
     */
    private static class Graph {
        private final List<List<Edge>> adjacency = new ArrayList<>();

        public void clear() {
            adjacency.clear();
        }

        /**
         * Add an edge (directed) from 'from' to 'to' with certain weight, attract.
         */
        public void addEdge(int from, int to, double weight, double attract) {
            ensureCapacity(from);
            adjacency.get(from).add(new Edge(from, to, weight, attract));
        }

        /**
         * Return all edges in the graph as a single list.
         */
        public List<Edge> getAllEdges() {
            List<Edge> all = new ArrayList<>();
            for (List<Edge> edges : adjacency) {
                all.addAll(edges);
            }
            return all;
        }

        private void ensureCapacity(int idx) {
            while (adjacency.size() <= idx) {
                adjacency.add(new ArrayList<>());
            }
        }

        record Edge(int from, int to, double weight, double attract) {}
    }

    /* ==================== Main Demo ==================== */

    public static void main(String[] args) {
        // Example usage
        UMAP<String> u = new UMAP<>(2); // 2D embedding
        u.setNNeighbors(2);
        u.setMinDist(0.1);
        u.setSpread(1.0);
        u.setNEpochs(20);
        u.setEmbeddingInitRadius(0.01);

        // Add some points
//        umap.put("A", new double[]{0.0, 0.0});
//        umap.put("B", new double[]{0.1, 0.1});
//        umap.put("C", new double[]{1.0, 1.0});
//        umap.put("D", new double[]{2.0, 2.0});

        var rng = new XoRoShiRo128PlusRandom();
        for (int i = 0; i < 1000; i++) {
            u.put("i" + i, new double[] { rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat() });
        }


        // Add a virtual node
        u.putVirtual("V1", 1.0);

        // Add an anchor (fixed in place at [2.5, 2.5])
        u.putAnchor("Anchor", new double[]{2.5, 2.5}, 1.0);

        // Possibly add edges if you want custom relationships
        // umap.addEdge("A", "V1", 1.0, 2.0);

        // Optimize
        u.commit();

        // Print result
        var embeddings = u.getEmbeddings();
        System.out.println("Final Embeddings:");
        embeddings.forEach((k, v) ->
            System.out.println("  " + k + " => " + Arrays.toString(v))
        );

        // Show in a Swing window (only valid if dimEmbed=2)
        u.showSwing2D("UMAP Enhanced Demo");
    }
}
