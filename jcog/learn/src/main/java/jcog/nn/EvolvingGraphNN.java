package jcog.nn;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import jcog.activation.DiffableFunction;
import jcog.activation.LinearActivation;
import jcog.activation.TanhActivation;
import jcog.data.list.Lst;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.DeltaPredictor;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;

import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static jcog.Util.clamp;
import static jcog.Util.sub;

/**
 * EvolvingGraphNN: A flexible, evolving, recurrent-capable NN.
 * Introduces a ComputeModel abstraction:
 * - Default feedforward
 * - BPTT (backprop through time)
 * - Iterative stabilization
 *
 * Aligns with academic norms:
 * - BPTT for recurrent sequences (Werbos, 1990)
 * - Iterative fixed-point approaches (e.g., Almeida 1987)
 */
public class EvolvingGraphNN extends DeltaPredictor {
    static final int INPUT = -1, HIDDEN = 0, OUTPUT = 1;

    final List<Node> ins = new Lst<>();
    public final IntObjectHashMap<Node> hiddens = new IntObjectHashMap<>();
    final List<Node> outs = new Lst<>();

    public final Set<Edge> edges = Sets.newIdentityHashSet();

    public Random random = new XoRoShiRo128PlusRandom();

    public DiffableFunction activationDefault = TanhActivation.the;
    public double gradClamp = Double.POSITIVE_INFINITY;
    public double weightClamp = Double.POSITIVE_INFINITY;
    public double weightRangeDefault = 0.1;

    private int nextNodeID = 1;
    private transient List<Node> nodesSorted;
    private boolean networkChanged;

    private ComputeModel model =
        new IterativeComputeModel(4, 1E-9f);
        //new BPTTComputeModel(4);
        //new DefaultComputeModel();

    public EvolvingGraphNN() {}

    public EvolvingGraphNN(int inCount, int outCount) {
        addInputs(inCount);
        addOutputs(outCount);
    }

    public void setModel(ComputeModel model) { this.model = model; }
    public void setRandom(Random rng) { this.random = rng; }

    public void addInputs(int count) {
        for (int i = 0; i < count; i++) ins.add(newNode(LinearActivation.the, INPUT));
    }

    public void addOutputs(int count) {
        addOutputs(count, activationDefault);
    }

    public void addOutputs(int count, DiffableFunction activation) {
        for (int i = 0; i < count; i++) outs.add(newNode(activation, OUTPUT));
    }

    public Node newHiddenNode() {
        var n = newNode(activationDefault, HIDDEN);
        hiddens.put(n.id, n);
        setNetworkChanged();
        return n;
    }

    public void addEdge(Node from, Node to) {
        if (from == to) throw new UnsupportedOperationException("No self loops.");
        if (from.outs.stream().anyMatch(e->e.to==to)) return;
        var e = new Edge(from, to, weightRandom());
        edges.add(e);
        from.outs.add(e);
        to.ins.add(e);
        setNetworkChanged();
    }

    private List<Node> addLayer(int n) {
        var l = new Lst<Node>(n);
        for (var i = 0; i < n; i++)
            l.add(newHiddenNode());
        return l;
    }

    public void addLayersConnectFull(int... sizes) {
        assert (sizes.length > 0);
        var layers = new Lst<Iterable<Node>>(sizes.length + 2);
        layers.add(ins);
        for (int s : sizes)
            layers.add(addLayer(s));
        layers.add(outs);
        for (var i = 1; i < layers.size(); i++)
            connectFull(layers.get(i - 1), layers.get(i));
    }

    private void connectFull(Iterable<Node> from, Iterable<Node> to) {
        from.forEach(s -> to.forEach(t -> addEdge(s, t)));
    }

    public Node randomNode(boolean inputs, boolean hiddens, boolean outputs) {
        if (!inputs && !hiddens && !outputs) throw new UnsupportedOperationException();
        int viableNodes = (inputs ? inputCount() : 0) + (outputs ? outputCount() : 0) + (hiddens ? hiddenCount() : 0);
        if (viableNodes < 1) throw new UnsupportedOperationException();

        var nodes = nodes();
        do {
            var n = nodes.get(random.nextInt(nodes.size()));
            if ((inputs && n.type == INPUT) || (outputs && n.type == OUTPUT) || (hiddens && n.type == HIDDEN))
                return n;
        } while (true);
    }

    public int hiddenCount() {
        return hiddens.size();
    }

    private Node randomNode() {
        return randomNode(true, true, true);
    }

    public double[] forward(double[] x) { return model.forward(this, x); }
    public void backward(double[] delta, double lr) { model.backward(this, delta, lr); }

    public double[] put(double[] x, double[] y, double lr) {
        double[] out = forward(x);
        backward(sub(y, out), lr);
        return out;
    }

    public void put(double[][] X, double[][] Y, double lr) {
        for (int i = 0; i < X.length; i++)
            put(X[i], Y[i], lr);
    }

    public double[] outputValues() { return outs.stream().mapToDouble(n->n.value).toArray(); }
    public int inputCount() { return ins.size(); }
    public int outputCount() { return outs.size(); }

    public void removeWeakestNode() {
        hiddens.stream().filter(Node::isHidden)
                .min(Comparator.comparingDouble(Node::importance))
                .ifPresent(this::removeNode);
    }

    public void removeWeakestEdge() {
        edges.stream().min(Comparator.comparingDouble(e->Math.abs(e.weight)))
                .filter(e->!( (e.from.isInput() && e.from.outs.size()<2) || (e.to.isOutput() && e.to.ins.size()<2) ))
                .ifPresent(this::removeEdgeAndDisconnectedNode);
    }

    public boolean removeEdgeAndDisconnectedNode(Edge e) {
        if (removeEdge(e)) {
            if (e.from.outs.isEmpty()) removeNode(e.from);
            if (e.to.ins.isEmpty())   removeNode(e.to);
            return true;
        }
        return false;
    }

    public boolean removeEdge(Edge e) {
        if (edges.remove(e)) {
            e.from.outs.remove(e);
            e.to.ins.remove(e);
            setNetworkChanged();
            return true;
        }
        return false;
    }

    @Override public void putDelta(double[] d, float pri) { backward(d, pri); }
    @Override public double[] get(double[] x) { return forward(x); }
    @Override public void clear(Random rng) { setRandom(rng); edges.forEach(e->e.weight=weightRandom()); }

    public synchronized List<Node> nodes() {
        if (networkChanged || nodesSorted == null) {
            nodesSorted = topologicalSort();
            networkChanged = false;
        }
        return nodesSorted;
    }

    void setNetworkChanged() { networkChanged = true; nodesSorted = null; }
    double weightRandom() { return (random.nextDouble()*2-1)*weightRangeDefault; }

    private List<Node> topologicalSort() {
        Set<Node> visited = Sets.newIdentityHashSet();
        Lst<Node> sorted = new Lst<>(hiddens.size()+ins.size()+outs.size());
        Iterables.concat(ins, hiddens).forEach(n->visitNonOutputs(n, visited, sorted));
        var rev = Lists.reverse(sorted);
        rev.addAll(outs);
        return rev;
    }

    static void visitNonOutputs(Node n, Set<Node> visited, List<Node> sorted) {
        if (n.isOutput() || !visited.add(n)) return;
        for (var e : n.outs) visitNonOutputs(e.to, visited, sorted);
        sorted.add(n);
    }

    void removeNode(Node n) {
        if (hiddens.remove(n.id) != null) {
            List.copyOf(n.ins).forEach(this::removeEdge);
            List.copyOf(n.outs).forEach(this::removeEdge);
            setNetworkChanged();
        }
    }

    private Node newNode(DiffableFunction act, int t) {
        var n = new Node(nextNodeID++, act);
        n.type = (byte) t;
        if (t == HIDDEN) hiddens.put(n.id, n);
        return n;
    }

    double grad(double g) {
        return gradClamp == Double.POSITIVE_INFINITY ? g : clamp(g, -gradClamp, gradClamp);
    }

    double weight(double w) {
        return weightClamp == Double.POSITIVE_INFINITY ? w : clamp(w, -weightClamp, weightClamp);
    }

    public interface ComputeModel {
        double[] forward(EvolvingGraphNN nn, double[] input);
        void backward(EvolvingGraphNN nn, double[] delta, double lr);
    }

    public class DefaultComputeModel implements ComputeModel {
        @Override public double[] forward(EvolvingGraphNN nn, double[] input) {
            for (int i = 0; i < input.length; i++) ins.get(i).value = input[i];
            nn.nodes().forEach(Node::activate);
            return nn.outputValues();
        }

        @Override public void backward(EvolvingGraphNN nn, double[] delta, double lr) {
            for (int o = 0; o < outs.size(); o++) outs.get(o).grad(delta[o]);
            var N = nn.nodes();
            for (int i = N.size()-1; i>=0; i--) updateGrad(N.get(i));
            for (var n : N) {
                for (var e : n.ins) e.weight = weight(e.weight + grad(e.grad)*lr);
                if (!n.isInput()) n.bias = weight(n.bias + grad(n.grad)*lr);
            }
            nn.zeroGrad();
        }
    }

    public class BPTTComputeModel implements ComputeModel {
        private final int timesteps;
        private double[][] inputHistory;
        public BPTTComputeModel(int timesteps) { this.timesteps = timesteps; }

        @Override public double[] forward(EvolvingGraphNN nn, double[] input) {
            // Allocate history
            nn.nodes().forEach(n->n.valueHistory = new double[timesteps][]);
            inputHistory = new double[timesteps][];
            double[] out = null;
            for (int t = 0; t < timesteps; t++) {
                inputHistory[t] = input.clone();
                for (int i = 0; i < input.length; i++) ins.get(i).value = input[i];
                nn.nodes().forEach(Node::activate);
                for (int i = 0; i < nn.nodes().size(); i++) {
                    Node node = nn.nodes().get(i);
                    node.valueHistory[t] = new double[]{node.value};
                }
                out = nn.outputValues();
            }
            return out;
        }

        @Override public void backward(EvolvingGraphNN nn, double[] delta, double lr) {
            for (int o = 0; o < outs.size(); o++) outs.get(o).grad(delta[o]);
            var N = nn.nodes();
            for (int t = timesteps-1; t >= 0; t--) {
                for (int i = 0; i < N.size(); i++) {
                    Node node = N.get(i);
                    node.value = node.valueHistory[t][0];
                }
                for (int i = N.size()-1; i>=0; i--) updateGrad(N.get(i));
            }
            for (var n : N) {
                for (var e : n.ins) e.weight = weight(e.weight + grad(e.grad)*lr);
                if (!n.isInput()) n.bias = weight(n.bias + grad(n.grad)*lr);
            }
            nn.zeroGrad();
        }
    }

    public class IterativeComputeModel implements ComputeModel {
        private final int maxIter;
        private final double tol;
        public IterativeComputeModel(int maxIter, double tol) { this.maxIter=maxIter; this.tol=tol; }

        @Override public double[] forward(EvolvingGraphNN nn, double[] input) {
            for (int i = 0; i < input.length; i++) ins.get(i).value = input[i];
            double[] prev = new double[nn.nodes().size()];
            double[] curr = nn.nodes().stream().mapToDouble(n->n.value).toArray();
            int iter=0;
            double diff;
            do {
                nn.nodes().forEach(Node::activate);
                prev = curr;
                curr = nn.nodes().stream().mapToDouble(n->n.value).toArray();
                diff=0;
                for (int i=0; i<curr.length; i++) diff+=Math.abs(curr[i]-prev[i]);
            } while(diff>tol && ++iter<maxIter);
            return nn.outputValues();
        }

        @Override public void backward(EvolvingGraphNN nn, double[] delta, double lr) {
            for (int o = 0; o < outs.size(); o++) outs.get(o).grad(delta[o]);
            var N = nn.nodes();
            for (int i = N.size()-1; i>=0; i--) updateGrad(N.get(i));
            for (var n : N) {
                for (var e : n.ins) e.weight = weight(e.weight + grad(e.grad)*lr);
                if (!n.isInput()) n.bias = weight(n.bias + grad(n.grad)*lr);
            }
            nn.zeroGrad();
        }
    }

    void zeroGrad() {
        for (var e : edges) e.grad=0;
        for (var n : nodes()) n.zeroGrad();
    }

    static void updateGrad(Node n) {
        if (!n.isOutput()) n.grad(n.outs.stream().mapToDouble(Edge::gradWeighted).sum());
        double g = n.grad;
        for (var e:n.ins) e.grad = g * e.from.value;
    }

    public static class Node {
        public final int id;
        public byte type;
        public double value, bias, grad;
        public DiffableFunction activation;
        public List<Edge> ins = new Lst<>(), outs = new Lst<>();
        public double[][] valueHistory;

        Node(int id, DiffableFunction act) { this.id=id; this.activation=act; }

        void activate() {
            if (isInput()) return;
            double sum = bias; for (var e:ins) sum += e.weight*e.from.value;
            value = activation!=null? activation.valueOf(sum):sum;
        }

        public double importance() {
            double imp= Math.abs(bias);
            for (var e:ins) imp+= Math.abs(e.weight);
            for (var e:outs) imp+= Math.abs(e.weight);
            return imp;
        }

        void grad(double d) { grad += d*(activation!=null?activation.derivative(value):1); }
        void zeroGrad() { grad=0; }

        public boolean isInput(){return type==INPUT;}
        public boolean isOutput(){return type==OUTPUT;}
        boolean isHidden(){return type==HIDDEN;}

        public boolean hasEdgeFrom(Node x) {
            return ins.stream().anyMatch(z -> z.from == x);
        }

        public boolean hasEdgeTo(Node x) {
            return outs.stream().anyMatch(z -> z.to == x);
        }
    }

    public static class Edge {
        public final Node from, to;
        public double weight, grad;
        Edge(Node f, Node t, double w){ from=f; to=t; weight=w; }
        double gradWeighted(){ return to.grad*weight; }
    }
}
