package jcog.memoize;

import jcog.data.list.Lst;
import jcog.data.set.ArrayHashSet;
import org.eclipse.collections.impl.tuple.Tuples;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.function.*;

/** memoization computation graph context */
public class MemoGraph {
    private final Map<Object, Node> nodes = new HashMap<>();
    private final Map<Object, Object> val = new HashMap<>();

    private final Set<Node> active = new HashSet<>();
    private final Queue<Consumer<MemoGraph>> queue = new ArrayDeque<>();

    private final Map<Class<? extends BiFunction>, SharedFn> shared = new HashMap<>();

    /** sequence to execute after compile */
    private Lst<Node> proc;

    private boolean running;

    private static final boolean SUPPLY_NULLS = false;

    private boolean sharedFnRetain = true;

    private Node node(Object key) {
        var node = nodes.get(key);
        if (node == null) throw new IllegalArgumentException("Key not found: " + key);
        return node;
    }

    private Node nodeAddUnique(Object key) {
        assertNotRunning();

        var n = nodes.get(key);
        if (n!=null)
            throw new IllegalStateException("duplicate key");
        var node = new Node<>(key);
        nodes.put(key, node);
        return node;
    }

    private Node nodeAdd(Object key) {
        assertNotRunning();
        return nodes.computeIfAbsent(key, Node::new);
    }

    private <K> Node<K,?> nodeAdd(K key, Function<K,?> f) {
        assertNotRunning();
        return nodes.computeIfAbsent(key, k -> new Node(k, f));
    }

    private <K> Node<K,?> nodeAdd(K key, Supplier s) {
        return nodeAdd(key, k->s.get());
    }

    public MemoGraph put(Object key, Supplier s) {
        if (running) {
            queue.add((G)->G.put(key, s));
            return this;
        }

        nodeAdd(key, s);
        return this;
    }

    public <X,Y> MemoGraph then(Object inKey, Function<X, Y> f, Object outKey) {
        if (running) {
            queue.add((G)->G.then(inKey, f, outKey));
        } else {
            var in = node(inKey);
            var out = nodeAddUnique(outKey);
            out.addIn(in);
            in.addConsumer(outKey, f);
            in.addOut(out);
        }

        return this;
    }

    public <R> MemoGraph thenAll(List<Object> inKeys, Function<List, R> f, Object outKey) {
        var is = inKeys.size();
        if (is == 0)
            throw new UnsupportedOperationException();
        else if (is ==1)
            return then(inKeys.getFirst(), f, outKey);

        if (running) {
            queue.add((G)->G.thenAll(inKeys, f, outKey));
            return this;
        }

        var out = nodeAdd(outKey);

        var inNodes = inKeys.stream().map(this::node).toList();

        // Add dependency connections
        for (var in : inNodes) {
            in.addOut(out);
            out.addIn(in);
        }

        // Add consumer only to the last input node
        // It will execute only when all dependencies are ready
        inNodes.getLast().addConsumer(outKey, x -> f.apply(all(inNodes)));
        return this;
    }

    private Lst<Object> all(List<Node> inNodes) {
        var l = new Lst<>(inNodes.size());
        for (var n : inNodes)
            l.add(get(n.key));
        return l;
    }

    /**
     * Register a consumer for a specific output key
     */
    public <X> MemoGraph get(Object key, Consumer<X> consumer) {
        assertNotRunning();
        var n = node(key);
        n.addOutputConsumer(consumer);
        active.add(n);
        return this;
    }

    @SuppressWarnings("unchecked")
    public MemoGraph run() {
        while (compile()) {
            running = true;

            compute();
            runActive();

            running = false;

            runQueued();
        }
        return this;
    }

    private void compute() {
        var p = this.proc;
        this.proc = null;
        for (int i = 0, size = p.size(); i < size; i++)
            p.getAndNull(i).compute(val);
        p.delete();
    }

    /** notify consumers */
    private void runActive() {
        active.removeIf(n -> {
            n.supply(val);
            return true;
        });
    }

    private void runQueued() {
        Consumer<MemoGraph> r;
        while ((r = queue.poll())!=null) {
            r.accept(this);
        }
    }

    private boolean compile() {
        if (proc != null)
            throw new IllegalStateException("Already compiled");

        return !active.isEmpty() && (this.proc = procedure()) != null;
    }

    /** Single-pass topological sort of only required nodes */
    private Lst<Node> procedure() {
        return _procedure().list.reverseThis();
    }

    private ArrayHashSet<Node> _procedure() {
        var required = required();
        var as = new ArrayHashSet<Node>(required.size());
        for (var n : required)
            if (!n.visit(as, required))
                throw new IllegalStateException("Cycle detected");
        return as;
    }

    /** Collect required nodes working backwards from active nodes */
    private Set<Node> required() {
        var required = new LinkedHashSet<Node>();
        var queue = new ArrayDeque<>(active);
        while (!queue.isEmpty()) {
            var n = queue.poll();
            if (required.add(n) && n.in != null)
                queue.addAll(n.in);
        }
        return required;
    }

    @SuppressWarnings("unchecked")
    public <X> X get(Object key) {
        var result = val.get(key);
        if (result == null)
            throw MISSING_VALUE;

        return (X) result;
    }

    public void clear() {
        running = false; proc = null; queue.clear(); active.clear();
//        if (running || proc != null || !queue.isEmpty() || !active.isEmpty())
//            throw new IllegalStateException("clear() fault");

        //nodes.values().forEach(Node::clear);
        nodes.clear();

        val.clear();

        if (sharedFnRetain)
            shared.values().forEach(SharedFn::clear);
        else
            shared.clear();
    }

    private void assertNotRunning() {
        if (running)
            throw new IllegalStateException("Already running");
    }

    /** this is like a combined put/get, which will run in isolation even if nothign depends on it */
    public void once(Object key, Runnable r) {
        once(key, k->r.run());
    }

    public <X,Y> void once(X k1, Y k2, Consumer<Y> c) {
        once(k1, k2, (x,y)->c.accept(y));
    }

    public <X,Y> void once(X k1, Y k2, BiConsumer<X,Y> c) {
        once(Tuples.pair(k1, k2),
                p->c.accept(p.getOne(), p.getTwo()));
    }

    public final <X> void once(X x, Consumer<X> c) {
        //Objects.requireNonNull(x);
        if (running)
            queue.add(G -> G._once(x, c));
        else
            _once(x, c);
    }

    private <X> void _once(X x, Consumer<X> c) {
        var prev = nodes.get(x);
        if (prev == null) {
            var n = new Node<>(x, c);
            nodes.put(x, n);
            active.add(n);
        }
    }

    /**
     * Compute or reuse shared result while maintaining task isolation
     */
    @SuppressWarnings("unchecked")
    private <X, Y> SharedFn<X, Y> shared(BiFunction<MemoGraph, X, Y> f) {
        return shared.computeIfAbsent(
                f.getClass(),
                k -> new SharedFn<>(f)
        );
    }

    public final <X, Y> Y share(X x, Function<X, Y> f) {
        return share(x, (g,X)->f.apply(X));
    }

    @Nullable
    public <X, Y> Y share(X x, BiFunction<MemoGraph, X, Y> f) {
        //Objects.requireNonNull(x);
        return shared(f).get(x, this);
    }

    public <X, Y> void chain(X x, BiFunction<MemoGraph, X, Y> f, BiConsumer<X, Y> cont) {
        once(x, X -> {
            var s = shared(f);
            var y = s.get(X, this);
            if (y != null)
                cont.accept(X, y);
            else
                s.await(X, cont);
        });
    }

    private static <T> Set<T> lazySet(Set<T> set) {
        return set == null ? newSet() : set;
    }

    private static <T> Set<T> newSet() {
        return new HashSet<>(1);
        //return new UnifiedSet<>(1);
    }

    private static <T> List<T> lazyList(List<T> list) {
        return list == null ? new Lst<>(1) : list;
    }

    private static class Keysumer<X, Y> {
        protected final Function<X, Y> f;
        final Object key;

        Keysumer(X outKey, Function<X, Y> f) {
            this.key = outKey;
            this.f = f;
        }

        boolean run(Object x, Map<Object, Object> values) {
            return null!=values.computeIfAbsent(/*out*/key, k -> x == null ? null : f.apply((X) x));
        }

        @Override public final int hashCode() {
            return key.hashCode();
        }

        @Override
        public final boolean equals(Object x) {
            return this==x || (x instanceof Keysumer X && key.equals(X.key));
        }
    }

    private static final class Node<X,Y> extends Keysumer {

        Node(X key, Consumer<X> f) {
            this(key, (x)->{
                f.accept(x);
                return null;
            });
        }

        Node(X key) {
            this(key, (Function)null);
        }

        Node(X key, Function<X,Y> f) {
            super(key, f);
        }

        @Nullable Set<Node> in, out;

        @Nullable List<Keysumer<?, ?>> keysumers;

        /** output consumers */
        @Nullable List<Consumer> outsumers;

        void addIn(Node n) {
            (in = lazySet(in)).add(n);
        }

        void addOut(Node n) {
            (out = lazySet(out)).add(n);
        }

        void addOutputConsumer(Consumer<X> consumer) {
            (outsumers = lazyList(outsumers)).add(consumer);
        }

        void addConsumer(X key, Function<X,Y> f) {
            (keysumers = lazyList(keysumers)).add(new Keysumer<>(key, f));
        }

        void compute(Map values) {
            var out = keysumers != null;
            var value = value(values, out);
            if (out)
                for (var c : keysumers)
                    c.run(value, values);
        }

        private Object value(Map values, boolean out) {
            return (f != null) ? values.computeIfAbsent(key, f) :
                (out ? values.get(key) : null);
        }

        boolean visit(ArrayHashSet<Node> visited, Set<Node> required) {
            if (!visited.add(this))
                return true;

            if (this.out != null) {
                for (var dep : this.out)
                    if (required.contains(dep) && !dep.visit(visited, required))
                        return false;
            }

            return true;
        }

        void supply(Map<Object, Object> val) {
            var v = val.get(key);
            if (v != null || SUPPLY_NULLS)
                for (var c : outsumers)
                    c.accept(v);
        }

        public void clear() {
            if (in!=null) { in.clear(); in = null; }
            if (out!=null) { out.clear(); out = null; }
            if (keysumers !=null) { keysumers.clear(); keysumers = null; }
            if (outsumers !=null) { outsumers.clear(); outsumers = null; }
        }

    }


    private static class SharedFn<X, Y> {
        private final Map<Object, Y> computed = new HashMap<>();
        private final Map<Object, Deque<BiConsumer<X, Y>>> pending = new HashMap<>();
        private final BiFunction<MemoGraph, X, Y> f;

        SharedFn(BiFunction<MemoGraph, X, Y> f) {
            this.f = f;
        }

        @Nullable
        Y get(X x, MemoGraph g) {
            // Already computed?
            var yKnown = computed.get(x);
            if (yKnown != null)
                return yKnown;

            if (g.running) {
                g.queue.add((G) -> compute(x, G));
                return null; // Trigger await
            } else
                return compute(x, g);
        }

        private @Nullable Y compute(X key, MemoGraph g) {
            var y = f.apply(g, key);
            computed.put(key, y);
            if (y != null)
                notify(key, y);
            return y;
        }

        private void notify(X x, Y y) {
            var consumers = pending.remove(x);
            if (consumers != null) {
                BiConsumer<X, Y> next;
                while ((next=consumers.poll())!=null) {
                    next.accept(x, y);
                }
            }
        }

        private void await(X key, BiConsumer<X, Y> consumer) {
            var y = computed.get(key);
            if (y != null)
                consumer.accept(key, y);
            else
                pending.computeIfAbsent(key, k ->
                        new ArrayDeque<>(1)).add(consumer);
        }

        void clear() {
            computed.clear();
            pending.clear();
        }
    }

    public static final IllegalStateException MISSING_VALUE = new IllegalStateException("Missing value");
//    public static final IllegalStateException CLEAR_FAULT = new IllegalStateException("Clear fault");

}