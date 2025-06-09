package jcog.data.graph;

import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Streams;
import jcog.Util;
import jcog.data.graph.edge.ImmutableDirectedEdge;
import jcog.data.graph.path.FromTo;
import jcog.data.graph.search.Search;
import jcog.data.iterator.Concaterator;
import jcog.data.list.Lst;
import jcog.data.map.UnifriedMap;
import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.api.tuple.primitive.BooleanObjectPair;
import org.jetbrains.annotations.Nullable;

import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static jcog.Util.emptyIterable;
import static jcog.data.graph.search.Search.newQueue;


/**
 * graph rooted in a set of vertices (nodes),
 * providing access to edges only indirectly through them
 * (ie. the edges are not stored in their own index but only secondarily as part of the vertices)
 * <p>
 * TODO abstract into subclasses:
 * HashNodeGraph backed by HashMap node and edge containers
 * BagNodeGraph backed by Bag's/Bagregate's
 * <p>
 * then replace TermWidget/EDraw stuff with BagNodeGraph
 */
public class MapNodeGraph<N, E> {

    protected final Map<N, AbstractNode<N, E>> nodes;

    public MapNodeGraph(Map<N, AbstractNode<N, E>> nodes) {
        this.nodes = nodes;
    }

    public MapNodeGraph(int initialCapacity) {
        this(new UnifriedMap<>(initialCapacity));
        //this(new LinkedHashMap<>(initialCapacity));
        //this(new HashMap<>(initialCapacity));
    }

    public MapNodeGraph() {
        this(0);
    }

    static <N, E> boolean addEdge(MutableNode<N, E> from, MutableNode<N, E> to, FromTo<AbstractNode<N, E>, E> ee) {
        if (from.addOut(ee)) {
            var a = to.addIn(ee);
            //assert (a);
            return true;
        }
        return false;
    }

    public void clear() {
        nodes.clear();
    }

    public final boolean removeNode(N key) {
        return removeNode(key, null, null);
    }

    public boolean removeNode(N key, @Nullable Consumer<FromTo<AbstractNode<N, E>, E>> inEdges, @Nullable Consumer<FromTo<AbstractNode<N, E>, E>> outEdges) {
        var removed = nodes.remove(key);
        if (removed == null) return false;

        removed.edges(true, false).forEach(compose(inEdges, this::edgeRemoveOut));
        removed.edges(false, true).forEach(compose(outEdges, this::edgeRemoveIn));
        onRemoved(removed);
        return true;
    }

    private static <N, E> Consumer<FromTo<AbstractNode<N, E>, E>> compose(@Nullable Consumer<FromTo<AbstractNode<N, E>, E>> x, Consumer<FromTo<AbstractNode<N, E>, E>> y) {
        return x != null ? x.andThen(y) : y;
    }

    public final MutableNode<N, E> addNode(N key) {
        return (MutableNode<N, E>) nodes.compute(key, (k, existing) -> {
            if (existing != null)
                return existing;
            else {
                var newNode = newNode(k);
                onAdd(newNode);
                return newNode;
            }
        });
    }

    protected MutableNode<N, E> newNode(N data) {
        return new MutableNode<>(data);
    }

    protected void onAdd(AbstractNode<N, E> r) {

    }

    protected void onRemoved(AbstractNode<N, E> r) {

    }

    /**
     * creates the nodes if they do not exist yet
     */
    public final boolean addEdge(N from, E data, N to) {
        return addEdge(addNode(from), data, to);
    }

    public final boolean addEdgeIfNodesExist(N from, E data, N to) {
        return addEdgeByNode((MutableNode<N, E>) node(from), data, (MutableNode<N, E>) node(to));
    }

    public final boolean addEdgeByNode(MutableNode<N, E> from, E data, MutableNode<N, E> to) {
        return addEdge(from, to, new ImmutableDirectedEdge<>(from, data, to));
    }

    public final boolean addEdge(MutableNode<N, E> from, E data, N to) {
        return addEdgeByNode(from, data, addNode(to));
    }

    public final boolean addEdge(N from, E data, MutableNode<N, E> to) {
        return addEdgeByNode(addNode(from), data, to);
    }

    public boolean addEdge(FromTo<AbstractNode<N, E>, E> ee) {
        return addEdge((MutableNode<N, E>) (ee.from()), (MutableNode<N, E>) (ee.to()), ee);
    }

    public final AbstractNode<N, E> node(Object key) {
        return nodes.get(key);
    }

    public final Collection<AbstractNode<N, E>> nodes() {
        return nodes.values();
    }

    public final int nodeCount() {
        return nodes.size();
    }

    public final void forEachNode(Consumer<AbstractNode<N, E>> n) {
        nodes().forEach(n);
    }

    public boolean edgeRemove(FromTo<AbstractNode<N, E>, E> e) {
        if (edgeRemoveOut(e)) {
            var removed = edgeRemoveIn(e);
            //assert (removed);
            return true;
        }
        return false;
    }

    private boolean edgeRemoveIn(FromTo<AbstractNode<N, E>, E> e) {
        return ((MutableNode) e.to()).removeIn(e);
    }

    private boolean edgeRemoveOut(FromTo<AbstractNode<N, E>, E> e) {
        return ((MutableNode) e.from()).removeOut(e);
    }

    public Stream<FromTo<AbstractNode<N, E>, E>> edges() {
        return nodes().stream().flatMap(AbstractNode::streamOut);
    }

    public void edges(Consumer<FromTo<AbstractNode<N, E>,E>> each) {
        forEachNode(n -> n.edges(false,true).forEach(each));
    }

    @Override
    public String toString() {
        var s = new StringBuilder(1024);
        Consumer println = e -> s.append(e).append('\n');
        s.append("Nodes: ");
        forEachNode(println);

        s.append("Edges: ");
        edges().forEach(println);

        return s.toString();
    }

    /**
     * relinks all edges in 'from' to 'to' before removing 'from'
     */
    public boolean mergeNodes(N from, N to) {
        var fromNode = (MutableNode<N, E>) nodes.get(from);
        if (fromNode != null) {
            var toNode = (MutableNode<N, E>) nodes.get(to);
            if (toNode != null) {
                if (fromNode != toNode) {
                    var e = fromNode.ins() + fromNode.outs();
                    if (e > 0) {
                        List<FromTo> removed = new Lst<>(e);
                        fromNode.edges(true, false).forEach(inEdge -> {
                            removed.add(inEdge);
                            var x = (MutableNode) (inEdge.from());
                            if (x != fromNode)
                                addEdgeByNode(x, inEdge.id(), toNode);
                        });
                        fromNode.edges(false, true).forEach(outEdge -> {
                            removed.add(outEdge);
                            var x = (MutableNode) (outEdge.to());
                            if (x != fromNode)
                                addEdgeByNode(toNode, outEdge.id(), x);
                        });
                        removed.forEach(this::edgeRemove);
                        //assert (fromNode.ins() == 0 && fromNode.outs() == 0);
                    }
                }
                removeNode(from);
                return true;
            }
        }
        return false;
    }

    public final boolean hasNode(N s) {
        return nodes.containsKey(s);
    }

    public final Iterable<N> nodeIDs() {
        return Iterables.transform(nodes(), node -> node.id);
    }

    public final boolean dfs(Object root, Search<N, E> search) {
        return search.dfsRoot(root instanceof Iterable i ? i : List.of(root), this);
    }

    public void rootsEach(Consumer<AbstractNode<N, E>> each) {
        forEachNode(n -> {
            if (n.edgeCount(true, false) == 0)
                each.accept(n);
        });
    }

    public void bfsEach(Consumer<AbstractNode<N, E>> each) {
        rootsEach(n -> bfsEach(n, each));
    }

    /** iterate all nodes, in topologically sorted order */
    public void bfsEach(AbstractNode<N, E> root, Consumer<AbstractNode<N, E>> each) {
        each.accept(root);
        bfs(root.id, new Search<>() {
            @Override
            protected boolean go(List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path, AbstractNode<N, E> next) {
                each.accept(next);
                return true;
            }
        });
    }

    private boolean dfs(Iterable<N> roots, Search<N, E> search) {
        return search.dfsRoot(roots, this);
    }

    public final boolean bfs(Object root, Search<N, E> search) {
        var c = nodeCount();
        return switch (c) {
            case 0, 1 -> true;
            case 2 -> dfs(root, search);
            default -> bfs(root, newQueue(c), search);
        };
    }

    public boolean bfs(Iterable<?> roots, Search<N, E> search) {
        var c = nodeCount();
        return switch (c) {
            case 0, 1 -> true;
            case 2 -> dfs(roots, search);
            default -> bfs(roots, newQueue(c), search);
        };
    }

    public boolean bfs(Iterable<?> roots, Queue<Pair<List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>>, AbstractNode<N, E>>> q, Search<N, E> search) {
        return search.bfs(roots, q, this);
    }

    public boolean bfs(Object root, Queue<Pair<List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>>, AbstractNode<N, E>>> q, Search<N, E> search) {
        return search.bfsRoot(root, q, this);
    }

    public void print() {
        print(System.out);
    }

    private void print(PrintStream out) {
        forEachNode(node -> node.print(out));
    }

    public abstract static class AbstractNode<N, E> {
        private static final AtomicInteger serials = new AtomicInteger(1);
        public final N id;
        public final int serial = serials.getAndIncrement();
        final int hash;

        protected AbstractNode(N id) {
            this.id = id;
            this.hash = id.hashCode();
        }

        public static <X,Y> FromTo<AbstractNode<Y, X>,X> edge(AbstractNode<Y, X> from, X what, AbstractNode<Y, X> to) {
            return new ImmutableDirectedEdge<>(from, what, to);
        }

        private static <N, E> List<FromTo<AbstractNode<N,E>,E>> edgeList(Iterable<FromTo<AbstractNode<N, E>, E>> edgesUnfiltered, @Nullable Predicate<FromTo<AbstractNode<N, E>, E>> filter) {
            var z = new Lst<>(filter == null ?
                edgesUnfiltered :
                Iterables.filter(edgesUnfiltered, filter::test)
            );
            return z.isEmpty() ? emptyIterable : z;
        }

        private static <N, E> boolean empty(Iterable<FromTo<AbstractNode<N, E>, E>> l) {
            return l == emptyIterable ||
                   (l instanceof Collection c && c.isEmpty());
        }

        //        public Stream<N> successors() {
//            return streamOut().map(e -> e.to().id);
//        }
//
//        public Stream<N> predecessors() {
//            return streamIn().map(e -> e.from().id);
//        }

        @Override
        public final boolean equals(Object obj) {
            return this == obj;
        }

        @Override
        public final int hashCode() {
            return hash;
        }

        @Override
        public String toString() {
            return id.toString();
        }

        public abstract Iterable<FromTo<AbstractNode<N,E>, E>> edges(boolean in, boolean out);

        /** warning this buffers the iteration */
        public List<FromTo<AbstractNode<N,E>, E>> edgeList(boolean in, boolean out, @Nullable Predicate<FromTo<AbstractNode<N,E>, E>> filter) {
            var edgesRaw = edges(in, out);
            if (empty(edgesRaw))
                return emptyIterable;
            else
                return edgesRaw instanceof List<FromTo<AbstractNode<N,E>, E>> edgesRawList ?
                    edgeList(edgesRawList, filter) :
                    edgeList(edgesRaw, filter);
        }

        public Iterator<FromTo<AbstractNode<N,E>, E>> edgeIteratorIn() {
            return edgeIterator(true, false);
        }

        public Iterator<FromTo<AbstractNode<N,E>, E>> edgeIteratorOut() {
            return edgeIterator(false, true);
        }

        public Iterator<FromTo<AbstractNode<N,E>, E>> edgeIterator(boolean in, boolean out) {
            return edges(in, out).iterator();
        }

        public Iterator<FromTo<AbstractNode<N,E>, E>> edgeIterator(boolean in, boolean out, @Nullable Predicate<FromTo<AbstractNode<N, E>, E>> filter) {
            var l = edgeIterator(in, out);
            return l == Util.emptyIterator ? l : filter != null ? Iterators.filter(l, filter::test) : l;
        }

        /** TODO Iterator version of this, like edges and edgesIterator */
        public Iterable<? extends AbstractNode<N,E>> nodes(boolean in, boolean out) {
            if (!in && !out)
                return Collections.EMPTY_LIST;

            Function<FromTo<AbstractNode<N,E>, E>, AbstractNode<N,E>> other = x -> x.other(this);
            final com.google.common.base.Function<FromTo<AbstractNode<N,E>, E>, AbstractNode<N,E>> otherApply = other::apply;
            var i = in ? Iterables.transform(edges(true, false), otherApply) : null;
            var o = out ? Iterables.transform(edges(false,true), otherApply) : null;
            if (in && out)
                return ()-> Concaterator.concat(i, o); //HACK
            else if (in)
                return i;
            else //if (out)
                return o;
        }

        public Stream<FromTo<AbstractNode<N,E>, E>> streamIn() {
            return Streams.stream(edges(true, false));
        }

        public Stream<FromTo<AbstractNode<N,E>, E>> streamOut() {
            return Streams.stream(edges(false, true));
        }

        public Stream<FromTo<AbstractNode<N, E>, E>> stream() {
            return Streams.stream(edges(true, true));
        }

        public void print(PrintStream out) {
            out.println(this);
            stream().forEach(e -> out.println("\t" + e));
        }

        public int edgeCount(boolean in, boolean out) {
            return (int) ((in ? streamIn().count() : 0) + (out ? streamOut().count() : 0));
        }
    }
}
