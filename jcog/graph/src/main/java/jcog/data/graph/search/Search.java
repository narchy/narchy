package jcog.data.graph.search;

import jcog.data.graph.MapNodeGraph;
import jcog.data.graph.MapNodeGraph.AbstractNode;
import jcog.data.graph.path.FromTo;
import jcog.data.list.Cons;
import jcog.data.list.Lst;
import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.api.tuple.primitive.BooleanObjectPair;
import org.eclipse.collections.impl.tuple.Tuples;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.List;
import java.util.Queue;

import static java.util.Collections.EMPTY_LIST;
import static jcog.data.graph.search.TraveLog.id;
import static org.eclipse.collections.impl.tuple.primitive.PrimitiveTuples.pair;

/**
 * a search process instance
 * <p>
 * general purpose recursive search for DFS/BFS/A* algorithms
 * backtrack/cyclic prevention guaranteed to visit each vertex at most once.
 * - an instance may be recycled multiple times
 * - multiple instances may concurrently access the same graph
 * <p>
 * NOT multi-thread safe in any way.
 */
public abstract class Search<N, E> {

    public final TraveLog log;


    protected Search() {
        this(
                new TraveLog.IntHashTraveLog(8)
                //new TraveLog.RoaringHashTraveLog()
        );
    }

    private Search(TraveLog log) {
        this.log = log;
    }

    public static <N, E> AbstractNode<N, E> pathStart(List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path, int n) {
        var step = path.get(n);
        return step.getTwo().from(step.getOne());
    }

    public static <N, E> AbstractNode<N, E> pathStart(List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path) {
        return pathStart(path, 0);
    }

    /**
     * optimized for Cons usage
     */
    public static <N, E> AbstractNode<N, E> pathEnd(List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path) {
        var step = path instanceof Cons ?
                ((Cons<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>>) path).tail : path.get(path.size() - 1);
        return step.getTwo().to(step.getOne());
    }

    public static ArrayDeque newQueue(int c) {
        return new ArrayDeque(2 * (int) Math.ceil(Math.log(c)) /* estimate */);
    }

    public void clear() {
        log.clear();
    }

    protected abstract boolean go(List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path, AbstractNode<N, E> next);

    protected boolean visit(AbstractNode<N, E> n) {
        return log.visit(id(n));
    }

    protected boolean visited(AbstractNode<N, E> n) {
        return log.hasVisited(id(n));
    }

    public int visits() {
        return log.size();
    }

    private boolean bfsNode(AbstractNode<N, E> start, Queue<Pair<List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>>, AbstractNode<N, E>>> q) {
        clear();
        visit(start);

        if (q == null)
            q = newQueue(1);
        else
            q.clear();

        q.add(Tuples.pair(EMPTY_LIST, start));

        Pair<List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>>, AbstractNode<N, E>> n;
        while ((n = q.poll()) != null) {

            var at = n.getTwo();

            var path = n.getOne();

            //for (Iterator<FromTo<Node<N, E>, E>> iterator = iterator(at, path); iterator.hasNext(); ) {

            for (var e : search(at, path)) {
                var next = next(e, at, path);
                if (next == null || !visit(next))
                    continue;

                q.add(Tuples.pair(Cons.the(path, hop(e, next)), next));
            }

            if (!path.isEmpty() && !go(path, at))
                return false;
        }

        return true;
    }

    private BooleanObjectPair<FromTo<AbstractNode<N, E>, E>> hop(FromTo<AbstractNode<N, E>, E> e, AbstractNode<N, E> next) {
        return pair(next == e.to(), e);
    }

    /**
     * can be overridden to hijack the determined next destination
     */
    protected AbstractNode<N, E> next(FromTo<AbstractNode<N, E>, E> e, AbstractNode<N, E> at, List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path) {
        return e.other(at);
    }

    public boolean bfs(Object startingNode) {
        return bfs(Collections.singleton(startingNode));
    }

    public boolean bfs(Iterable<?> startingNodes) {
        return bfs(startingNodes, null, null);
    }

    private boolean dfsNode(AbstractNode<N, E> at, List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path) {

        if (visit(at)) {
            for (var e : search(at, path)) {

                var next = next(e, at, path);

                if (next == null || next == at || visited(next))
                    continue;

                path.add(hop(e, next));

                if (!go(path, next) || !dfsNode(next, path))
                    return false;

                ((Lst) path).removeLastFast();
            }
        }

        return true;
    }

    //    @Deprecated protected Iterable<FromTo<Node<N, E>, E>> find(Node<N, E> n, List<BooleanObjectPair<FromTo<Node<N, E>, E>>> path) {
//        return n.edges(true, true);
//    }
    protected Iterable<FromTo<AbstractNode<N, E>, E>> search(AbstractNode<N, E> n, List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>> path) {
        return n.edges(true, true);
    }

    /**
     * q is recycleable between executions automatically. just provide a pre-allocated ArrayDeque or similar.
     */
    public boolean bfs(Iterable<?> startingNodes, Queue<Pair<List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>>, AbstractNode<N, E>>> q, MapNodeGraph<N, E> g) {
        for (var n : startingNodes)
            if (!bfsRoot(n, q, g))
                return false;

        return true;
    }

    /**
     * assumes starting nodes are unique
     */
    public boolean dfsRoot(Iterable<?> startingNodes, MapNodeGraph<N, E> g) {

        for (var n : startingNodes) {
            AbstractNode<N, E> nn;
            if (n instanceof AbstractNode nnn)
                nn = nnn;
            else {
                nn = g.node(n);
                if (nn == null)
                    continue; //throw new NullPointerException("unknown node: " + n);
            }

            clear();
            if (!dfsNode(nn, new Lst<>()))
                return false;
        }

        return true;
    }

    public boolean bfsRoot(Object start, @Nullable Queue<Pair<List<BooleanObjectPair<FromTo<AbstractNode<N, E>, E>>>, AbstractNode<N, E>>> q, MapNodeGraph<N, E> g) {
        AbstractNode<N, E> nn;
        if (start instanceof AbstractNode startNode)
            nn = startNode;
        else {
            nn = g.node(start);
            if (nn == null)
                return true; //throw new NullPointerException("unknown node: " + start);
        }

        return bfsNode(nn, q);
    }
}