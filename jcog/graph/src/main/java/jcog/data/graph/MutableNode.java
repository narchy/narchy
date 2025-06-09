package jcog.data.graph;

import jcog.data.graph.path.FromTo;
import jcog.data.set.ArrayHashSet;
import jcog.data.set.ArrayUnenforcedSortedSet;
import jcog.data.set.UnenforcedConcatSet;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.stream.Stream;

import static java.util.Collections.EMPTY_LIST;
import static jcog.Util.emptyIterable;
import static jcog.Util.emptyIterator;
import static jcog.data.iterator.Concaterator.concat;

public class MutableNode<N, E> extends MapNodeGraph.AbstractNode<N, E> {

    private Collection<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> in, out;

    public MutableNode(N id) {
        this(id, EMPTY_LIST, EMPTY_LIST);
    }

    public MutableNode(N id, Collection<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> in, Collection<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> out) {
        super(id);
        this.in = in;
        this.out = out;
    }

    @Override
    public int edgeCount(boolean in, boolean out) {
        return (in ? ins() : 0) + (out ? outs() : 0);
    }

    @Override
    public Iterable<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> edges(boolean IN, boolean OUT) {
        var in = this.in;
        var ie = !IN || in.isEmpty();
        var out = this.out;
        var oe = !OUT || out.isEmpty();
        if (ie && oe) return emptyIterable;
        else if (ie) return out;
        else if (oe) return in;
        else
            return new UnenforcedConcatSet<>(out, in);
            //return Iterables.concat(out, in);
    }

    @Override
    public Iterator<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> edgeIterator(boolean IN, boolean OUT) {
        var in = this.in;
        var ie = !IN || in.isEmpty();
        var out = this.out;
        var oe = !OUT || out.isEmpty();
        if (ie && oe) return emptyIterator;
        else if (ie) return out.iterator();
        else if (oe) return in.iterator();
        else return concat(out, in);
    }

    final int ins() {
        return ins(true);
    }

    int ins(boolean countSelfLoops) {
        return countSelfLoops ? in.size() : (int) streamIn().filter(e -> e.from() != this).count();
    }

    int outs() {
        return out.size();
    }

    Collection<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> newEdgeCollection(int cap) {
        return new ArrayHashSet<>(cap);
    }

    @SafeVarargs
    final Collection<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> newEdgeCollection(FromTo<MapNodeGraph.AbstractNode<N, E>, E>... ff) {
        var c = newEdgeCollection(ff.length);
        Collections.addAll(c, ff);
        return c;
    }

    public boolean addIn(FromTo<MapNodeGraph.AbstractNode<N, E>, E> e) {
        return addSet(e, true);
    }

    public boolean addOut(FromTo<MapNodeGraph.AbstractNode<N, E>, E> e) {
        return addSet(e, false);
    }


    public boolean removeIn(FromTo<MapNodeGraph.AbstractNode<N, E>, E> e) {
        return removeSet(e, true);
    }

    public boolean removeOut(FromTo<MapNodeGraph.AbstractNode<N, E>, E> e) {
        return removeSet(e, false);
    }

    public void removeIn(MapNodeGraph.AbstractNode<N, E> src) {
        edgeList(true, false, e -> e.from() == src)
                .forEach(e -> removeSet(e, true));
    }

    public void removeOut(MapNodeGraph.AbstractNode<N, E> target) {
        edgeList(false, true, e -> e.to() == target)
                .forEach(e -> removeSet(e, false));
    }

    private boolean addSet(FromTo<MapNodeGraph.AbstractNode<N, E>, E> e, boolean inOrOut) {
        boolean changed;
        var c = inOrOut ? in : out;
        if (c == EMPTY_LIST) {
            c = ArrayUnenforcedSortedSet.the(e);
            changed = true;
        } else {
            if (c instanceof ArrayUnenforcedSortedSet.One<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> tos) {
                var x = tos.element;
                if (changed = !x.equals(e))
                    c = newEdgeCollection(x, e);
            } else
                changed = c.add(e);
        }

        if (changed) {
            if (inOrOut) in = c;
            else out = c;
        }
        return changed;
    }
    private boolean removeSet(FromTo<MapNodeGraph.AbstractNode<N, E>, E> e, boolean inOrOut) {
        var s = inOrOut ? in : out;
        if (s == EMPTY_LIST)
            return false;

        boolean changed;
        if (s instanceof ArrayUnenforcedSortedSet.One ss) {
            if (changed = (ss.first().equals(e)))
                s = EMPTY_LIST;
        } else {
            if (changed = s.remove(e)) {
                s = switch (s.size()) {
                    case 0 -> throw new UnsupportedOperationException();
                    case 1 -> set1(s);
                    default -> s;
                };
            }
            //TODO downgrade
        }

        if (changed) {
            if (inOrOut) in = s;
            else out = s;
        }

        return changed;
    }

    private static <N, E> Collection<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> set1(Collection<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> s) {
        return ArrayUnenforcedSortedSet.the(switch (s) {
            case ArrayHashSet<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> s0 -> s0.first();
            case ArrayUnenforcedSortedSet.One<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> s1 -> s1.element;
            case null, default -> s.iterator().next();
        });
    }

    @Override
    public Stream<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> streamIn() {
        return in.stream();
    }

    @Override
    public Stream<FromTo<MapNodeGraph.AbstractNode<N, E>, E>> streamOut() {
        return out.stream();
    }

}
