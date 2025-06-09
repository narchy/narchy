package jcog.data.graph.edge;

import jcog.data.graph.MapNodeGraph;
import jcog.data.graph.path.FromTo;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;

/**
 * immutable directed edge with cached hashcode
 */
public class ImmutableDirectedEdge<N, E> implements FromTo<MapNodeGraph.AbstractNode<N, E>, E> {

    public final MapNodeGraph.AbstractNode<N, E> from;
    public final MapNodeGraph.AbstractNode<N, E> to;
    private final @Nullable E id;
    private final int hash;

    public ImmutableDirectedEdge(MapNodeGraph.AbstractNode<N, E> from, @Nullable E id, MapNodeGraph.AbstractNode<N, E> to) {
        this.hash = FromTo.hash(from, id, to);
        this.id = id;
        this.from = from;
        this.to = to;
    }

    @Override
    public FromTo<MapNodeGraph.AbstractNode<N, E>, E> reverse() {
        return new ImmutableDirectedEdge<>(to, id, from);
    }

    @Override
    public final MapNodeGraph.AbstractNode<N, E> from() {
        return from;
    }

    @Override
    public final E id() {
        return id;
    }

    @Override
    public final MapNodeGraph.AbstractNode<N, E> to() {
        return to;
    }

    @Override
    public int hashCode() {
        return hash;
    }

    @Override
    public boolean equals(Object x) {
        if (this == x) return true;
        if (!(x instanceof FromTo ee) || hash != x.hashCode()) return false;
        return from.equals(ee.from()) && to.equals(ee.to()) && Objects.equals(id, ee.id());
    }

    public boolean isSelfLoop() {
        return from.equals(to);
    }

    @Override
    public String toString() {
        return from + " => " + id + " => " + to;
    }


}