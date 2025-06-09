package jcog.data.graph;

import jcog.data.graph.path.FromTo;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;

public abstract class AbstractMutableDirectedEdge<N, E> implements FromTo<MapNodeGraph.AbstractNode<N, E>, E> {
    protected MapNodeGraph.AbstractNode<N, E> from;
    protected MapNodeGraph.AbstractNode<N, E> to;
    protected E id;

    protected AbstractMutableDirectedEdge() {

    }

    protected AbstractMutableDirectedEdge(MapNodeGraph.AbstractNode<N, E> from, MapNodeGraph.AbstractNode<N, E> to, @Nullable E id) {
        set(from, to, id);
    }

    public void from(MapNodeGraph.AbstractNode<N, E> from) {
        if (!this.from.equals(from)) {
            this.from = from;
            rehash();
        }
    }

    public void to(MapNodeGraph.AbstractNode<N, E> to) {
        if (!this.to.equals(to)) {
            this.to = to;
            rehash();
        }
    }

    public void id(E id) {
        if (!Objects.equals(this.id, id)) {
            this.id = id;
            rehash();
        }
    }

    protected abstract void rehash();

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
    public final boolean equals(Object x) {
        return this == x
                   ||  (
               (x instanceof FromTo e)
                   &&
               (hashDynamic() || (hashCode() == x.hashCode()))
                   &&
               from.equals(e.from())
                   &&
               to.equals(e.to())
                   &&
               Objects.equals(id, e.id())
        );
    }

    /** whether hash is computed dynamically (true) or is cached (false) */
    protected abstract boolean hashDynamic();

    @Override
    public abstract int hashCode();

    @Override
    public String toString() {
        return from + " => " + id + " => " + to;
    }

    public void set(MapNodeGraph.AbstractNode<N, E> from, MapNodeGraph.AbstractNode<N, E> to, E id) {
        this.from = from;
        this.to = to;
        this.id = id;
        rehash();
    }
}