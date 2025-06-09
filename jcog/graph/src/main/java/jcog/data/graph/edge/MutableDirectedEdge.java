package jcog.data.graph.edge;

import jcog.data.graph.AbstractMutableDirectedEdge;
import jcog.data.graph.MapNodeGraph;
import jcog.data.graph.path.FromTo;
import org.jetbrains.annotations.Nullable;

public class MutableDirectedEdge<N,E> extends AbstractMutableDirectedEdge<N, E> {
    private int hash;

    MutableDirectedEdge(MapNodeGraph.AbstractNode<N, E> from, @Nullable E id, MapNodeGraph.AbstractNode<N, E> to) {
        super(from, to, id);
    }

    @Override
    protected void rehash() {
        this.hash = FromTo.hash(from, id, to);
    }

    @Override
    public int hashCode() {
        return hash;
    }

    @Override
    protected boolean hashDynamic() {
        return false;
    }

}
