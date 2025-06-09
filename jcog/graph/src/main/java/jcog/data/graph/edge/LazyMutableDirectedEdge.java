package jcog.data.graph.edge;

import jcog.data.graph.AbstractMutableDirectedEdge;
import jcog.data.graph.MapNodeGraph;
import jcog.data.graph.path.FromTo;
import org.jetbrains.annotations.Nullable;

public class LazyMutableDirectedEdge<N,E> extends AbstractMutableDirectedEdge<N, E> {

    public LazyMutableDirectedEdge() {
        super();
    }
    public LazyMutableDirectedEdge(MapNodeGraph.AbstractNode<N, E> from, @Nullable E id, MapNodeGraph.AbstractNode<N, E> to) {
        super(from, to, id);
    }

    @Override
    protected void rehash() {
    }

    @Override
    protected boolean hashDynamic() {
        return true;
    }

    @Override
    public int hashCode() {
        return FromTo.hash(from, id, to);
    }
}
