/*
 * Copyright (c) 2003-2005 The BISON Project
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 */

package jcog.data.graph;

import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import jcog.data.list.Lst;
import jcog.data.map.ObjIntHashMap;
import jcog.func.TriConsumer;
import jcog.func.TriFunction;
import org.eclipse.collections.api.block.function.primitive.FloatFunction;
import org.eclipse.collections.api.factory.primitive.IntSets;
import org.eclipse.collections.api.set.primitive.ImmutableIntSet;
import org.eclipse.collections.api.set.primitive.IntSet;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.LongObjectHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectFloatHashMap;
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;
import org.jetbrains.annotations.Nullable;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import java.util.stream.StreamSupport;

/**
 * Implements a graph which uses the neighbour list representation.
 * No multiple edges are allowed. The implementation also supports the
 * growing of the graph. This is very useful when the number of nodes is
 * not known in advance or when we construct a graph reading a file.
 */
public class AdjGraph<V, E> extends IntIndexedGraph<V, E> implements Serializable {

	/**
	 * Contains the indices of the nodes. The vector "nodes" contains this
	 * information implicitly but this way we can find indexes in log time at
	 * the cost of memory (node that the edge lists typically use much more memory
	 * than this anyway). Note that the nodes vector is still necessary to
	 * provide constant access to nodes based on indexes.
	 */
	private final ObjIntHashMap<Node<V>> nodes;
	private final IntObjectHashMap<Node<V>> antinodes;
	/**
	 * edge values indexed by their unique 64-bit id formed from the id's of the src (32bit int) and target (32bit int)
	 */
	private final LongObjectHashMap<E> edges;

	/**
	 * Indicates if the graph is directed.
	 */
	private final boolean directed;
	private int serial;

	/**
	 * Constructs an empty graph. That is, the graph has zero nodes, but any
	 * number of nodes and edges can be added later.
	 *
	 * @param directed if true the graph will be directed
	 */
	public AdjGraph(boolean directed) {
		this(directed, 0, 0);
	}


	private AdjGraph(boolean directed, int nodeCap, int edgeCap) {
		this.directed = directed;
		edges = new LongObjectHashMap<>(edgeCap);
		nodes = new ObjIntHashMap<>(nodeCap);
		antinodes = new IntObjectHashMap<>(nodeCap);
	}

	public void clear() {
		edges.clear(); nodes.clear(); antinodes.clear();
	}

	public V node(int i) {
		return antinodes.get(i).v;
	}

//	public boolean addIfNew(V s) {
//		if (this.node(s) == -1) {
//			_addNode(s);
//			return true;
//		}
//		return false;
//	}


	public int node(V node) {
		return nodes.getIfAbsent(node, -1);
	}

	/**
	 * If the given object is not associated with a node yet, adds a new
	 * node. Returns the index of the node. If the graph was constructed to have
	 * a specific size, it is not possible to add nodes and therefore calling
	 * this method will throw an exception.
	 *
	 * @throws NullPointerException if the size was specified at construction time.
	 */
	public int addNode(V o) {
		int index = node(o);
		return index == -1 ? _addNode(o) : index;
	}

	private int _addNode(V o) {
		int id = serial++;
		var n = new Node<>(o, id);
		nodes.put(n, id);
		antinodes.put(id, n);
		return id;
	}

	public boolean setEdge(V i, V j, E value) {
		int ii = node(i);
		if (ii != -1) {
			int jj = node(j);
			if (jj != -1)
				return setEdge(ii, jj, value);
		}
		return false;
	}

	private boolean setEdge(int i, int j, E value) {

		assert(i!=j);

		if (!directed) {
			if (i > j) {
				int ij = i;
				i = j;
				j = ij;
			}
		}

		if (edges.put(eid(i, j), value) == null) {
			boolean ir = antinodes.get(i).eAdd(j); assert(ir);
			if (!directed) {
				boolean jr = antinodes.get(j).eAdd(i); assert(jr);
			}
			return true;
		}
		return false;
	}


	/**
	 * Returns null always.
	 */
	@Override
	public @Nullable E edge(int i, int j) {
		return edges.get(eid(i, j));
	}

	public @Nullable E edge(V i, V j) {
		int ii = node(i);
		if (ii != -1) {
			int jj = node(j);
			if (jj != -1)
				return edge(ii, jj);
		}
		return null;
	}

	@Override
	public String toString() {
		return nodes + " * " + Joiner.on(",").join(StreamSupport.stream(edges.keyValuesView().spliterator(), false).map((e) -> {
			long id = e.getOne();
			return (id >> 32) + "->" + (id & 0xffffffffL) + '=' + e.getTwo();
		}).toList());
	}

	@Override
	public boolean removeEdge(int i, int j) {
		boolean ret = antinodes.get(i).eRemove(j);
		if (ret && !directed) {
			boolean ret2 = antinodes.get(j).eRemove(i);
			assert (ret2);
		}
		boolean ret3 = edges.remove(eid(i, j)) != null;
		assert (ret3);
		return ret;
	}

	public final long eid(int i, int j) {
		if (!directed) {
			if (j < i) {
				int x = j;
				j = i;
				i = x;
			}
		}
		long x = i;
		x <<= 32;
		x |= j;
		return x;
	}

	@Override
	public boolean isEdge(int r, int c) {
		return antinodes.get(r).e().contains(c);
	}

	@Override
	public IntSet neighborsOut(int r) {
		return antinodes.get(r).e();
	}

	public void neighborEdges(V v, BiConsumer<V, E> each) {
		int i = node(v);
		if (i >= 0)
			neighborEdges(antinodes.get(i), each);
	}

    private void neighborEdges(Node<V> n, BiConsumer<V, E> each) {
	    int i = n.id;
        n.e().forEach(ee -> each.accept(antinodes.get(ee).v, edge(i, ee)));
    }

    public void neighborEdges(V v, BiFunction<V, E, E> each) {
		int i = node(v);
		if (i < 0) return;
		antinodes.get(i).e().forEach(ee -> {
			E x = edge(i, ee);
			Node<V> nohd = antinodes.get(ee);
			E y = each.apply(nohd.v, x);
			if (y != x)
				setEdge(i, ee, y);
		});
	}

	/**
	 * If the graph was gradually grown using {@link #addNode}, returns the
	 * object associated with the node, otherwise null
	 */
	@Override
	public V vertex(int v) {
		return antinodes.get(v).v;
	}

	@Override
	public int size() {
		return nodes.size();
	}

	@Override
	public boolean directed() {
		return directed;
	}

	@Override
	public int degreeOut(int i) {
		return antinodes.get(i).e().size();
	}

	public int nodeCount() {
		return nodes.size();
	}

	public int edgeCount() {
		return edges.size();
	}

	public E edgeIfMissing(V s, V p, E ifMissing) {
		@Nullable E existing = edge(s, p);
		return existing!=null ? existing : ifMissing;
	}


	/** returns true if the edge did not exist */
	public boolean setEdgeIfMissing(V s, V p, E e) {
		@Nullable E existing = edge(s, p);
		if (existing != null) return false;

		setEdge(s, p, e);
		return true;
	}

	private E setEdgeIfMissing(int s, int p, Supplier<E> ifMissing) {
		@Nullable E existing = edge(s, p);
		if (existing != null) return existing;

		E ee = ifMissing.get();
		setEdge(s, p, ee);
		return ee;
	}

	public boolean removeEdge(V i, V j) {
		int ii = node(i);
		if (ii != -1) {
			int jj = node(j);
			if (jj != -1)
				return removeEdge(ii, jj);
		}
		return false;
	}

	private void eachNode(TriConsumer<Node<V>, Node<V>, E> edge) {
		edges.forEachKeyValue((eid, ev) -> {
            Node<V> na = antinodes.get((int) (eid >> 32)),
			        nb = antinodes.get((int) (eid));
			if (na == null || nb == null) throw new NullPointerException();
			edge.accept(na, nb, ev);
		});
	}

	/**
	 * returns true if modified
	 */
	private void each(TriConsumer<V, V, E> edge) {
		eachNode((an, bn, e) -> edge.accept(an.v, bn.v, e));
	}

	public AdjGraph<V, E> compact() {
		return compact((x, y, z) -> z);
	}

	private AdjGraph<V, E> compact(TriFunction<V, V, E, E> retain) {
		AdjGraph g = new AdjGraph(directed);
		each((a, b, e) -> {
			E e1 = retain.apply(a, b, e);
			if (e1 != null)
				g.setEdge(g.addNode(a), g.addNode(b), e1);
		});
		return g;
	}

	/**
	 * Saves the given graph to
	 * the given stream in GML format.
	 */
	public void writeGML(PrintStream out) {

		out.println("graph [ directed " + (directed ? "1" : "0"));

		for (Node<V> k : nodes.keySet())
			out.println("node [ id " + k.id + " label \"" +
				k.v.toString().replace('\"', '\'') + "\" ]");

		eachNode((a, b, e) -> out.println(
			"edge [ source " + a.id + " target " + b.id + " label \"" +
				e.toString().replace('\"', '\'') + "\" ]"));

		out.println("]");
	}

    public Iterable<V> nodes() {
        return Iterables.transform(nodes.keysView(), x->x.v);
    }


//	public boolean addEdge(V s, V t, E x) {
//		int S = addNode(s);
//		int T = addNode(t);
//		return setEdge(S, T, x);
//	}



	/**
	 * Contains sets of node indexes. If "nodes" is not null, indices are
	 * defined by "nodes", otherwise they correspond to 0,1,...
	 */
	public final static class Node<V> {
		public final V v;
		public final int id;

		/** lazily allocated */
		@Nullable private IntHashSet e;

		Node(V v, int id) {
			super();
			this.v = v;
			this.id = id;
			this.e = null;
		}

		@Override
		public String toString() {
			return v.toString();
		}

		@Override
		public boolean equals(Object obj) {
			return this == obj ||
				(!(obj instanceof Node /* assumes nodes dont contain nodes as values */) && v.equals(obj)) ||
				(obj instanceof Node n && v.equals(n.v));
		}

		@Override
		public int hashCode() {
			return v.hashCode();
		}

		private static final ImmutableIntSet EMPTY = IntSets.immutable.empty();

		public IntSet e() {
			return e == null ? EMPTY : e;
		}

		public boolean eAdd(int i) {
			if (e == null) this.e = new IntHashSet(1);
			return e.add(i);
		}

		public boolean eRemove(int i) {
            return e != null && e.remove(i);
		}
	}

	public Map<Node<V>, Double> betweennessCentrality() {
		Map<Node<V>, Double> centrality = new HashMap<>();
		for (Node<V> s : antinodes.values()) {
			Map<Node<V>, List<Node<V>>> pred = new HashMap<>();
			Map<Node<V>, Integer> dist = new HashMap<>();
			Map<Node<V>, Double> sigma = new HashMap<>();
			Map<Node<V>, Double> delta = new HashMap<>();
			for (Node<V> w : antinodes.values()) {
				pred.put(w, new Lst<>());
				dist.put(w, -1);
				sigma.put(w, 0.0);
				delta.put(w, 0.0);
			}
			sigma.put(s, 1.0);
			dist.put(s, 0);
			List<Node<V>> queue = new Lst<>();
			List<Node<V>> stack = new Lst<>();
			queue.add(s);

			// Breadth-first search
			while (!queue.isEmpty()) {
				Node<V> v = queue.removeFirst();
				stack.add(v);
				v.e().forEach(w_id -> {
					Node<V> w = antinodes.get(w_id);
					// Path discovery
					if (dist.get(w) < 0) {
						dist.put(w, dist.get(v) + 1);
						queue.add(w);
					}
					// Path counting
					if (dist.get(w) == dist.get(v) + 1) {
						sigma.put(w, sigma.get(w) + sigma.get(v));
						pred.get(w).add(v);
					}
				});
			}

			// Accumulation
			while (!stack.isEmpty()) {
				Node<V> w = stack.removeLast();
				for (Node<V> v : pred.get(w)) {
					double c = (sigma.get(v) / sigma.get(w)) * (1 + delta.get(w));
					delta.put(v, delta.get(v) + c);
				}
				if (w != s) {
					centrality.put(w, centrality.getOrDefault(w, 0.0) + delta.get(w));
				}
			}
		}

		// Normalization (optional)
		int c = antinodes.size();
		double factor = 1.0 / ((c - 1) * (c - 2));
		centrality.replaceAll((n, val) -> val * factor);

		return centrality;
	}

	public ObjectFloatHashMap<V> outDegreeCentrality(FloatFunction<E> edgeWeight) {
		var outDegreeCentrality = new ObjectFloatHashMap<V>(size());
		antinodes.forEachKeyValue((id, node) -> {
			final float[] sum = {0};
			node.e().forEach(neighborId -> {
                sum[0] += edgeWeight.floatValueOf(edge(node.v, antinodes.get(neighborId).v));
			});
			outDegreeCentrality.put(node.v, sum[0]);
		});
		return outDegreeCentrality;
	}

	public Map<Node<V>, Float> inDegreeCentrality(FloatFunction<E> edgeWeight) {
		Map<Node<V>, Float> inDegreeCentrality = new HashMap<>();
		// Initialize in-degree centrality
		antinodes.forEachKeyValue((id, node) -> inDegreeCentrality.put(node, 0.0f));

		// Calculate weighted in-degrees by iterating over all nodes' edges
		for (Node<V> node : antinodes.values()) {
			node.e().forEach(neighborId -> {
				Node<V> neighbor = antinodes.get(neighborId);
				E edgeData = edge(neighbor.v, node.v);
				inDegreeCentrality.put(neighbor, inDegreeCentrality.get(neighbor) + edgeWeight.floatValueOf(edgeData));
			});
		}
		return inDegreeCentrality;
	}

}




