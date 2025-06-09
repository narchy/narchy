/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

/*
 * This file is available under and governed by the GNU General Public
 * License version 2 only, as published by the Free Software Foundation.
 * However, the following notice accompanied the original version of this
 * file:
 *
 * Written by Doug Lea with assistance from members of JCP JSR-166
 * Expert Group and released to the public domain, as explained at
 * http://creativecommons.org/publicdomain/zero/1.0/
 */

package jcog.data.map;

import jcog.data.list.Lst;
import org.jetbrains.annotations.Nullable;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.*;
import java.util.concurrent.ConcurrentNavigableMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.*;

/**
 * A scalable concurrent {@link ConcurrentNavigableMap} implementation.
 * The map is sorted according to the {@linkplain Comparable natural
 * ordering} of its keys, or by a {@link Comparator} provided at map
 * creation time, depending on which constructor is used.
 *
 * <p>This class implements a concurrent variant of <a
 * href="http://en.wikipedia.org/wiki/Skip_list" target="_top">SkipLists</a>
 * providing expected average <i>log(n)</i> time cost for the
 * {@code containsKey}, {@code get}, {@code put} and
 * {@code remove} operations and their variants.  Insertion, removal,
 * update, and access operations safely execute concurrently by
 * multiple threads.
 *
 * <p>Iterators and spliterators are
 * <a href="package-summary.html#Weakly"><i>weakly consistent</i></a>.
 *
 * <p>Ascending key ordered views and their iterators are faster than
 * descending ones.
 *
 * <p>All {@code Map.Entry} pairs returned by methods in this class
 * and its views represent snapshots of mappings at the time they were
 * produced. They do <em>not</em> support the {@code Entry.setValue}
 * method. (Note however that it is possible to change mappings in the
 * associated map using {@code put}, {@code putIfAbsent}, or
 * {@code replace}, depending on exactly which effect you need.)
 *
 * <p>Beware that bulk operations {@code putAll}, {@code equals},
 * {@code toArray}, {@code containsValue}, and {@code clear} are
 * <em>not</em> guaranteed to be performed atomically. For example, an
 * iterator operating concurrently with a {@code putAll} operation
 * might view only some of the added elements.
 *
 * <p>This class and its views and iterators implement all of the
 * <em>optional</em> methods of the {@link Map} and {@link Iterator}
 * interfaces. Like most other concurrent collections, this class does
 * <em>not</em> permit the use of {@code null} keys or values because some
 * null return values cannot be reliably distinguished from the absence of
 * elements.
 *
 * <p>This class is a member of the
 * <a href="{@docRoot}/java.base/java/util/package-summary.html#CollectionsFramework">
 * Java Collections Framework</a>.
 *
 * @param <K> the type of keys maintained by this map
 * @param <V> the type of mapped values
 * @author Doug Lea
 * @since 1.6
 */
public abstract class ConcurrentSkipListMap2<K, V> extends AbstractMap<K, V>
        implements ConcurrentNavigableMap<K, V> {
    /*
     * This class implements a tree-like two-dimensionally linked skip
     * list in which the index levels are represented in separate
     * nodes from the base nodes holding data.  There are two reasons
     * for taking this approach instead of the usual array-based
     * structure: 1) Array based implementations seem to encounter
     * more complexity and overhead 2) We can use cheaper algorithms
     * for the heavily-traversed index lists than can be used for the
     * base lists.  Here's a picture of some of the basics for a
     * possible list with 2 levels of index:
     *
     * Head nodes          Index nodes
     * +-+    right        +-+                      +-+
     * |2|---------------->| |--------------------->| |->null
     * +-+                 +-+                      +-+
     *  | down              |                        |
     *  v                   v                        v
     * +-+            +-+  +-+       +-+            +-+       +-+
     * |1|----------->| |->| |------>| |----------->| |------>| |->null
     * +-+            +-+  +-+       +-+            +-+       +-+
     *  v              |    |         |              |         |
     * Nodes  next     v    v         v              v         v
     * +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+
     * | |->|A|->|B|->|C|->|D|->|E|->|F|->|G|->|H|->|I|->|J|->|K|->null
     * +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+  +-+
     *
     * The base lists use a variant of the HM linked ordered set
     * algorithm. See Tim Harris, "A pragmatic implementation of
     * non-blocking linked lists"
     * http://www.cl.cam.ac.uk/~tlh20/publications.html and Maged
     * Michael "High Performance Dynamic Lock-Free Hash Tables and
     * List-Based Sets"
     * http://www.research.ibm.com/people/m/michael/pubs.htm.  The
     * basic idea in these lists is to mark the "next" pointers of
     * deleted nodes when deleting to avoid conflicts with concurrent
     * insertions, and when traversing to keep track of triples
     * (predecessor, node, successor) in order to detect when and how
     * to unlink these deleted nodes.
     *
     * Rather than using mark-bits to mark list deletions (which can
     * be slow and space-intensive using AtomicMarkedReference), nodes
     * use direct CAS'able next pointers.  On deletion, instead of
     * marking a pointer, they splice in another node that can be
     * thought of as standing for a marked pointer (see method
     * unlinkNode).  Using plain nodes acts roughly like "boxed"
     * implementations of marked pointers, but uses new nodes only
     * when nodes are deleted, not for every link.  This requires less
     * space and supports faster traversal. Even if marked references
     * were better supported by JVMs, traversal using this technique
     * might still be faster because any search need only read ahead
     * one more node than otherwise required (to check for trailing
     * marker) rather than unmasking mark bits or whatever on each
     * read.
     *
     * This approach maintains the essential property needed in the HM
     * algorithm of changing the next-pointer of a deleted node so
     * that any other CAS of it will fail, but implements the idea by
     * changing the pointer to point to a different node (with
     * otherwise illegal null fields), not by marking it.  While it
     * would be possible to further squeeze space by defining marker
     * nodes not to have key/value fields, it isn't worth the extra
     * type-testing overhead.  The deletion markers are rarely
     * encountered during traversal, are easily detected via null
     * checks that are needed anyway, and are normally quickly garbage
     * collected. (Note that this technique would not work well in
     * systems without garbage collection.)
     *
     * In addition to using deletion markers, the lists also use
     * nullness of value fields to indicate deletion, in a style
     * similar to typical lazy-deletion schemes.  If a node's value is
     * null, then it is considered logically deleted and ignored even
     * though it is still reachable.
     *
     * Here's the sequence of events for a deletion of node n with
     * predecessor b and successor f, initially:
     *
     *        +------+       +------+      +------+
     *   ...  |   b  |------>|   n  |----->|   f  | ...
     *        +------+       +------+      +------+
     *
     * 1. CAS n's value field from non-null to null.
     *    Traversals encountering a node with null value ignore it.
     *    However, ongoing insertions and deletions might still modify
     *    n's next pointer.
     *
     * 2. CAS n's next pointer to point to a new marker node.
     *    From this point on, no other nodes can be appended to n.
     *    which avoids deletion errors in CAS-based linked lists.
     *
     *        +------+       +------+      +------+       +------+
     *   ...  |   b  |------>|   n  |----->|marker|------>|   f  | ...
     *        +------+       +------+      +------+       +------+
     *
     * 3. CAS b's next pointer over both n and its marker.
     *    From this point on, no new traversals will encounter n,
     *    and it can eventually be GCed.
     *        +------+                                    +------+
     *   ...  |   b  |----------------------------------->|   f  | ...
     *        +------+                                    +------+
     *
     * A failure at step 1 leads to simple retry due to a lost race
     * with another operation. Steps 2-3 can fail because some other
     * thread noticed during a traversal a node with null value and
     * helped out by marking and/or unlinking.  This helping-out
     * ensures that no thread can become stuck waiting for progress of
     * the deleting thread.
     *
     * Skip lists add indexing to this scheme, so that the base-level
     * traversals start close to the locations being found, inserted
     * or deleted -- usually base level traversals only traverse a few
     * nodes. This doesn't change the basic algorithm except for the
     * need to make sure base traversals start at predecessors (here,
     * b) that are not (structurally) deleted, otherwise retrying
     * after processing the deletion.
     *
     * Index levels are maintained using CAS to link and unlink
     * successors ("right" fields).  Races are allowed in index-list
     * operations that can (rarely) fail to link in a new index node.
     * (We can't do this of course for data nodes.)  However, even
     * when this happens, the index lists correctly guide search.
     * This can impact performance, but since skip lists are
     * probabilistic anyway, the net result is that under contention,
     * the effective "p" value may be lower than its nominal value.
     *
     * Index insertion and deletion sometimes require a separate
     * traversal pass occurring after the base-level action, to add or
     * remove index nodes.  This adds to single-threaded overhead, but
     * improves contended multithreaded performance by narrowing
     * interference windows, and allows deletion to ensure that all
     * index nodes will be made unreachable upon return from a public
     * remove operation, thus avoiding unwanted garbage retention.
     *
     * Indexing uses skip list parameters that maintain good search
     * performance while using sparser-than-usual indices: The
     * hardwired parameters k=1, p=0.5 (see method doPut) mean that
     * about one-quarter of the nodes have indices. Of those that do,
     * half have one level, a quarter have two, and so on (see Pugh's
     * Skip List Cookbook, sec 3.4), up to a maximum of 62 levels
     * (appropriate for up to 2^63 elements).  The expected total
     * space requirement for a map is slightly less than for the
     * current implementation of java.util.TreeMap.
     *
     * Changing the level of the index (i.e, the height of the
     * tree-like structure) also uses CAS.  Creation of an index with
     * height greater than the current level adds a level to the head
     * index by CAS'ing on a new top-most head. To maintain good
     * performance after a lot of removals, deletion methods
     * heuristically try to reduce the height if the topmost levels
     * appear to be empty.  This may encounter races in which it is
     * possible (but rare) to reduce and "lose" a level just as it is
     * about to contain an index (that will then never be
     * encountered). This does no structural harm, and in practice
     * appears to be a better option than allowing unrestrained growth
     * of levels.
     *
     * This class provides concurrent-reader-style memory consistency,
     * ensuring that read-only methods report status and/or values no
     * staler than those holding at method entry. This is done by
     * performing all publication and structural updates using
     * (volatile) CAS, placing an acquireFence in a few access
     * methods, and ensuring that linked objects are transitively
     * acquired via dependent reads (normally once) unless performing
     * a volatile-mode CAS operation (that also acts as an acquire and
     * release).  This form of fence-hoisting is similar to RCU and
     * related techniques (see McKenney's online book
     * https://www.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html)
     * It minimizes overhead that may otherwise occur when using so
     * many volatile-mode reads. Using explicit acquireFences is
     * logistically easier than targeting particular fields to be read
     * in acquire mode: fences are just hoisted up as far as possible,
     * to the entry points or loop headers of a few methods. A
     * potential disadvantage is that these few remaining fences are
     * not easily optimized away by compilers under exclusively
     * single-thread use.  It requires some care to avoid volatile
     * mode reads of other fields. (Note that the memory semantics of
     * a reference dependently read in plain mode exactly once are
     * equivalent to those for atomic opaque mode.)  Iterators and
     * other traversals encounter each node and value exactly once.
     * Other operations locate an element (or position to insert an
     * element) via a sequence of dereferences. This search is broken
     * into two parts. Method findPredecessor (and its specialized
     * embeddings) searches index nodes only, returning a base-level
     * predecessor of the key. Callers carry out the base-level
     * search, restarting if encountering a marker preventing link
     * modification.  In some cases, it is possible to encounter a
     * node multiple times while descending levels. For mutative
     * operations, the reported value is validated using CAS (else
     * retrying), preserving linearizability with respect to each
     * other. Others may return any (non-null) value holding in the
     * course of the method call.  (Search-based methods also include
     * some useless-looking explicit null checks designed to allow
     * more fields to be nulled out upon removal, to reduce floating
     * garbage, but which is not currently done, pending discovery of
     * a way to do this with less impact on other operations.)
     *
     * To produce random values without interference across threads,
     * we use within-JDK thread local random support (via the
     * "secondary seed", to avoid interference with user-level
     * ThreadLocalRandom.)
     *
     * For explanation of algorithms sharing at least a couple of
     * features with this one, see Mikhail Fomitchev's thesis
     * (http://www.cs.yorku.ca/~mikhail/), Keir Fraser's thesis
     * (http://www.cl.cam.ac.uk/users/kaf24/), and Hakan Sundell's
     * thesis (http://www.cs.chalmers.se/~phs/).
     *
     * Notation guide for local variables
     * Node:         b, n, f, p for  predecessor, node, successor, aux
     * Index:        q, r, d    for index node, right, down.
     * Head:         h
     * Keys:         k, key
     * Values:       v, value
     * Comparisons:  c
     */

//    /**
//     * The comparator used to maintain order in this map, or null if
//     * using natural ordering.
//     */
//    private final Comparator<K> comparator;

    /**
     * Lazily initialized topmost index of the skiplist.
     */
    private transient Index<K, V> head;
    /**
     * Lazily initialized element count
     */
    private transient LongAdder adder;
    /**
     * Lazily initialized key set
     */
    private transient KeySet<K, V> keySet;
    /**
     * Lazily initialized values collection
     */
    private transient Values<K, V> values;
    /**
     * Lazily initialized entry set
     */
    private transient EntrySet<K, V> entrySet;
    /**
     * Lazily initialized descending map
     */
    private transient SubMap<K, V> descendingMap;

    /**
     * Nodes hold keys and values, and are singly linked in sorted
     * order, possibly with some intervening marker nodes. The list is
     * headed by a header node accessible as head.node. Headers and
     * marker nodes have null keys. The val field (but currently not
     * the key field) is nulled out upon deletion.
     */
    static final class Node<K, V> {
        final K key; // currently, never detached
        @SuppressWarnings("CanBeFinal")
        V val;
        Node<K, V> next;

        Node(K key, V value, Node<K, V> next) {
            this.key = key;
            this.val = value;
            this.next = next;
        }

        Node(Node<K, V> next) {
            this(null, null, next);
        }
    }

    /**
     * Index nodes represent the levels of the skip list.
     */
    static final class Index<K, V> {
        final Node<K, V> node;  // currently, never detached
        final Index<K, V> down;
        Index<K, V> right;

        Index(Node<K, V> node, Index<K, V> down) {
            this.node = node;
            this.down = down;
        }

        Index(Node<K, V> node, Index<K, V> down, Index<K, V> right) {
            this(node, down);
            this.right = right;
        }
    }

    /**
     * Returns the header for base node list, or null if uninitialized
     */
    private Node<K, V> baseHead() {
        VarHandle.acquireFence();
        Index<K, V> h;
        return (h = head) == null ? null : h.node;
    }

    /**
     * Tries to unlink deleted node n from predecessor b (if both
     * exist), by first splicing in a marker if not already present.
     * Upon return, node n is sure to be unlinked from b, possibly
     * via the actions of some other thread.
     *
     * @param b if nonnull, predecessor
     * @param n if nonnull, node known to be deleted
     */
    private static <K, V> void unlinkNode(Node<K, V> b, Node<K, V> n) {
        if (b != null && n != null)
            NEXT.compareAndSet(b, n, unlinkTarget(n));
    }

    private static <K, V> @Nullable Node<K, V> unlinkTarget(Node<K, V> n) {
        Node<K, V> p;
        for (; ; ) {
            Node<K, V> f;
            if ((f = n.next) != null && f.key == null) {
                p = f.next;               // already marked
                break;
            } else if (NEXT.compareAndSet(n, f, new Node<>(f))) {
                p = f;                    // add marker
                break;
            }
        }
        return p;
    }

    /**
     * Adds to element count, initializing adder if necessary
     *
     * @param c count to add
     */
    private void addCount(int c) {
        LongAdder a;
        do {
        } while ((a = adder) == null &&
                !ADDER.compareAndSet(this, null, a = new LongAdder()));
        a.add(c);
    }

    /**
     * Returns element count, initializing adder if necessary.
     */
    private long getAdderCount() {
        LongAdder a;
        long c;
        do {
        } while ((a = adder) == null &&
                !ADDER.compareAndSet(this, null, a = new LongAdder()));
        return (c = a.sum()) <= 0L ? 0L : c; // ignore transient negatives
    }

    /* ---------------- Traversal -------------- */

    /**
     * Returns an index node with key strictly less than given key.
     * Also unlinks indexes to deleted nodes found along the way.
     * Callers rely on this side-effect of clearing indices to deleted
     * nodes.
     *
     * @param key if nonnull the key
     * @return a predecessor node of key, or null if uninitialized or null key
     */
    private Node<K, V> findPredecessor(@Nullable K key) {
        VarHandle.acquireFence();
        Index<K, V> q;
        return (q = head) == null ? null : _findPredecessor(key, q);
    }

    private Node<K, V> _findPredecessor(K key, Index<K, V> q) {
        for (Index<K, V> d; ; ) {
            q = nextPredecessor(key, q);
            if ((d = q.down) == null)
                return q.node;

            q = d;
        }
    }

    private Index<K, V> nextPredecessor(K key, Index<K, V> q) {
        Index<K, V> r;
        while ((r = q.right) != null) {
            Node<K, V> p = r.node;
            K k;
            if (p == null || p.val == null || (k = p.key) == null)  // unlink index to deleted node
                absorbRight(q, r);
            else if (compare(key, k) > 0)
                q = r;
            else
                break;
        }
        return q;
    }

    private static void absorbRight(Index q, Index r) {
        RIGHT.compareAndSet(q, r, r.right);
    }

    /**
     * Returns node holding key or null if no such, clearing out any
     * deleted nodes seen along the way.  Repeatedly traverses at
     * base-level looking for key starting at predecessor returned
     * from findPredecessor, processing base-level deletions as
     * encountered. Restarts occur, at traversal step encountering
     * node n, if n's key field is null, indicating it is a marker, so
     * its predecessor is deleted before continuing, which we help do
     * by re-finding a valid predecessor.  The traversal loops in
     * doPut, doRemove, and findNear all include the same checks.
     *
     * @param key the key
     * @return node holding key, or null if no such
     */
    private Node<K, V> findNode(K key) {
        Node<K, V> b;
        outer:
        while ((b = findPredecessor(key)) != null) {
            for (; ; ) {
                Node<K, V> n;
                K k;
                if ((n = b.next) == null)
                    break outer;               // empty
                if ((k = n.key) == null)
                    break;                     // b is deleted

                int c;
                if (n.val == null)
                    unlinkNode(b, n);          // n is deleted
                else if ((c = compare(key, k)) > 0)
                    b = n;
                else if (c == 0)
                    return n;
                else
                    break outer;
            }
        }
        return null;
    }

    /**
     * Gets value for key. Same idea as findNode, except skips over
     * deletions and markers, and returns first encountered value to
     * avoid possibly inconsistent rereads.
     *
     * @param key the key
     * @return the value, or null if absent
     */
    private V doGet(K key) {
        Index<K, V> q;
        VarHandle.acquireFence();

        if ((q = head) != null) {
            for (Index<K, V> r, d; ; ) {
                while ((r = q.right) != null) {
                    Node<K, V> p;
                    K k;
                    V v;
                    int c;
                    if ((p = r.node) == null || (k = p.key) == null || (v = p.val) == null)
                        RIGHT.compareAndSet(q, r, r.right);
                    else if ((c = compare(key, k)) > 0)
                        q = r;
                    else if (c == 0) {
                        return v; //result = v; break outer;
                    } else
                        break;
                }
                if ((d = q.down) != null)
                    q = d;
                else {
                    Node<K, V> b, n;
                    if ((b = q.node) != null) {
                        while ((n = b.next) != null) {
                            V v;
                            int c;
                            K k = n.key;
                            if ((v = n.val) == null || k == null ||
                                    (c = compare(key, k)) > 0)
                                b = n;
                            else {
                                if (c == 0)
                                    return v;//result = v;
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
        return null;
    }

    /* ---------------- Insertion -------------- */

    /**
     * Main insertion method.  Adds element if not present, or
     * replaces value if present and onlyIfAbsent is false.
     *
     * @param key          the key
     * @param value        the value that must be associated with key
     * @param onlyIfAbsent if should not insert if already present
     * @return the old value, or null if newly inserted
     */
    private V doPut(K key, V value, boolean onlyIfAbsent) {
        for (; ; ) {
            VarHandle.acquireFence();
            int levels = 0;                    // number of levels descended
            Index<K, V> h;
            Node<K, V> b;
            if ((h = head) == null) {          // try to initialize
                Node<K, V> base = new Node<>(null);
                h = new Index<>(base, null);
                b = HEAD.compareAndSet(this, null, h) ? base : null;
            } else {
                for (Index<K, V> q = h, r, d; ; ) { // count while descending
                    while ((r = q.right) != null) {
                        Node<K, V> p;
                        K k;
                        if ((p = r.node) == null || (k = p.key) == null || p.val == null)
                            absorbRight(q, r);
                        else if (compare(key, k) > 0)
                            q = r;
                        else
                            break;
                    }
                    if ((d = q.down) != null) {
                        ++levels;
                        q = d;
                    } else {
                        b = q.node;
                        break;
                    }
                }
            }
            if (b != null) {
                Node<K, V> z = null;              // new node, if inserted
                for (; ; ) {                       // find insertion point
                    Node<K, V> n, p;
                    K k;
                    V v;
                    int c;
                    if ((n = b.next) == null) {
//                        if (b.key == null)       // if empty, type check key now
//                            cmp(key, key);
                        c = -1;
                    } else if ((k = n.key) == null)
                        break;                   // can't append; restart
                    else if ((v = n.val) == null) {
                        unlinkNode(b, n);
                        c = 1;
                    } else if ((c = compare(key, k)) > 0)
                        b = n;
                    else if (c == 0 &&
                            (onlyIfAbsent || VAL.compareAndSet(n, v, value)))
                        return v;

                    if (c < 0 &&
                            NEXT.compareAndSet(b, n, p = new Node<>(key, value, n))) {
                        z = p;
                        break;
                    }
                }

                if (z != null) {
                    final ThreadLocalRandom rng = ThreadLocalRandom.current();
                    int lr = rng.nextInt(); //nextSecondarySeed();
                    if ((lr & 0x3) == 0) {       // add indices with 1/4 prob
                        int hr = rng.nextInt(); //ThreadLocalRandom.nextSecondarySeed();
                        long rnd = (((long) hr) << 32) | (lr & 0xffffffffL);
                        int skips = levels;      // levels to descend before add
                        Index<K, V> x = null;
                        for (; ; ) {               // create at most 62 indices
                            x = new Index<>(z, x);
                            if (rnd >= 0L || --skips < 0)
                                break;
                            else
                                rnd <<= 1;
                        }
                        if (addIndices(h, skips, x) && skips < 0 && head == h)         // try to add new level
                            HEAD.compareAndSet(this, h,
                                    new Index<>(h.node, h, new Index<>(z, x)));
                        if (z.val == null)       // deleted while adding indices
                            findPredecessor(key); // clean
                    }
                    addCount(1);
                    return null;
                }
            }
        }
    }

    /**
     * Add indices after an insertion. Descends iteratively to the
     * highest level of insertion, then recursively, to chain index
     * nodes to lower ones. Returns null on (staleness) failure,
     * disabling higher-level insertions. Recursion depths are
     * exponentially less probable.
     *
     * @param q     starting index for current level
     * @param skips levels to skip before inserting
     * @param x     index for this insertion
     * @param cmp   comparator
     */
    private boolean addIndices(Index<K, V> q, int skips, Index<K, V> x) {
        Node<K, V> z;
        K key;
        if (x != null && (z = x.node) != null && (key = z.key) != null &&
                q != null) {                            // hoist checks
            boolean retrying = false;
            for (; ; ) {                              // find splice point
                Index<K, V> r, d;
                int c;
                if ((r = q.right) != null) {
                    Node<K, V> p;
                    K k;
                    if ((p = r.node) == null || (k = p.key) == null || p.val == null) {
                        absorbRight(q, r);
                        c = 0;
                    } else if ((c = compare(key, k)) > 0)
                        q = r;
                    else if (c == 0)
                        break;                      // stale
                } else
                    c = -1;

                if (c < 0) {
                    if ((d = q.down) != null && skips > 0) {
                        --skips;
                        q = d;
                    } else if (d != null && !retrying && !addIndices(d, 0, x.down))
                        break;
                    else {
                        if (RIGHT.compareAndSet(q, (x.right = r), x))
                            return true;
                        else
                            retrying = true;         // re-find splice point
                    }
                }
            }
        }
        return false;
    }

    abstract protected int compare(K x, K y);


    /* ---------------- Deletion -------------- */

    /**
     * Main deletion method. Locates node, nulls value, appends a
     * deletion marker, unlinks predecessor, removes associated index
     * nodes, and possibly reduces head index level.
     *
     * @param key   the key
     * @param value if non-null, the value that must be
     *              associated with key
     * @return the node, or null if not found
     */
    final V doRemove(K key, V value) {
        V result = null;
        Node<K, V> b;
        outer:
        while ((b = findPredecessor(key)) != null &&
                result == null) {
            for (; ; ) {
                Node<K, V> n;
                K k;
                V v;
                int c;
                if ((n = b.next) == null)
                    break outer;
                else if ((k = n.key) == null)
                    break;
                else if ((v = n.val) == null)
                    unlinkNode(b, n);
                else if ((c = compare(key, k)) > 0)
                    b = n;
                else if (c < 0)
                    break outer;
                else if (value != null && !value.equals(v))
                    break outer;
                else if (VAL.compareAndSet(n, v, null)) {
                    result = v;
                    unlinkNode(b, n);
                    break; // loop to clean up
                }
            }
        }
        if (result != null) {
            tryReduceLevel();
            addCount(-1);
        }
        return result;
    }

    /**
     * Possibly reduce head level if it has no nodes.  This method can
     * (rarely) make mistakes, in which case levels can disappear even
     * though they are about to contain index nodes. This impacts
     * performance, not correctness.  To minimize mistakes as well as
     * to reduce hysteresis, the level is reduced by one only if the
     * topmost three levels look empty. Also, if the removed level
     * looks non-empty after CAS, we try to change it back quick
     * before anyone notices our mistake! (This trick works pretty
     * well because this method will practically never make mistakes
     * unless current thread stalls immediately before first CAS, in
     * which case it is very unlikely to stall again immediately
     * afterwards, so will recover.)
     * <p>
     * We put up with all this rather than just let levels grow
     * because otherwise, even a small map that has undergone a large
     * number of insertions and removals will have a lot of levels,
     * slowing down access more than would an occasional unwanted
     * reduction.
     */
    private void tryReduceLevel() {
        Index<K, V> h, d, e;
        if ((h = head) != null && h.right == null &&
                (d = h.down) != null && d.right == null &&
                (e = d.down) != null && e.right == null &&
                HEAD.compareAndSet(this, h, d) &&
                h.right != null)   // recheck
            HEAD.compareAndSet(this, d, h);  // try to backout
    }

    /* ---------------- Finding and removing first element -------------- */

    /**
     * Gets first valid node, unlinking deleted nodes if encountered.
     *
     * @return first node or null if empty
     */
    final Node<K, V> findFirst() {
        Node<K, V> b, n;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                if (n.val == null)
                    unlinkNode(b, n);
                else
                    return n;
            }
        }
        return null;
    }

    /**
     * Entry snapshot version of findFirst
     */
    private AbstractMap.SimpleImmutableEntry<K, V> findFirstEntry() {
        Node<K, V> b, n;
        V v;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                if ((v = n.val) == null)
                    unlinkNode(b, n);
                else
                    return new AbstractMap.SimpleImmutableEntry<>(n.key, v);
            }
        }
        return null;
    }

    /**
     * Removes first entry; returns its snapshot.
     *
     * @return null if empty, else snapshot of first entry
     */
    private AbstractMap.SimpleImmutableEntry<K, V> doRemoveFirstEntry() {
        Node<K, V> b, n;
        V v;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                if ((v = n.val) == null || VAL.compareAndSet(n, v, null)) {
                    K k = n.key;
                    unlinkNode(b, n);
                    if (v != null) {
                        tryReduceLevel();
                        findPredecessor(k); // clean index
                        addCount(-1);
                        return new AbstractMap.SimpleImmutableEntry<>(k, v);
                    }
                }
            }
        }
        return null;
    }

    /* ---------------- Finding and removing last element -------------- */

    /**
     * Specialized version of find to get last valid node.
     *
     * @return last node or null if empty
     */
    final Node<K, V> findLast() {
        outer:
        for (; ; ) {
            Index<K, V> q;
            Node<K, V> b;
            VarHandle.acquireFence();
            if ((q = head) == null)
                break;
            for (Index<K, V> r, d; ; ) {
                while ((r = q.right) != null) {
                    Node<K, V> p;
                    if ((p = r.node) == null || p.val == null)
                        absorbRight(q, r);
                    else
                        q = r;
                }
                if ((d = q.down) != null)
                    q = d;
                else {
                    b = q.node;
                    break;
                }
            }
            if (b != null) {
                for (; ; ) {
                    Node<K, V> n;
                    if ((n = b.next) == null) {
                        if (b.key == null) // empty
                            break outer;
                        else
                            return b;
                    } else if (n.key == null)
                        break;
                    else if (n.val == null)
                        unlinkNode(b, n);
                    else
                        b = n;
                }
            }
        }
        return null;
    }

    /**
     * Entry version of findLast
     *
     * @return Entry for last node or null if empty
     */
    private AbstractMap.SimpleImmutableEntry<K, V> findLastEntry() {
        for (; ; ) {
            Node<K, V> n;
            V v;
            if ((n = findLast()) == null)
                return null;
            if ((v = n.val) != null)
                return new AbstractMap.SimpleImmutableEntry<>(n.key, v);
        }
    }

    /**
     * Removes last entry; returns its snapshot.
     * Specialized variant of doRemove.
     *
     * @return null if empty, else snapshot of last entry
     */
    private Map.Entry<K, V> doRemoveLastEntry() {
        outer:
        for (; ; ) {
            Index<K, V> q;
            Node<K, V> b;
            VarHandle.acquireFence();
            if ((q = head) == null)
                break;
            for (; ; ) {
                Index<K, V> d, r;
                Node<K, V> p;
                while ((r = q.right) != null) {
                    if ((p = r.node) == null || p.val == null)
                        absorbRight(q, r);
                    else if (p.next != null)
                        q = r;  // continue only if a successor
                    else
                        break;
                }
                if ((d = q.down) != null)
                    q = d;
                else {
                    b = q.node;
                    break;
                }
            }
            if (b != null) {
                for (; ; ) {
                    Node<K, V> n;
                    K k;
                    V v;
                    if ((n = b.next) == null) {
                        if (b.key == null) // empty
                            break outer;
                        else
                            break; // retry
                    } else if ((k = n.key) == null)
                        break;
                    else if ((v = n.val) == null)
                        unlinkNode(b, n);
                    else if (n.next != null)
                        b = n;
                    else if (VAL.compareAndSet(n, v, null)) {
                        unlinkNode(b, n);
                        tryReduceLevel();
                        findPredecessor(k); // clean index
                        addCount(-1);
                        return new AbstractMap.SimpleImmutableEntry<>(k, v);
                    }
                }
            }
        }
        return null;
    }

    /* ---------------- Relational operations -------------- */

    // Control values OR'ed as arguments to findNear

    private static final int EQ = 1;
    private static final int LT = 2;
    private static final int GT = 0; // Actually checked as !LT

    /**
     * Utility for ceiling, floor, lower, higher methods.
     *
     * @param key the key
     * @param rel the relation -- OR'ed combination of EQ, LT, GT
     * @return nearest node fitting relation, or null if no such
     */
    final Node<K, V> findNear(K key, final int rel) {
        boolean relLTnz = (rel & LT) != 0;
        boolean relEQnz = (rel & EQ) != 0;

        for (; ; ) {
            Node<K, V> b;
            if ((b = findPredecessor(key)) == null)
                return null; // empty

            for (; ; ) {
                Node<K, V> n;
                K k;
                if ((n = b.next) == null)
                    return relLTnz && b.key != null ? b : null;
                else if ((k = n.key) == null)
                    break;
                else if (n.val == null)
                    unlinkNode(b, n);
                else {
                    int c;
                    if (((((c = compare(key, k))) == 0) && relEQnz)
                            ||
                            (c < 0 && !relLTnz)) {
                        return n;
                    } else if (c <= 0 && relLTnz) {
                        return b.key != null ? b : null;
                    } else
                        b = n;
                }
            }
        }
    }

    /**
     * Variant of findNear returning SimpleImmutableEntry
     *
     * @param key the key
     * @param rel the relation -- OR'ed combination of EQ, LT, GT
     * @return Entry fitting relation, or null if no such
     */
    final AbstractMap.SimpleImmutableEntry<K, V> findNearEntry(K key, int rel) {
        for (; ; ) {
            Node<K, V> n;
            if ((n = findNear(key, rel)) == null)
                return null;
            V v;
            if ((v = n.val) != null)
                return new AbstractMap.SimpleImmutableEntry<>(n.key, v);
        }
    }

    /**
     * Returns {@code true} if this map contains a mapping for the specified
     * key.
     *
     * @param key key whose presence in this map is to be tested
     * @return {@code true} if this map contains a mapping for the specified key
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    public boolean containsKey(Object key) {
        return get(key) != null;
    }

    /**
     * Returns the value to which the specified key is mapped,
     * or {@code null} if this map contains no mapping for the key.
     *
     * <p>More formally, if this map contains a mapping from a key
     * {@code k} to a value {@code v} such that {@code key} compares
     * equal to {@code k} according to the map's ordering, then this
     * method returns {@code v}; otherwise it returns {@code null}.
     * (There can be at most one such mapping.)
     *
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    public V get(Object key) {
        return doGet((K) key);
    }

    /**
     * Returns the value to which the specified key is mapped,
     * or the given defaultValue if this map contains no mapping for the key.
     *
     * @param key          the key
     * @param defaultValue the value to return if this map contains
     *                     no mapping for the given key
     * @return the mapping for the key, if present; else the defaultValue
     * @since 1.8
     */
    public V getOrDefault(Object key, V defaultValue) {
        V v;
        return (v = get(key)) == null ? defaultValue : v;
    }

    /**
     * Associates the specified value with the specified key in this map.
     * If the map previously contained a mapping for the key, the old
     * value is replaced.
     *
     * @param key   key with which the specified value is to be associated
     * @param value value to be associated with the specified key
     * @return the previous value associated with the specified key, or
     * {@code null} if there was no mapping for the key
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    public V put(K key, V value) {
        return doPut(key, value, false);
    }

    /**
     * Removes the mapping for the specified key from this map if present.
     *
     * @param key key for which mapping should be removed
     * @return the previous value associated with the specified key, or
     * {@code null} if there was no mapping for the key
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    public V remove(Object key) {
        return doRemove((K) key, null);
    }

    /**
     * Returns {@code true} if this map maps one or more keys to the
     * specified value.  This operation requires time linear in the
     * map size. Additionally, it is possible for the map to change
     * during execution of this method, in which case the returned
     * result may be inaccurate.
     *
     * @param value value whose presence in this map is to be tested
     * @return {@code true} if a mapping to {@code value} exists;
     * {@code false} otherwise
     */
    public boolean containsValue(Object value) {
        Node<K, V> b, n;
        V v;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                if ((v = n.val) != null && value.equals(v))
                    return true;
                else
                    b = n;
            }
        }
        return false;
    }

    /**
     * {@inheritDoc}
     */
    public int size() {
        long c;
        return baseHead() == null ? 0 :
                (c = getAdderCount()) >= Integer.MAX_VALUE ?
                        Integer.MAX_VALUE : (int) c;
    }

    /**
     * {@inheritDoc}
     */
    public boolean isEmpty() {
        return findFirst() == null;
    }

    /**
     * Removes all of the mappings from this map.
     */
    public void clear() {
        Index<K, V> h, r, d;
        Node<K, V> b;
        VarHandle.acquireFence();
        while ((h = head) != null) {
            if ((r = h.right) != null)        // remove indices
                RIGHT.compareAndSet(h, r, null);
            else if ((d = h.down) != null)    // remove levels
                HEAD.compareAndSet(this, h, d);
            else {
                int count = 0;
                if ((b = h.node) != null) {    // remove nodes
                    Node<K, V> n;
                    V v;
                    while ((n = b.next) != null) {
                        if ((v = n.val) != null &&
                                VAL.compareAndSet(n, v, null)) {
                            --count;
                            v = null;
                        }
                        if (v == null)
                            unlinkNode(b, n);
                    }
                }
                if (count != 0L)
                    addCount(count);
                else
                    break;
            }
        }
    }

    /**
     * If the specified key is not already associated with a value,
     * attempts to compute its value using the given mapping function
     * and enters it into this map unless {@code null}.  The function
     * is <em>NOT</em> guaranteed to be applied once atomically only
     * if the value is not present.
     *
     * @param key             key with which the specified value is to be associated
     * @param mappingFunction the function to compute a value
     * @return the current (existing or computed) value associated with
     * the specified key, or null if the computed value is null
     * @since 1.8
     */
    public V computeIfAbsent(K key,
                             Function<? super K, ? extends V> mappingFunction) {
        V v, p, r;
        if ((v = doGet(key)) == null &&
                (r = mappingFunction.apply(key)) != null)
            v = (p = doPut(key, r, true)) == null ? r : p;
        return v;
    }

    /**
     * If the value for the specified key is present, attempts to
     * compute a new mapping given the key and its current mapped
     * value. The function is <em>NOT</em> guaranteed to be applied
     * once atomically.
     *
     * @param key               key with which a value may be associated
     * @param remappingFunction the function to compute a value
     * @return the new value associated with the specified key, or null if none
     * @since 1.8
     */
    public V computeIfPresent(K key,
                              BiFunction<? super K, ? super V, ? extends V> remappingFunction) {
        Node<K, V> n;
        V v;
        while ((n = findNode(key)) != null) {
            if ((v = n.val) != null) {
                V r = remappingFunction.apply(key, v);
                if (r != null) {
                    if (VAL.compareAndSet(n, v, r))
                        return r;
                } else if (doRemove(key, v) != null)
                    break;
            }
        }
        return null;
    }

    /**
     * Attempts to compute a mapping for the specified key and its
     * current mapped value (or {@code null} if there is no current
     * mapping). The function is <em>NOT</em> guaranteed to be applied
     * once atomically.
     *
     * @param key               key with which the specified value is to be associated
     * @param remappingFunction the function to compute a value
     * @return the new value associated with the specified key, or null if none
     * @since 1.8
     */
    public V compute(K key,
                     BiFunction<? super K, ? super V, ? extends V> remappingFunction) {
        for (; ; ) {
            Node<K, V> n;
            V v;
            V r;
            if ((n = findNode(key)) == null) {
                if ((r = remappingFunction.apply(key, null)) == null)
                    break;
                if (doPut(key, r, true) == null)
                    return r;
            } else if ((v = n.val) != null) {
                if ((r = remappingFunction.apply(key, v)) != null) {
                    if (VAL.compareAndSet(n, v, r))
                        return r;
                } else if (doRemove(key, v) != null)
                    break;
            }
        }
        return null;
    }

    /**
     * If the specified key is not already associated with a value,
     * associates it with the given value.  Otherwise, replaces the
     * value with the results of the given remapping function, or
     * removes if {@code null}. The function is <em>NOT</em>
     * guaranteed to be applied once atomically.
     *
     * @param key               key with which the specified value is to be associated
     * @param value             the value to use if absent
     * @param remappingFunction the function to recompute a value if present
     * @return the new value associated with the specified key, or null if none
     * @since 1.8
     */
    public V merge(K key, V value,
                   BiFunction<? super V, ? super V, ? extends V> remappingFunction) {
        for (; ; ) {
            Node<K, V> n;
            V v;
            V r;
            if ((n = findNode(key)) == null) {
                if (doPut(key, value, true) == null)
                    return value;
            } else if ((v = n.val) != null) {
                if ((r = remappingFunction.apply(v, value)) != null) {
                    if (VAL.compareAndSet(n, v, r))
                        return r;
                } else if (doRemove(key, v) != null)
                    return null;
            }
        }
    }

    /* ---------------- View methods -------------- */

    /*
     * Note: Lazy initialization works for views because view classes
     * are stateless/immutable so it doesn't matter wrt correctness if
     * more than one is created (which will only rarely happen).  Even
     * so, the following idiom conservatively ensures that the method
     * returns the one it created if it does so, not one created by
     * another racing thread.
     */

    /**
     * Returns a {@link NavigableSet} view of the keys contained in this map.
     *
     * <p>The set's iterator returns the keys in ascending order.
     * The set's spliterator additionally reports {@link Spliterator#CONCURRENT},
     * {@link Spliterator#NONNULL}, {@link Spliterator#SORTED} and
     * {@link Spliterator#ORDERED}, with an encounter order that is ascending
     * key order.
     *
     * <p>The {@linkplain Spliterator#getComparator() spliterator's comparator}
     * is {@code null} if the {@linkplain #comparator() map's comparator}
     * is {@code null}.
     * Otherwise, the spliterator's comparator is the same as or imposes the
     * same total ordering as the map's comparator.
     *
     * <p>The set is backed by the map, so changes to the map are
     * reflected in the set, and vice-versa.  The set supports element
     * removal, which removes the corresponding mapping from the map,
     * via the {@code Iterator.remove}, {@code Set.remove},
     * {@code removeAll}, {@code retainAll}, and {@code clear}
     * operations.  It does not support the {@code add} or {@code addAll}
     * operations.
     *
     * <p>The view's iterators and spliterators are
     * <a href="package-summary.html#Weakly"><i>weakly consistent</i></a>.
     *
     * <p>This method is equivalent to method {@code navigableKeySet}.
     *
     * @return a navigable set view of the keys in this map
     */
    public NavigableSet<K> keySet() {
        KeySet<K, V> ks;
        if ((ks = keySet) != null) return ks;
        return keySet = new KeySet<>(this);
    }

    public NavigableSet<K> navigableKeySet() {
        KeySet<K, V> ks;
        if ((ks = keySet) != null) return ks;
        return keySet = new KeySet<>(this);
    }

    /**
     * Returns a {@link Collection} view of the values contained in this map.
     * <p>The collection's iterator returns the values in ascending order
     * of the corresponding keys. The collections's spliterator additionally
     * reports {@link Spliterator#CONCURRENT}, {@link Spliterator#NONNULL} and
     * {@link Spliterator#ORDERED}, with an encounter order that is ascending
     * order of the corresponding keys.
     *
     * <p>The collection is backed by the map, so changes to the map are
     * reflected in the collection, and vice-versa.  The collection
     * supports element removal, which removes the corresponding
     * mapping from the map, via the {@code Iterator.remove},
     * {@code Collection.remove}, {@code removeAll},
     * {@code retainAll} and {@code clear} operations.  It does not
     * support the {@code add} or {@code addAll} operations.
     *
     * <p>The view's iterators and spliterators are
     * <a href="package-summary.html#Weakly"><i>weakly consistent</i></a>.
     */
    public Collection<V> values() {
        Values<K, V> vs;
        if ((vs = values) != null) return vs;
        return values = new Values<>(this);
    }

    /**
     * Returns a {@link Set} view of the mappings contained in this map.
     *
     * <p>The set's iterator returns the entries in ascending key order.  The
     * set's spliterator additionally reports {@link Spliterator#CONCURRENT},
     * {@link Spliterator#NONNULL}, {@link Spliterator#SORTED} and
     * {@link Spliterator#ORDERED}, with an encounter order that is ascending
     * key order.
     *
     * <p>The set is backed by the map, so changes to the map are
     * reflected in the set, and vice-versa.  The set supports element
     * removal, which removes the corresponding mapping from the map,
     * via the {@code Iterator.remove}, {@code Set.remove},
     * {@code removeAll}, {@code retainAll} and {@code clear}
     * operations.  It does not support the {@code add} or
     * {@code addAll} operations.
     *
     * <p>The view's iterators and spliterators are
     * <a href="package-summary.html#Weakly"><i>weakly consistent</i></a>.
     *
     * <p>The {@code Map.Entry} elements traversed by the {@code iterator}
     * or {@code spliterator} do <em>not</em> support the {@code setValue}
     * operation.
     *
     * @return a set view of the mappings contained in this map,
     * sorted in ascending key order
     */
    public Set<Map.Entry<K, V>> entrySet() {
        EntrySet<K, V> es;
        if ((es = entrySet) != null) return es;
        return entrySet = new EntrySet<>(this);
    }

    public ConcurrentNavigableMap<K, V> descendingMap() {
        ConcurrentNavigableMap<K, V> dm;
        if ((dm = descendingMap) != null) return dm;
        return descendingMap =
                new SubMap<>(this, null, false, null, false, true);
    }

    public NavigableSet<K> descendingKeySet() {
        return descendingMap().navigableKeySet();
    }

    /* ---------------- AbstractMap Overrides -------------- */


    /**
     * Compares the specified object with this map for equality.
     * Returns {@code true} if the given object is also a map and the
     * two maps represent the same mappings.  More formally, two maps
     * {@code m1} and {@code m2} represent the same mappings if
     * {@code m1.entrySet().equals(m2.entrySet())}.  This
     * operation may return misleading results if either map is
     * concurrently modified during execution of this method.
     *
     * @param o object to be compared for equality with this map
     * @return {@code true} if the specified object is equal to this map
     */
    public boolean equals(Object o) {
        throw new UnsupportedOperationException();
    }

    /* ------ ConcurrentMap API methods ------ */

    /**
     * {@inheritDoc}
     *
     * @return the previous value associated with the specified key,
     * or {@code null} if there was no mapping for the key
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    public V putIfAbsent(K key, V value) {
        return doPut(key, value, true);
    }

    /**
     * {@inheritDoc}
     *
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    boolean _remove(K key, V value) {
        return value != null && doRemove(key, value) != null;
    }

    public final boolean remove(Object key, Object value) {
        return _remove((K) key, (V) value);
    }


    /**
     * {@inheritDoc}
     *
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    public boolean replace(K key, V oldValue, V newValue) {
        for (; ; ) {
            Node<K, V> n;
            V v;
            if ((n = findNode(key)) == null)
                return false;
            if ((v = n.val) != null) {
                if (!oldValue.equals(v))
                    return false;
                if (VAL.compareAndSet(n, v, newValue))
                    return true;
            }
        }
    }

    /**
     * {@inheritDoc}
     *
     * @return the previous value associated with the specified key,
     * or {@code null} if there was no mapping for the key
     * @throws ClassCastException if the specified key cannot be compared
     *                            with the keys currently in the map
     */
    public V replace(K key, V value) {
        for (; ; ) {
            Node<K, V> n;
            V v;
            if ((n = findNode(key)) == null)
                return null;
            if ((v = n.val) != null && VAL.compareAndSet(n, v, value))
                return v;
        }
    }

    @Deprecated
    @Override
    public final Comparator<K> comparator() {
        return this::compare;
    }

    //    /**
//     * @throws NoSuchElementException {@inheritDoc}
//     */
    @Nullable
    public K firstKey() {
        Node<K, V> n = findFirst();
        return n == null ? null : n.key;
    }

    //    /**
//     * @throws NoSuchElementException {@inheritDoc}
//     */
    @Nullable
    public K lastKey() {
        Node<K, V> n = findLast();
        return n == null ? null : n.key;
    }

    /**
     * @throws ClassCastException       {@inheritDoc}
     * @throws IllegalArgumentException {@inheritDoc}
     */
    public ConcurrentNavigableMap<K, V> subMap(K fromKey,
                                               boolean fromInclusive,
                                               K toKey,
                                               boolean toInclusive) {
        return new SubMap<>
                (this, fromKey, fromInclusive, toKey, toInclusive, false);
    }

    /**
     * @throws ClassCastException       {@inheritDoc}
     * @throws IllegalArgumentException {@inheritDoc}
     */
    public ConcurrentNavigableMap<K, V> headMap(K toKey, boolean inclusive) {
        return new SubMap<>(this, null, false, toKey, inclusive, false);
    }

    public ConcurrentNavigableMap<K, V> headMapDescending(K toKey, boolean inclusive) {
        return new SubMap<>(this, null, false, toKey, inclusive, true);
    }

    /**
     * @throws ClassCastException       {@inheritDoc}
     * @throws IllegalArgumentException {@inheritDoc}
     */
    public ConcurrentNavigableMap<K, V> tailMap(K fromKey, boolean inclusive) {
        return new SubMap<>
                (this, fromKey, inclusive, null, false, false);
    }

    /**
     * @throws ClassCastException       {@inheritDoc}
     * @throws IllegalArgumentException {@inheritDoc}
     */
    public ConcurrentNavigableMap<K, V> subMap(K fromKey, K toKey) {
        return subMap(fromKey, true, toKey, false);
    }

    /**
     * @throws ClassCastException       {@inheritDoc}
     * @throws IllegalArgumentException {@inheritDoc}
     */
    public ConcurrentNavigableMap<K, V> headMap(K toKey) {
        return headMap(toKey, false);
    }

    /**
     * @throws ClassCastException       {@inheritDoc}
     * @throws IllegalArgumentException {@inheritDoc}
     */
    public ConcurrentNavigableMap<K, V> tailMap(K fromKey) {
        return tailMap(fromKey, true);
    }

    /* ---------------- Relational operations -------------- */

    /**
     * Returns a key-value mapping associated with the greatest key
     * strictly less than the given key, or {@code null} if there is
     * no such key. The returned entry does <em>not</em> support the
     * {@code Entry.setValue} method.
     *
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public Map.Entry<K, V> lowerEntry(K key) {
        return findNearEntry(key, LT);
    }

    /**
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public K lowerKey(K key) {
        Node<K, V> n = findNear(key, LT);
        return n == null ? null : n.key;
    }

    /**
     * Returns a key-value mapping associated with the greatest key
     * less than or equal to the given key, or {@code null} if there
     * is no such key. The returned entry does <em>not</em> support
     * the {@code Entry.setValue} method.
     *
     * @param key the key
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public Map.Entry<K, V> floorEntry(K key) {
        return findNearEntry(key, LT | EQ);
    }

    /**
     * @param key the key
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public K floorKey(K key) {
        Node<K, V> n = findNear(key, LT | EQ);
        return n == null ? null : n.key;
    }

    /**
     * Returns a key-value mapping associated with the least key
     * greater than or equal to the given key, or {@code null} if
     * there is no such entry. The returned entry does <em>not</em>
     * support the {@code Entry.setValue} method.
     *
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public Map.Entry<K, V> ceilingEntry(K key) {
        return findNearEntry(key, GT | EQ);
    }

    /**
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public K ceilingKey(K key) {
        Node<K, V> n = findNear(key, GT | EQ);
        return n == null ? null : n.key;
    }

    /**
     * Returns a key-value mapping associated with the least key
     * strictly greater than the given key, or {@code null} if there
     * is no such key. The returned entry does <em>not</em> support
     * the {@code Entry.setValue} method.
     *
     * @param key the key
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public Map.Entry<K, V> higherEntry(K key) {
        return findNearEntry(key, GT);
    }

    /**
     * @param key the key
     * @throws ClassCastException   {@inheritDoc}
     * @throws NullPointerException if the specified key is null
     */
    public K higherKey(K key) {
        Node<K, V> n = findNear(key, GT);
        return n == null ? null : n.key;
    }

    /**
     * Returns a key-value mapping associated with the least
     * key in this map, or {@code null} if the map is empty.
     * The returned entry does <em>not</em> support
     * the {@code Entry.setValue} method.
     */
    public Map.Entry<K, V> firstEntry() {
        return findFirstEntry();
    }

    /**
     * Returns a key-value mapping associated with the greatest
     * key in this map, or {@code null} if the map is empty.
     * The returned entry does <em>not</em> support
     * the {@code Entry.setValue} method.
     */
    public Map.Entry<K, V> lastEntry() {
        return findLastEntry();
    }

    /**
     * Removes and returns a key-value mapping associated with
     * the least key in this map, or {@code null} if the map is empty.
     * The returned entry does <em>not</em> support
     * the {@code Entry.setValue} method.
     */
    public Map.Entry<K, V> pollFirstEntry() {
        return doRemoveFirstEntry();
    }

    /**
     * Removes and returns a key-value mapping associated with
     * the greatest key in this map, or {@code null} if the map is empty.
     * The returned entry does <em>not</em> support
     * the {@code Entry.setValue} method.
     */
    public Map.Entry<K, V> pollLastEntry() {
        return doRemoveLastEntry();
    }

    /* ---------------- Iterators -------------- */

    /**
     * Base of iterator classes
     */
    private sealed abstract class Iter<T> implements Iterator<T> {
        /**
         * the last node returned by next()
         */
        Node<K, V> lastReturned;
        /**
         * the next node to return from next();
         */
        Node<K, V> next;
        /**
         * Cache of next value field to maintain weak consistency
         */
        V nextValue;

        /**
         * Initializes ascending iterator for entire range.
         */
        Iter() {
            advance(baseHead());
        }

        public final boolean hasNext() {
            return next != null;
        }

        /**
         * Advances next to higher entry.
         */
        final void advance(Node<K, V> b) {
            Node<K, V> n = null;
            V v = null;
            if ((lastReturned = b) != null) {
                while ((n = b.next) != null && (v = n.val) == null)
                    b = n;
            }
            nextValue = v;
            next = n;
        }

        public final void remove() {
            Node<K, V> n;
            K k;
            if ((n = lastReturned) == null || (k = n.key) == null)
                throw new IllegalStateException();
            // It would not be worth all of the overhead to directly
            // unlink from here. Using remove is fast enough.
            ConcurrentSkipListMap2.this.remove(k);
            lastReturned = null;
        }
    }

    final class ValueIterator extends Iter<V> {
        public V next() {
            V v = nextValue;
            advance(next);
            return v;
        }
    }

    final class KeyIterator extends Iter<K> {
        public K next() {
            Node<K, V> n = next;
            K k = n.key;
            advance(n);
            return k;
        }
    }

    final class EntryIterator extends Iter<Map.Entry<K, V>> {
        public Map.Entry<K, V> next() {
            Node<K, V> n = next;
            V v = nextValue;
            advance(n);
            return new AbstractMap.SimpleImmutableEntry<>(n.key, v);
        }
    }

    /* ---------------- View Classes -------------- */

    /*
     * View classes are static, delegating to a ConcurrentNavigableMap
     * to allow use by SubMaps, which outweighs the ugliness of
     * needing type-tests for Iterator methods.
     */

    private static <E> List<E> toList(Collection<E> c) {
        // Using size() here would be a pessimization.
        List<E> list = new Lst<>(c.size());
        list.addAll(c);
        return list;
    }

    static final class KeySet<K, V> extends AbstractSet<K> implements NavigableSet<K> {
        final ConcurrentNavigableMap<K, V> m;

        KeySet(ConcurrentNavigableMap<K, V> map) {
            m = map;
        }

        public int size() {
            return m.size();
        }

        public boolean isEmpty() {
            return m.isEmpty();
        }

        public boolean contains(Object o) {
            return m.containsKey(o);
        }

        public boolean remove(Object o) {
            return m.remove(o) != null;
        }

        public void clear() {
            m.clear();
        }

        public K lower(K e) {
            return m.lowerKey(e);
        }

        public K floor(K e) {
            return m.floorKey(e);
        }

        public K ceiling(K e) {
            return m.ceilingKey(e);
        }

        public K higher(K e) {
            return m.higherKey(e);
        }

        public Comparator<? super K> comparator() {
            return m.comparator();
        }

        public K first() {
            return m.firstKey();
        }

        public K last() {
            return m.lastKey();
        }

        public K pollFirst() {
            return keyOrNull(m.pollFirstEntry());
        }

        public K pollLast() {
            return keyOrNull(m.pollLastEntry());
        }

        public Iterator<K> iterator() {
            return m instanceof ConcurrentSkipListMap2 c
                    ? c.new KeyIterator()
                    : ((SubMap<K, V>) m).new SubMapKeyIterator();
        }

        public boolean equals(Object o) {
            if (o == this)
                return true;
            return o instanceof Collection<?> c && containsAll(c) && c.containsAll(this);
            //            try {
            //            } catch (ClassCastException | NullPointerException unused) {
//                return false;
//            }
        }

        public Object[] toArray() {
            return toList(this).toArray();
        }

        public <T> T[] toArray(T[] a) {
            return toList(this).toArray(a);
        }

        public Iterator<K> descendingIterator() {
            return descendingSet().iterator();
        }

        public NavigableSet<K> subSet(K fromElement,
                                      boolean fromInclusive,
                                      K toElement,
                                      boolean toInclusive) {
            return new KeySet<>(m.subMap(fromElement, fromInclusive,
                    toElement, toInclusive));
        }

        public NavigableSet<K> headSet(K toElement, boolean inclusive) {
            return new KeySet<>(m.headMap(toElement, inclusive));
        }

        public NavigableSet<K> tailSet(K fromElement, boolean inclusive) {
            return new KeySet<>(m.tailMap(fromElement, inclusive));
        }

        public NavigableSet<K> subSet(K fromElement, K toElement) {
            return subSet(fromElement, true, toElement, false);
        }

        public NavigableSet<K> headSet(K toElement) {
            return headSet(toElement, false);
        }

        public NavigableSet<K> tailSet(K fromElement) {
            return tailSet(fromElement, true);
        }

        public NavigableSet<K> descendingSet() {
            return new KeySet<>(m.descendingMap());
        }

        public Spliterator<K> spliterator() {
            return m instanceof ConcurrentSkipListMap2 c
                    ? c.keySpliterator()
                    : ((SubMap<K, V>) m).new SubMapKeyIterator();
        }
    }

    @Nullable
    private static <K, V> K keyOrNull(Entry<K, V> e) {
        return e == null ? null : e.getKey();
    }

    final static class Values<K, V> extends AbstractCollection<V> {
        final ConcurrentNavigableMap<K, V> m;

        Values(ConcurrentNavigableMap<K, V> map) {
            m = map;
        }

        public Iterator<V> iterator() {
            return m instanceof ConcurrentSkipListMap2 c
                    ? c.new ValueIterator()
                    : ((SubMap<K, V>) m).new SubMapValueIterator();
        }

        public int size() {
            return m.size();
        }

        public boolean isEmpty() {
            return m.isEmpty();
        }

        public boolean contains(Object o) {
            return m.containsValue(o);
        }

        public void clear() {
            m.clear();
        }

        public Object[] toArray() {
            return toList(this).toArray();
        }

        public <T> T[] toArray(T[] a) {
            return toList(this).toArray(a);
        }

        public Spliterator<V> spliterator() {
            return m instanceof ConcurrentSkipListMap2 c
                    ? c.valueSpliterator()
                    : ((SubMap<K, V>) m).new SubMapValueIterator();
        }

        public boolean removeIf(Predicate<? super V> filter) {
            if (m instanceof ConcurrentSkipListMap2 c)
                return c.removeValueIf(filter);
            // else use iterator
            Iterator<Map.Entry<K, V>> it =
                    ((SubMap<K, V>) m).new SubMapEntryIterator();
            boolean removed = false;
            while (it.hasNext()) {
                Map.Entry<K, V> e = it.next();
                V v = e.getValue();
                if (filter.test(v) &&
                        //m._remove(e.getKey(), v))
                        ((ConcurrentSkipListMap2<K, V>) m)._remove(e.getKey(), v))
                    removed = true;
            }
            return removed;
        }
    }

    static final class EntrySet<K, V> extends AbstractSet<Map.Entry<K, V>> {
        final ConcurrentNavigableMap<K, V> m;

        EntrySet(ConcurrentNavigableMap<K, V> map) {
            m = map;
        }

        public Iterator<Map.Entry<K, V>> iterator() {
            return m instanceof ConcurrentSkipListMap2 c
                    ? c.new EntryIterator()
                    : ((SubMap<K, V>) m).new SubMapEntryIterator();
        }

        public boolean contains(Object o) {
            if (!(o instanceof Entry<?, ?> e))
                return false;
            V v = m.get(e.getKey());
            return v != null && v.equals(e.getValue());
        }

        public boolean remove(Object o) {
            return o instanceof Entry<?, ?> e && m.remove(e.getKey(), e.getValue());
        }

        public boolean isEmpty() {
            return m.isEmpty();
        }

        public int size() {
            return m.size();
        }

        public void clear() {
            m.clear();
        }

        public boolean equals(Object o) {
            if (o == this)
                return true;
            return o instanceof Collection<?> c && containsAll(c) && c.containsAll(this);
            //            try {
            //            } catch (ClassCastException | NullPointerException unused) {
//                return false;
//            }
        }

        public Object[] toArray() {
            return toList(this).toArray();
        }

        public <T> T[] toArray(T[] a) {
            return toList(this).toArray(a);
        }

        public Spliterator<Map.Entry<K, V>> spliterator() {
            return m instanceof ConcurrentSkipListMap2 c
                    ? c.entrySpliterator()
                    : ((SubMap<K, V>) m).new SubMapEntryIterator();
        }

        public boolean removeIf(Predicate<? super Map.Entry<K, V>> filter) {
            if (m instanceof ConcurrentSkipListMap2 c)
                return c.removeEntryIf(filter);
            // else use iterator
            Iterator<Map.Entry<K, V>> it =
                    ((SubMap<K, V>) m).new SubMapEntryIterator();
            boolean removed = false;
            while (it.hasNext()) {
                Map.Entry<K, V> e = it.next();
                if (filter.test(e) && m.remove(e.getKey(), e.getValue()))
                    removed = true;
            }
            return removed;
        }
    }

    /**
     * Submaps returned by {@link ConcurrentSkipListMap2} submap operations
     * represent a subrange of mappings of their underlying maps.
     * Instances of this class support all methods of their underlying
     * maps, differing in that mappings outside their range are ignored,
     * and attempts to add mappings outside their ranges result in {@link
     * IllegalArgumentException}.  Instances of this class are constructed
     * only using the {@code subMap}, {@code headMap}, and {@code tailMap}
     * methods of their underlying maps.
     *
     * @serial include
     */
    static final class SubMap<K, V> extends AbstractMap<K, V> implements ConcurrentNavigableMap<K, V> {

        /**
         * Underlying map
         */
        final ConcurrentSkipListMap2<K, V> m;
        /**
         * lower bound key, or null if from start
         */

        private final K lo;
        /**
         * upper bound key, or null if to end
         */

        private final K hi;
        /**
         * inclusion flag for lo
         */
        private final boolean loInclusive;
        /**
         * inclusion flag for hi
         */
        private final boolean hiInclusive;
        /**
         * direction
         */
        final boolean isDescending;

        // Lazily initialized view holders
        private transient KeySet<K, V> keySetView;
        private transient Values<K, V> valuesView;
        private transient EntrySet<K, V> entrySetView;

        /**
         * Creates a new submap, initializing all fields.
         */
        SubMap(ConcurrentSkipListMap2<K, V> map,
               K fromKey, boolean fromInclusive,
               K toKey, boolean toInclusive,
               boolean isDescending) {
//            if (fromKey != null && toKey != null &&
//                    map.cmp(fromKey, toKey) > 0)
//                throw new IllegalArgumentException("inconsistent range");
            this.m = map;
            this.lo = fromKey;
            this.hi = toKey;
            this.loInclusive = fromInclusive;
            this.hiInclusive = toInclusive;
            this.isDescending = isDescending;
        }

        /* ----------------  Utilities -------------- */

        boolean tooLow(K key) {
            int c;
            return lo != null && ((c = m.compare(key, lo)) < 0 ||
                    c == 0 && !loInclusive);
        }

        boolean tooHigh(K key) {
            int c;
            return hi != null && ((c = m.compare(key, hi)) > 0 ||
                    c == 0 && !hiInclusive);
        }

        boolean inBounds(K key) {
            return !tooLow(key) && !tooHigh(key);
        }

//        void checkKeyBounds(K key) {
////            if (key == null)             throw new NullPointerException();
////            if (!inBounds(key))
////                throw new IllegalArgumentException("key out of range");
//        }

        /**
         * Returns true if node key is less than upper bound of range.
         */
        boolean isBeforeEnd(ConcurrentSkipListMap2.Node<K, V> n) {
            if (n == null)
                return false;
            if (hi == null)
                return true;
            K k = n.key;
            if (k == null) // pass by markers and headers
                return true;
            int c = m.compare(k, hi);
            return c < 0 || (c == 0 && hiInclusive);
        }

        /**
         * Returns lowest node. This node might not be in range, so
         * most usages need to check bounds.
         */
        ConcurrentSkipListMap2.Node<K, V> loNode() {
            return lo == null ? m.findFirst() : m.findNear(lo, loInclusive ? GT | EQ : GT);
        }

        /**
         * Returns highest node. This node might not be in range, so
         * most usages need to check bounds.
         */
        ConcurrentSkipListMap2.Node<K, V> hiNode() {
            return hi == null ? m.findLast() : m.findNear(hi, hiInclusive ? LT | EQ : LT);
        }

        /**
         * Returns lowest absolute key (ignoring directionality).
         */
        K lowestKey() {
            ConcurrentSkipListMap2.Node<K, V> n = loNode();
            if (isBeforeEnd(n))
                return n.key;
            else
                throw new NoSuchElementException();
        }

        /**
         * Returns highest absolute key (ignoring directionality).
         */
        K highestKey() {
            ConcurrentSkipListMap2.Node<K, V> n = hiNode();
            if (n != null) {
                return n.key;
//                K last = n.key;
//                if (inBounds(last))
//                    return last;
            }
            throw new NoSuchElementException();
        }

        Map.Entry<K, V> lowestEntry() {
            for (; ; ) {
                ConcurrentSkipListMap2.Node<K, V> n;
                V v;
                if ((n = loNode()) == null || !isBeforeEnd(n))
                    return null;
                else if ((v = n.val) != null)
                    return new AbstractMap.SimpleImmutableEntry<>(n.key, v);
            }
        }

        Map.Entry<K, V> highestEntry() {
            for (; ; ) {
                ConcurrentSkipListMap2.Node<K, V> n;
                V v;
                if ((n = hiNode()) == null || !inBounds(n.key))
                    return null;
                else if ((v = n.val) != null)
                    return new AbstractMap.SimpleImmutableEntry<>(n.key, v);
            }
        }

        Map.Entry<K, V> removeLowest() {
            for (; ; ) {
                ConcurrentSkipListMap2.Node<K, V> n;
                K k;
                V v;
                if ((n = loNode()) == null)
                    return null;
                else if (!inBounds(k = n.key))
                    return null;
                else if ((v = m.doRemove(k, null)) != null)
                    return new AbstractMap.SimpleImmutableEntry<>(k, v);
            }
        }

        Map.Entry<K, V> removeHighest() {
            for (; ; ) {
                ConcurrentSkipListMap2.Node<K, V> n;
                K k;
                V v;
                if ((n = hiNode()) == null)
                    return null;
                else if (!inBounds(k = n.key))
                    return null;
                else if ((v = m.doRemove(k, null)) != null)
                    return new AbstractMap.SimpleImmutableEntry<>(k, v);
            }
        }

        /**
         * Submap version of ConcurrentSkipListMap.findNearEntry.
         */
        Map.Entry<K, V> getNearEntry(K key, int rel) {
            if (isDescending) { // adjust relation for direction
                if ((rel & LT) == 0)
                    rel |= LT;
                else
                    rel &= ~LT;
            }
            if (tooLow(key))
                return (rel & LT) != 0 ? null : lowestEntry();
            if (tooHigh(key))
                return (rel & LT) != 0 ? highestEntry() : null;
            var e = m.findNearEntry(key, rel);
            return e == null || !inBounds(e.getKey()) ? null : e;
        }

        // Almost the same as getNearEntry, except for keys
        K getNearKey(K key, int rel) {
            if (isDescending) { // adjust relation for direction
                if ((rel & LT) == 0)
                    rel |= LT;
                else
                    rel &= ~LT;
            }
            if (tooLow(key)) {
                if ((rel & LT) == 0) {
                    ConcurrentSkipListMap2.Node<K, V> n = loNode();
                    if (isBeforeEnd(n))
                        return n.key;
                }
                return null;
            }
            if (tooHigh(key)) {
                if ((rel & LT) != 0) {
                    ConcurrentSkipListMap2.Node<K, V> n = hiNode();
                    if (n != null) {
                        K last = n.key;
                        if (inBounds(last))
                            return last;
                    }
                }
                return null;
            }
            for (; ; ) {
                Node<K, V> n = m.findNear(key, rel);
                if (n == null || !inBounds(n.key))
                    return null;
                if (n.val != null)
                    return n.key;
            }
        }

        /* ----------------  Map API methods -------------- */

        public boolean containsKey(Object key) {
            return inBounds((K) key) && m.containsKey(key);
        }

        public V get(Object key) {
            return inBounds((K) key) ? m.get(key) : null;
        }

        public V put(K key, V value) {
            //checkKeyBounds(key);
            return m.put(key, value);
        }

        public V remove(Object key) {
            return inBounds((K) key) ? m.remove(key) : null;
        }

        public int size() {
            int count = 0;
            for (ConcurrentSkipListMap2.Node<K, V> n = loNode();
                 isBeforeEnd(n);
                 n = n.next) {
                if (n.val != null)
                    ++count;
            }
            //return count >= Integer.MAX_VALUE - 1 ? Integer.MAX_VALUE - 1 : count;
            return Math.min(count, Integer.MAX_VALUE - 1);
        }

        public boolean isEmpty() {
            return !isBeforeEnd(loNode());
        }

        public boolean containsValue(Object value) {
            for (var n = loNode(); isBeforeEnd(n); n = n.next) {
                V v = n.val;
                if (v != null && value.equals(v))
                    return true;
            }
            return false;
        }

        public void clear() {
//            Comparator<K> cmp = m.comparator();
            for (var n = loNode(); isBeforeEnd(n); n = n.next) {
                if (n.val != null)
                    m.remove(n.key);
            }
        }

        /* ----------------  ConcurrentMap API methods -------------- */

        public V putIfAbsent(K key, V value) {
            //checkKeyBounds(key);
            return m.putIfAbsent(key, value);
        }

        public boolean remove(Object key, Object value) {
            return inBounds((K) key) && m.remove(key, value);
        }

        public boolean replace(K key, V oldValue, V newValue) {
            //checkKeyBounds(key);
            return m.replace(key, oldValue, newValue);
        }

        public V replace(K key, V value) {
            //checkKeyBounds(key);
            return m.replace(key, value);
        }

        /* ----------------  SortedMap API methods -------------- */

        public Comparator<? super K> comparator() {
            var cmp = m.comparator();
            return isDescending ? Collections.reverseOrder(cmp) : cmp;
        }

        /**
         * Utility to create submaps, where given bounds override
         * unbounded(null) ones and/or are checked against bounded ones.
         */
        SubMap<K, V> newSubMap(K fromKey, boolean fromInclusive,
                               K toKey, boolean toInclusive) {
            if (isDescending) { // flip senses
                K tk = fromKey;
                fromKey = toKey;
                toKey = tk;
                boolean ti = fromInclusive;
                fromInclusive = toInclusive;
                toInclusive = ti;
            }
            if (lo != null) {
                if (fromKey == null) {
                    fromKey = lo;
                    fromInclusive = loInclusive;
                }// else {
//                    int c = m.cmp(fromKey, lo);
//                    if (c < 0 || c == 0 && !loInclusive && fromInclusive)
//                        throw new IllegalArgumentException("key out of range");
//                }
            }
            if (hi != null) {
                if (toKey == null) {
                    toKey = hi;
                    toInclusive = hiInclusive;
                } //else {
//                    int c = m.cmp(toKey, hi);
//                    if (c > 0 || c == 0 && !hiInclusive && toInclusive)
//                        throw new IllegalArgumentException("key out of range");
//                }
            }
            return new SubMap<>(m, fromKey, fromInclusive,
                    toKey, toInclusive, isDescending);
        }

        public SubMap<K, V> subMap(K fromKey, boolean fromInclusive,
                                   K toKey, boolean toInclusive) {
            return newSubMap(fromKey, fromInclusive, toKey, toInclusive);
        }

        public SubMap<K, V> headMap(K toKey, boolean inclusive) {
            return newSubMap(null, false, toKey, inclusive);
        }

        public SubMap<K, V> tailMap(K fromKey, boolean inclusive) {
            return newSubMap(fromKey, inclusive, null, false);
        }

        public SubMap<K, V> subMap(K fromKey, K toKey) {
            return subMap(fromKey, true, toKey, false);
        }

        public SubMap<K, V> headMap(K toKey) {
            return headMap(toKey, false);
        }

        public SubMap<K, V> tailMap(K fromKey) {
            return tailMap(fromKey, true);
        }

        public SubMap<K, V> descendingMap() {
            return new SubMap<>(m, lo, loInclusive,
                    hi, hiInclusive, !isDescending);
        }

        /* ----------------  Relational methods -------------- */

        public Map.Entry<K, V> ceilingEntry(K key) {
            return getNearEntry(key, GT | EQ);
        }

        public K ceilingKey(K key) {
            return getNearKey(key, GT | EQ);
        }

        public Map.Entry<K, V> lowerEntry(K key) {
            return getNearEntry(key, LT);
        }

        public K lowerKey(K key) {
            return getNearKey(key, LT);
        }

        public Map.Entry<K, V> floorEntry(K key) {
            return getNearEntry(key, LT | EQ);
        }

        public K floorKey(K key) {
            return getNearKey(key, LT | EQ);
        }

        public Map.Entry<K, V> higherEntry(K key) {
            return getNearEntry(key, GT);
        }

        public K higherKey(K key) {
            return getNearKey(key, GT);
        }

        public K firstKey() {
            return isDescending ? highestKey() : lowestKey();
        }

        public K lastKey() {
            return isDescending ? lowestKey() : highestKey();
        }

        public Map.Entry<K, V> firstEntry() {
            return isDescending ? highestEntry() : lowestEntry();
        }

        public Map.Entry<K, V> lastEntry() {
            return isDescending ? lowestEntry() : highestEntry();
        }

        public Map.Entry<K, V> pollFirstEntry() {
            return isDescending ? removeHighest() : removeLowest();
        }

        public Map.Entry<K, V> pollLastEntry() {
            return isDescending ? removeLowest() : removeHighest();
        }

        /* ---------------- Submap Views -------------- */

        public NavigableSet<K> keySet() {
            KeySet<K, V> ks;
            return (ks = keySetView) != null ? ks : (keySetView = new KeySet<>(this));
        }

        public NavigableSet<K> navigableKeySet() {
            KeySet<K, V> ks;
            return (ks = keySetView) != null ? ks : (keySetView = new KeySet<>(this));
        }

        public Collection<V> values() {
            Values<K, V> vs;
            return (vs = valuesView) != null ? vs : (valuesView = new Values<>(this));
        }

        public Set<Map.Entry<K, V>> entrySet() {
            EntrySet<K, V> es;
            return (es = entrySetView) != null ? es : (entrySetView = new EntrySet<>(this));
        }

        public NavigableSet<K> descendingKeySet() {
            return descendingMap().navigableKeySet();
        }

        /**
         * Variant of main Iter class to traverse through submaps.
         * Also serves as back-up Spliterator for views.
         */
        sealed abstract class SubMapIter<T> implements Iterator<T>, Spliterator<T> {
            /**
             * the last node returned by next()
             */
            Node<K, V> lastReturned;
            /**
             * the next node to return from next();
             */
            Node<K, V> next;
            /**
             * Cache of next value field to maintain weak consistency
             */
            V nextValue;

            SubMapIter() {
                VarHandle.acquireFence();
                for (; ; ) {
                    next = isDescending ? hiNode() : loNode();
                    if (next == null)
                        break;
                    V x = next.val;
                    if (x != null) {
                        if (!inBounds(next.key))
                            next = null;
                        else
                            nextValue = x;
                        break;
                    }
                }
            }

            public final boolean hasNext() {
                return next != null;
            }

            final void advance() {
                lastReturned = next;
                this.next = isDescending ? descend() : ascend();
            }

            private Node<K, V> ascend() {
                Node<K, V> next = this.next;
                for (; ; ) {
                    next = next.next;
                    if (next == null)
                        break;
                    V x = next.val;
                    if (x != null) {
                        if (tooHigh(next.key))
                            next = null;
                        else
                            nextValue = x;
                        break;
                    }
                }
                return next;
            }

            private Node<K, V> descend() {
                Node<K, V> next;
                for (; ; ) {
                    if ((next = m.findNear(lastReturned.key, LT)) == null)
                        break;
                    V x = next.val;
                    if (x != null) {
                        if (tooLow(next.key))
                            next = null;
                        else
                            nextValue = x;
                        break;
                    }
                }
                return next;
            }

            public void remove() {
                m.remove(lastReturned.key);
                lastReturned = null;
            }

            public Spliterator<T> trySplit() {
                return null;
            }

            public boolean tryAdvance(Consumer<? super T> action) {
                if (hasNext()) {
                    action.accept(next());
                    return true;
                }
                return false;
            }

            public void forEachRemaining(Consumer<? super T> action) {
                while (hasNext())
                    action.accept(next());
            }

            public long estimateSize() {
                return Long.MAX_VALUE;
            }

        }

        final class SubMapValueIterator extends SubMapIter<V> {
            public V next() {
                V v = nextValue;
                advance();
                return v;
            }

            public int characteristics() {
                return 0;
            }
        }

        final class SubMapKeyIterator extends SubMapIter<K> {
            public K next() {
                Node<K, V> n = next;
                advance();
                return n.key;
            }

            public int characteristics() {
                return DISTINCT | ORDERED |
                        SORTED;
            }

            public Comparator<? super K> getComparator() {
                return SubMap.this.comparator();
            }
        }

        final class SubMapEntryIterator extends SubMapIter<Map.Entry<K, V>> {
            public Map.Entry<K, V> next() {
                Node<K, V> n = next;
                V v = nextValue;
                advance();
                return new AbstractMap.SimpleImmutableEntry<>(n.key, v);
            }

            public int characteristics() {
                return DISTINCT;
            }
        }
    }

    // default Map method overrides

    public void forEach(BiConsumer<? super K, ? super V> action) {
        Node<K, V> b, n;
        V v;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                if ((v = n.val) != null)
                    action.accept(n.key, v);
                b = n;
            }
        }
    }

    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function) {
        Node<K, V> b, n;
        V v;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                while ((v = n.val) != null) {
                    V r = function.apply(n.key, v);
                    if (VAL.compareAndSet(n, v, r))
                        break;
                }
                b = n;
            }
        }
    }

    /**
     * Helper method for EntrySet.removeIf.
     */
    private boolean removeEntryIf(Predicate<? super Map.Entry<K, V>> function) {
        boolean removed = false;
        Node<K, V> b, n;
        V v;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                if ((v = n.val) != null) {
                    if (function.test(new SimpleImmutableEntry<>(n.key, v)) && remove(n.key, v))
                        removed = true;
                }
                b = n;
            }
        }
        return removed;
    }

    /**
     * Helper method for Values.removeIf.
     */
    private boolean removeValueIf(Predicate<? super V> function) {
        boolean removed = false;
        Node<K, V> b, n;
        V v;
        if ((b = baseHead()) != null) {
            while ((n = b.next) != null) {
                if ((v = n.val) != null && function.test(v) && remove(n.key, v))
                    removed = true;
                b = n;
            }
        }
        return removed;
    }

    /**
     * Base class providing common structure for Spliterators.
     * (Although not all that much common functionality; as usual for
     * view classes, details annoyingly vary in key, value, and entry
     * subclasses in ways that are not worth abstracting out for
     * internal classes.)
     * <p>
     * The basic split strategy is to recursively descend from top
     * level, row by row, descending to next row when either split
     * off, or the end of row is encountered. Control of the number of
     * splits relies on some statistical estimation: The expected
     * remaining number of elements of a skip list when advancing
     * either across or down decreases by about 25%.
     */
    sealed private abstract static class CSLMSpliterator<K, V> {
        final Comparator<? super K> comparator;
        final K fence;     // exclusive upper bound for keys, or null if to end
        Index<K, V> row;    // the level to split out
        Node<K, V> current; // current traversal node; initialize at origin
        long est;          // size estimate

        CSLMSpliterator(Comparator<? super K> comparator, Index<K, V> row,
                        Node<K, V> origin, K fence, long est) {
            this.comparator = comparator;
            this.row = row;
            this.current = origin;
            this.fence = fence;
            this.est = est;
        }

        final int cmp(K x, K y) {
            return comparator.compare(x, y);
        }

        public final long estimateSize() {
            return est;
        }
    }

    private static final class KeySpliterator<K, V> extends CSLMSpliterator<K, V>
            implements Spliterator<K> {
        KeySpliterator(Comparator<? super K> comparator, Index<K, V> row,
                       Node<K, V> origin, K fence, long est) {
            super(comparator, row, origin, fence, est);
        }

        public KeySpliterator<K, V> trySplit() {
            Node<K, V> e;
            K ek;
            K f = fence;
            if ((e = current) != null && (ek = e.key) != null) {
                for (Index<K, V> q = row; q != null; q = row = q.down) {
                    Index<K, V> s;
                    Node<K, V> b, n;
                    K sk;
                    if ((s = q.right) != null && (b = s.node) != null &&
                            (n = b.next) != null && n.val != null &&
                            (sk = n.key) != null && cmp(sk, ek) > 0 &&
                            (f == null || cmp(sk, f) < 0)) {
                        current = n;
                        Index<K, V> r = q.down;
                        row = s.right != null ? s : s.down;
                        est -= est >>> 2;
                        return new KeySpliterator<>(comparator, r, e, sk, est);
                    }
                }
            }
            return null;
        }

        public void forEachRemaining(Consumer<? super K> action) {
            K f = fence;
            Node<K, V> e = current;
            current = null;
            for (; e != null; e = e.next) {
                K k;
                if ((k = e.key) != null && f != null && cmp(f, k) <= 0)
                    break;
                if (e.val != null)
                    action.accept(k);
            }
        }

        public boolean tryAdvance(Consumer<? super K> action) {
            K f = fence;
            Node<K, V> e = current;
            for (; e != null; e = e.next) {
                K k;
                if ((k = e.key) != null && f != null && cmp(f, k) <= 0) {
                    //e = null;
                    break;
                }
                if (e.val != null) {
                    current = e.next;
                    action.accept(k);
                    return true;
                }
            }
            current = null /*e*/;
            return false;
        }

        public int characteristics() {
            return DISTINCT | SORTED |
                    ORDERED | CONCURRENT |
                    NONNULL;
        }

        public Comparator<? super K> getComparator() {
            return comparator;
        }
    }

    /**
     * factory method for KeySpliterator
     */
    private KeySpliterator<K, V> keySpliterator() {
        Index<K, V> h;
        Node<K, V> n;
        long est;
        VarHandle.acquireFence();
        if ((h = head) == null) {
            n = null;
            est = 0L;
        } else {
            n = h.node;
            est = getAdderCount();
        }
        return new KeySpliterator<>(this::compare, h, n, null, est);
    }

    private static final class ValueSpliterator<K, V> extends CSLMSpliterator<K, V>
            implements Spliterator<V> {
        ValueSpliterator(Comparator<? super K> comparator, Index<K, V> row,
                         Node<K, V> origin, K fence, long est) {
            super(comparator, row, origin, fence, est);
        }

        public ValueSpliterator<K, V> trySplit() {
            Node<K, V> e;
            K ek;
            K f = fence;
            if ((e = current) != null && (ek = e.key) != null) {
                for (Index<K, V> q = row; q != null; q = row = q.down) {
                    Index<K, V> s;
                    Node<K, V> b, n;
                    K sk;
                    if ((s = q.right) != null && (b = s.node) != null &&
                            (n = b.next) != null && n.val != null &&
                            (sk = n.key) != null && cmp(sk, ek) > 0 &&
                            (f == null || cmp(sk, f) < 0)) {
                        current = n;
                        Index<K, V> r = q.down;
                        row = s.right != null ? s : s.down;
                        est -= est >>> 2;
                        return new ValueSpliterator<>(comparator, r, e, sk, est);
                    }
                }
            }
            return null;
        }

        public void forEachRemaining(Consumer<? super V> action) {
            K f = fence;
            Node<K, V> e = current;
            current = null;
            for (; e != null; e = e.next) {
                K k;
                V v;
                if ((k = e.key) != null && f != null && cmp(f, k) <= 0)
                    break;
                if ((v = e.val) != null)
                    action.accept(v);
            }
        }

        public boolean tryAdvance(Consumer<? super V> action) {
            K f = fence;
            Node<K, V> e = current;
            for (; e != null; e = e.next) {
                K k;
                V v;
                if ((k = e.key) != null && f != null && cmp(f, k) <= 0) {
                    //e = null;
                    break;
                }
                if ((v = e.val) != null) {
                    current = e.next;
                    action.accept(v);
                    return true;
                }
            }
            current = null /*e*/;
            return false;
        }

        public int characteristics() {
            return CONCURRENT | ORDERED |
                    NONNULL;
        }
    }

    // Almost the same as keySpliterator()
    private ValueSpliterator<K, V> valueSpliterator() {
        Index<K, V> h;
        Node<K, V> n;
        long est;
        VarHandle.acquireFence();
        if ((h = head) == null) {
            n = null;
            est = 0;
        } else {
            n = h.node;
            est = getAdderCount();
        }
        return new ValueSpliterator<>(comparator(), h, n, null, est);
    }

    private static final class EntrySpliterator<K, V> extends CSLMSpliterator<K, V>
            implements Spliterator<Map.Entry<K, V>> {
        EntrySpliterator(Comparator<? super K> comparator, Index<K, V> row,
                         Node<K, V> origin, K fence, long est) {
            super(comparator, row, origin, fence, est);
        }

        public EntrySpliterator<K, V> trySplit() {
            Node<K, V> e;
            K ek;
            K f = fence;
            if ((e = current) != null && (ek = e.key) != null) {
                for (Index<K, V> q = row; q != null; q = row = q.down) {
                    Index<K, V> s;
                    Node<K, V> b, n;
                    K sk;
                    if ((s = q.right) != null && (b = s.node) != null &&
                            (n = b.next) != null && n.val != null &&
                            (sk = n.key) != null && cmp(sk, ek) > 0 &&
                            (f == null || cmp(sk, f) < 0)) {
                        current = n;
                        Index<K, V> r = q.down;
                        row = s.right != null ? s : s.down;
                        est -= est >>> 2;
                        return new EntrySpliterator<>(comparator, r, e, sk, est);
                    }
                }
            }
            return null;
        }

        public void forEachRemaining(Consumer<? super Map.Entry<K, V>> action) {
            K f = fence;
            Node<K, V> e = current;
            current = null;
            for (; e != null; e = e.next) {
                K k;
                V v;
                if ((k = e.key) != null && f != null && cmp(f, k) <= 0)
                    break;
                if ((v = e.val) != null) {
                    action.accept(new AbstractMap.SimpleImmutableEntry<>(k, v));
                }
            }
        }

        public boolean tryAdvance(Consumer<? super Map.Entry<K, V>> action) {
            K f = fence;
            Node<K, V> e = current;
            for (; e != null; e = e.next) {
                K k;
                V v;
                if ((k = e.key) != null && f != null && cmp(f, k) <= 0) {
                    //e = null;
                    break;
                }
                if ((v = e.val) != null) {
                    current = e.next;
                    action.accept
                            (new AbstractMap.SimpleImmutableEntry<>(k, v));
                    return true;
                }
            }
            current = null /*e*/;
            return false;
        }

        public int characteristics() {
            return DISTINCT | SORTED |
                    ORDERED | CONCURRENT |
                    NONNULL;
        }

        public Comparator<Map.Entry<K, V>> getComparator() {
            // Adapt or create a key-based comparator
//            if (comparator != null) {
            return Map.Entry.comparingByKey(comparator);
//            }
//            else {
//                return (Comparator<Map.Entry<K,V>> & Serializable) (e1, e2) -> {
//                    @SuppressWarnings("unchecked")
//                    Comparable<K> k1 = (Comparable<K>) e1.getKey();
//                    return k1.compareTo(e2.getKey());
//                };
//            }
        }
    }

    // Almost the same as keySpliterator()
    private EntrySpliterator<K, V> entrySpliterator() {
        Index<K, V> h;
        Node<K, V> n;
        long est;
        VarHandle.acquireFence();
        if ((h = head) == null) {
            n = null;
            est = 0;
        } else {
            n = h.node;
            est = getAdderCount();
        }
        return new EntrySpliterator<>(comparator(), h, n, null, est);
    }

    private static final VarHandle HEAD, ADDER, NEXT, VAL, RIGHT;
    static {
        try {
            MethodHandles.Lookup l = MethodHandles.lookup();
            HEAD = l.findVarHandle(ConcurrentSkipListMap2.class, "head", Index.class);
            ADDER = l.findVarHandle(ConcurrentSkipListMap2.class, "adder", LongAdder.class);
            RIGHT = l.findVarHandle(Index.class, "right", Index.class);
            NEXT = l.findVarHandle(Node.class, "next", Node.class).withInvokeExactBehavior();
            VAL = l.findVarHandle(Node.class, "val", Object.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }
}