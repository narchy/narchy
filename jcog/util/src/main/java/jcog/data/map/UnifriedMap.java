/*
 * Copyright (c) 2018 Goldman Sachs.
 * MODIFIED BY FRACTIONAL RESERVE USURY SCAM
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v. 1.0 which accompany this distribution.
 * The Eclipse Public License is available at http://www.eclipse.org/legal/epl-v10.html
 * and the Eclipse Distribution License is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 */

package jcog.data.map;

import org.eclipse.collections.api.block.function.Function0;
import org.eclipse.collections.api.block.function.Function2;
import org.eclipse.collections.api.block.predicate.Predicate;
import org.eclipse.collections.api.block.predicate.Predicate2;
import org.eclipse.collections.api.block.procedure.Procedure;
import org.eclipse.collections.api.block.procedure.Procedure2;
import org.eclipse.collections.api.block.procedure.primitive.ObjectIntProcedure;
import org.eclipse.collections.api.map.MutableMap;
import org.eclipse.collections.api.map.UnsortedMapIterable;
import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.block.factory.Functions;
import org.eclipse.collections.impl.block.factory.Predicates;
import org.eclipse.collections.impl.block.procedure.MapCollectProcedure;
import org.eclipse.collections.impl.list.mutable.FastList;
import org.eclipse.collections.impl.map.mutable.AbstractMutableMap;
import org.eclipse.collections.impl.parallel.BatchIterable;
import org.eclipse.collections.impl.set.mutable.UnifiedSet;
import org.eclipse.collections.impl.tuple.ImmutableEntry;
import org.eclipse.collections.impl.tuple.Tuples;
import org.eclipse.collections.impl.utility.ArrayIterate;
import org.eclipse.collections.impl.utility.Iterate;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.lang.ref.WeakReference;
import java.lang.reflect.Array;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Function;

/**
 * PATCHED with customizations
 * <p>
 * UnifiedMap stores key/value pairs in a single array, where alternate slots are keys and values. This is nicer to CPU caches as
 * consecutive memory addresses are very cheap to access. Entry objects are not stored in the table like in java.util.HashMap.
 * Instead of trying to deal with collisions in the main array using Entry objects, we put a special object in
 * the key slot and put a regular Object[] in the value slot. The array contains the key value pairs in consecutive slots,
 * just like the main array, but it's a linear list with no hashing.
 * <p>
 * The final result is a Map implementation that's leaner than java.util.HashMap and faster than Trove's THashMap.
 * The best of both approaches unified together, and thus the name UnifiedMap.
 */

@SuppressWarnings("ObjectEquality")
public class UnifriedMap<K, V> extends AbstractMutableMap<K, V>
        implements Externalizable, BatchIterable<V> {


    private static final Object CHAINED_KEY = new Object() {
        @Override
        public boolean equals(Object obj) {
            throw new RuntimeException("Possible corruption through unsynchronized concurrent modification.");
        }

        @Override
        public int hashCode() {
            throw new RuntimeException("Possible corruption through unsynchronized concurrent modification.");
        }

        @Override
        public String toString() {
            return "UnifiedMap.CHAINED_KEY";
        }
    };

    private static final float DEFAULT_LOAD_FACTOR = 0.75f;

    private static final int DEFAULT_INITIAL_CAPACITY = 4;


    private transient Object[] t;

    private transient int size;

    private float loadFactor = DEFAULT_LOAD_FACTOR;

    private int maxSize;

    public UnifriedMap() {
        this.allocate(DEFAULT_INITIAL_CAPACITY);
    }

    public UnifriedMap(int initialCapacity) {
        this(initialCapacity, DEFAULT_LOAD_FACTOR);
    }

    public UnifriedMap(int initialCapacity, float loadFactor) {
        if (initialCapacity < 0) throw new IllegalArgumentException("initial capacity cannot be less than 0");
        if (loadFactor <= 0.0) throw new IllegalArgumentException("load factor cannot be less than or equal to 0");
        if (loadFactor > 1.0) throw new IllegalArgumentException("load factor cannot be greater than 1");

        this.loadFactor = loadFactor;
        this.init(fastCeil(initialCapacity / loadFactor));
    }

    public UnifriedMap(Map<? extends K, ? extends V> map) {
        this(Math.max(map.size(), DEFAULT_INITIAL_CAPACITY), DEFAULT_LOAD_FACTOR);

        this.putAll(map);
    }

    public UnifriedMap(Pair<K, V>[] pairs) {
        this(Math.max(pairs.length, DEFAULT_INITIAL_CAPACITY), DEFAULT_LOAD_FACTOR);
        ArrayIterate.forEach(pairs, new MapCollectProcedure<Pair<K, V>, K, V>(
                this,
                Functions.firstOfPair(),
                Functions.secondOfPair()));
    }

    public static <K, V> UnifriedMap<K, V> newMap() {
        return new UnifriedMap<>();
    }

    public static <K, V> UnifriedMap<K, V> newMap(int size) {
        return new UnifriedMap<>(size);
    }

    public static <K, V> UnifriedMap<K, V> newMap(int size, float loadFactor) {
        return new UnifriedMap<>(size, loadFactor);
    }

    private static int fastCeil(float x) {
        var y = (int) x;
        return x - y > 0.0F ? y + 1 : y;
    }

    private static int chainedHashCode(Object[] chain) {
        var hashCode = 0;
        var cl = chain.length;
        for (var i = 0; i < cl; i += 2) {
            var cur = chain[i];
            if (cur == null) return hashCode;
            var value = chain[i + 1];
            hashCode += cur.hashCode() ^ (value == null ? 0 : value.hashCode());
        }
        return hashCode;
    }


    @Override
    public UnifriedMap<K, V> clone() {
        return new UnifriedMap<>(this);
    }

    @Override
    public MutableMap<K, V> newEmpty() {
        return new UnifriedMap<>();
    }

    @Override
    public MutableMap<K, V> newEmpty(int capacity) {
        return new UnifriedMap<>(capacity, this.loadFactor);
    }

    void init(int initialCapacity) {
        var capacity = 1;
        while (capacity < initialCapacity) capacity <<= 1;

        this.allocate(capacity);
    }

    boolean allocate(int capacity) {

        // the table size is twice the capacity to handle both keys and values
        var next = capacity << 1;
        if (t == null || t.length != next) {
            this.t = new Object[next];
            this.maxSize = Math.min(capacity - 1, (int) (capacity * this.loadFactor));
            return true;
        }

        return false;
    }

    int index(Object key) {
        // This function ensures that hashCodes that differ only by
        // constant multiples at each bit position have a bounded
        // number of collisions (approximately 8 at default load factor).
        var h = key == null ? 0 : key.hashCode();
        h ^= h >>> 20 ^ h >>> 12;
        h ^= h >>> 7 ^ h >>> 4;
        return (h & (this.t.length >> 1) - 1) << 1;
    }

    @Override
    public void clear() {
        if (this.size == 0) return;
        this.size = 0;

        if (!allocate(DEFAULT_INITIAL_CAPACITY))
            Arrays.fill(t, null);
    }

    @Override
    public V put(K key, V value) {
        var index = this.index(key);
        var t = this.t;
        var cur = t[index];
        if (cur == null) {
            rehashGrow(index, key, value, t);
            return null;
        } else if (cur != CHAINED_KEY && cur.equals(key)) {
            var result = t[index + 1];
            t[index + 1] = value;
            return (V)result;
        } else
            return this.chainedPut(key, index, value);
    }

    private V chainedPut(K key, int index, V value) {
        if (this.t[index] == CHAINED_KEY) {
            var chain = (Object[]) this.t[index + 1];
            var cl = chain.length;
            for (var i = 0; i < cl; i += 2) {
                if (chain[i] == null) {
                    rehashGrow(i, key, value, chain);
                    return null;
                } else if (chain[i].equals(key)) {
                    var result = (V) chain[i + 1];
                    chain[i + 1] = value;
                    return result;
                }
            }
            var newChain = new Object[cl + 4];
            System.arraycopy(chain, 0, newChain, 0, cl);
            this.t[index + 1] = newChain;
            rehashGrow(cl, key, value, newChain);
        } else {
            chainingGrow(key, index, value);
        }
        return null;
    }

    private void rehashGrow() {
        if (++this.size > this.maxSize) {
            var tPrev = this.t;
            allocate(tPrev.length);
            copyFrom(tPrev);
        }
    }

    private void copyFrom(Object[] t) {
        this.size = 0;
        int oldLength = t.length;
        for (var i = 0; i < oldLength; i += 2)
            rehashGrow(t, i);
    }

    private void rehashGrow(Object[] old, int i) {
        var cur = old[i];
        if (cur == CHAINED_KEY) {
            var chain = (Object[]) old[i + 1];
            var cl = chain.length;
            for (var j = 0; j < cl; j += 2)
                if (chain[j] != null)
                    this.put((K) chain[j], (V) chain[j + 1]);
        } else
            if (cur != null)
                this.put((K) cur, (V) old[i + 1]);
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public V updateValue(K key, Function0<? extends V> factory, Function<? super V, ? extends V> function) {
        var index = this.index(key);
        var t = this.t;
        var cur = t[index];
        if (cur == null) {
            var result = function.apply(factory.value());
            t[index] = key;
            t[index + 1] = result;
            ++this.size;
            return result;
        }
        if (cur != CHAINED_KEY && cur.equals(key)) {
            var newValue = function.apply((V) t[index + 1]);
            t[index + 1] = newValue;
            return newValue;
        }
        return this.chainedUpdateValue(key, index, factory, function);
    }

    private V chainedUpdateValue(K key, int index, Function0<? extends V> factory, Function<? super V, ? extends V> function) {
        if (this.t[index] == CHAINED_KEY) {
            var chain = (Object[]) this.t[index + 1];
            for (var i = 0; i < chain.length; i += 2) {
                var ci = chain[i];
                if (ci == null) {
                    var result = function.apply(factory.value());
                    rehashGrow(i, key, result, chain);
                    return result;
                }
                if (ci.equals(key)) {
                    var result = function.apply((V) chain[i + 1]);
                    chain[i + 1] = result;
                    return result;
                }
            }
            var newChain = new Object[chain.length + 4];
            System.arraycopy(chain, 0, newChain, 0, chain.length);
            this.t[index + 1] = newChain;
            newChain[chain.length] = key;
            var result = function.apply(factory.value());
            newChain[chain.length + 1] = result;
            rehashGrow();
            return result;
        }
        var result = function.apply(factory.value());
        chainingGrow(key, index, result);
        return result;
    }

    @Override
    public <P> V updateValueWith(K key, Function0<? extends V> factory, Function2<? super V, ? super P, ? extends V> function, P parameter) {
        var index = this.index(key);
        var t = this.t;
        var cur = t[index];
        if (cur == null) {
            var result = function.value(factory.value(), parameter);
            t[index] = key;
            t[index + 1] = result;
            ++this.size;
            return result;
        }
        if (cur != CHAINED_KEY && cur.equals(key)) {
            var newValue = function.value((V) t[index + 1], parameter);
            t[index + 1] = newValue;
            return newValue;
        }
        return this.chainedUpdateValueWith(key, index, factory, function, parameter);
    }

    private <P> V chainedUpdateValueWith(
            K key,
            int index,
            Function0<? extends V> factory,
            Function2<? super V, ? super P, ? extends V> function,
            P parameter) {
        if (this.t[index] == CHAINED_KEY) {
            var chain = (Object[]) this.t[index + 1];
            var cl = chain.length;
            for (var i = 0; i < cl; i += 2) {
                if (chain[i] == null) {
                    var result = function.value(factory.value(), parameter);
                    rehashGrow(i, key, result, chain);
                    return result;
                }
                if (chain[i].equals(key)) {
                    var result = function.value((V) chain[i + 1], parameter);
                    chain[i + 1] = result;
                    return result;
                }
            }
            var newChain = new Object[cl + 4];
            System.arraycopy(chain, 0, newChain, 0, cl);
            this.t[index + 1] = newChain;
            newChain[cl] = key;
            var result = function.value(factory.value(), parameter);
            newChain[cl + 1] = result;
            rehashGrow();
            return result;
        }

        var result = function.value(factory.value(), parameter);
        chainingGrow(key, index, result);
        return result;
    }


    @Override
    public final V computeIfAbsent(K x, Function<? super K, ? extends V> f) {
        var index = this.index(x);
        var t = this.t;
        var cur = t[index];
        if (cur == null) {
            return compute(x, f, index, t);
        } else {
            return (V) (cur != CHAINED_KEY && cur.equals(x) ?
                t[index + 1] :
                this.chainedGetIfAbsentPutWith(x, index, f, x));
        }
    }

    private V compute(K x, Function<? super K, ? extends V> f, int index, Object[] t) {
        var y = f.apply(x);
        rehashGrow(index, x, y, t);
        return y;
    }

    private void rehashGrow(int index, K k, Object/*V*/ v, Object[] t) {
        t[index] = k;
        t[index + 1] = v;
        rehashGrow();
    }


    @Override
    public V getIfAbsentPut(K key, Function0<? extends V> function) {
        var index = this.index(key);
        var cur = this.t[index];

        if (cur == null) {
            var result = function.value();
            rehashGrow(index, key, result, this.t);
            return result;
        }
        return cur != CHAINED_KEY && cur.equals(key) ?
                (V) this.t[index + 1] :
                this.chainedGetIfAbsentPut(key, index, function);
    }

    private V chainedGetIfAbsentPut(K key, int index, Function0<? extends V> function) {
        V result = null;
        if (this.t[index] == CHAINED_KEY) {
            var chain = (Object[]) this.t[index + 1];
            var i = 0;
            for (; i < chain.length; i += 2) {
                if (chain[i] == null) {
                    result = function.value();
                    rehashGrow(i, key, result, chain);
                    break;
                }
                if (chain[i].equals(key)) {
                    result = (V) chain[i + 1];
                    break;
                }
            }
            if (i == chain.length) {
                result = function.value();
                var newChain = new Object[chain.length + 4];
                System.arraycopy(chain, 0, newChain, 0, chain.length);
                newChain[i] = key;
                newChain[i + 1] = result;
                this.t[index + 1] = newChain;
                rehashGrow();
            }
        } else {
            result = function.value();
            chainingGrow(key, index, result);
        }
        return result;
    }

    private void chainingGrow(Object key, int index, Object/*V*/ result) {
        chaining(t, key, index, result);
        rehashGrow();
    }

    private static void chaining(Object[] t, Object/*K*/ key, int index, Object/*V*/ value) {
        var a = t[index];
        t[index] = CHAINED_KEY;
        t[index + 1] = new Object[] {
            a, /*b*/t[index + 1], key, value
        };
    }

    @Nullable
    @Override
    public V putIfAbsent(K key, V value) {
        return putIfAbsent(key, value, false);
    }

    @Override
    public V getIfAbsentPut(K key, V value) {
        return putIfAbsent(key, value, true);
    }

    private V putIfAbsent(K key, V value, boolean get) {
        var index = index(key);
        var t = this.t;
        var cur = t[index];
        if (cur == null) {
            rehashGrow(index, key, value, t);
            return get ? value : null;
        } else {
            return cur != CHAINED_KEY && cur.equals(key) ?
                    (V) t[index + 1] :
                    chainedGetIfAbsentPut(key, index, value);
        }
    }

    private V chainedGetIfAbsentPut(K key, int index, V value) {
        var result = value;
        if (this.t[index] == CHAINED_KEY) {
            var chain = (Object[]) this.t[index + 1];
            var i = 0;
            var l = chain.length;
            for (; i < l; i += 2) {
                if (chain[i] == null) {
                    rehashGrow(i, key, value, chain);
                    break;
                }
                if (chain[i].equals(key)) {
                    result = (V) chain[i + 1];
                    break;
                }
            }
            if (i == l) {
                var newChain = new Object[l + 4];
                System.arraycopy(chain, 0, newChain, 0, l);
                newChain[i] = key;
                newChain[i + 1] = value;
                t[index + 1] = newChain;
                rehashGrow();
            }
        } else {
            chainingGrow(key, index, value);
        }
        return result;
    }


    private <P> Object chainedGetIfAbsentPutWith(K key, int index, Function<? super P, ? extends V> function, P parameter) {
        Object result = null;
        if (this.t[index] == CHAINED_KEY) {
            var chain = (Object[]) this.t[index + 1];
            var i = 0;
            var l = chain.length;
            for (; i < l; i += 2) {
                var ci = chain[i];
                if (ci == null) {
                    rehashGrow(i, key, result = function.apply(parameter), chain);
                    break;
                } else if (ci.equals(key)) {
                    result = chain[i + 1];
                    break;
                }
            }
            if (i == l) {
                result = function.apply(parameter);
                var newChain = new Object[l + 4];
                System.arraycopy(chain, 0, newChain, 0, l);
                newChain[i] = key;
                newChain[i + 1] = result;
                this.t[index + 1] = newChain;
                rehashGrow();
            }
        } else {
            chainingGrow(key, index, result = function.apply(parameter));
        }
        return result;
    }

    @Override
    public final V get(Object key) {
        var t = this.t;
        var index = this.index(key);
        var cur = t[index];
        if (cur != null) {
            var val = t[index + 1];
            if (cur == CHAINED_KEY) return (V) getFromChain((Object[]) val, key);
            if (cur.equals(key)) return (V)val;
        }
        return null;
    }

    @Nullable static private Object getFromChain(Object[] chain, Object key) {
        var l = chain.length;
        for (var i = 0; i < l; i += 2) {
            var k = chain[i];
            if (k == null) return null;
            if (k.equals(key)) return chain[i + 1];
        }
        return null;
    }

    @Override
    public boolean containsKey(Object key) {
        var t = this.t;
        var index = this.index(key);
        var x = t[index];
        if (x == null)
            return false;
        else if (x != CHAINED_KEY)
            return x.equals(key);
        else
            return getFromChain((Object[]) t[index + 1], key)!=null;
    }


    @Override
    public boolean containsValue(Object value) {
        var t = this.t;
        var l = t.length;
        for (var i = 0; i < l; i += 2) {
            var ti = t[i];
            if (ti == CHAINED_KEY) {
                if (getFromChain((Object[]) t[i + 1], value)!=null)
                    return true;
            } else if (ti != null && value.equals(t[i + 1]))
                return true;
        }
        return false;
    }



    @Override
    public void forEach(BiConsumer<? super K, ? super V> procedure) {
        var t = this.t;
        var l = t.length;
        for (var i = 0; i < l; i += 2) {
            var cur = t[i];
            if (cur == CHAINED_KEY) chainedForEachEntry((Object[]) t[i + 1], procedure);
            else if (cur != null) procedure.accept((K) cur, (V) t[i + 1]);
        }
    }

    @Override
    public final void forEachKeyValue(Procedure2<? super K,? super V> procedure) {
        forEach(procedure);
    }

    @Override
    public V getFirst() {
        var t = this.t;
        var l = t.length;
        for (var i = 0; i < l; i += 2) {
            var cur = t[i];
            if (cur == CHAINED_KEY)
                return (V) ((Object[]) t[i + 1])[1];
            else if (cur != null)
                return (V) t[i + 1];
        }
        return null;
    }

    @Override
    public <E> MutableMap<K, V> collectKeysAndValues(
            Iterable<E> iterable,
            org.eclipse.collections.api.block.function.Function<? super E, ? extends K> keyFunction,
            org.eclipse.collections.api.block.function.Function<? super E, ? extends V> valueFunction) {
        Iterate.forEach(iterable, new MapCollectProcedure<>(this, keyFunction, valueFunction));
        return this;
    }

    @Override
    public V removeKey(K key) {
        return this.remove(key);
    }

    @Override
    public boolean removeIf(Predicate2<? super K, ? super V> predicate) {
        var previousOccupied = this.size;
        for (var index = 0; index < this.t.length; index += 2) {
            var cur = this.t[index];
            if (cur == CHAINED_KEY) {
                var chain = (Object[]) this.t[index + 1];
                for (var chIndex = 0; chIndex < chain.length; ) {
                    if (chain[chIndex] == null) break;
                    var key = (K) chain[chIndex];
                    var value = (V) chain[chIndex + 1];
                    if (predicate.accept(key, value))
                        this.overwriteWithLastElementFromChain(chain, index, chIndex);
                    else chIndex += 2;
                }
            } else if (cur!=null) {
                var key = (K) cur;
                var value = (V) this.t[index + 1];
                if (predicate.accept(key, value)) {
                    this.t[index] = null;
                    this.t[index + 1] = null;
                    this.size--;
                }
            }
        }
        return previousOccupied > this.size;
    }

    private static void chainedForEachEntry(Object[] chain, BiConsumer procedure) {
        var l = chain.length;
        for (var i = 0; i < l; i += 2) {
            var cur = chain[i];
            if (cur == null) return;
            procedure.accept(cur, chain[i + 1]);
        }
    }

    @Override
    public int getBatchCount(int batchSize) {
        return Math.max(1, this.t.length / 2 / batchSize);
    }

    @Override
    public void batchForEach(Procedure<? super V> procedure, int sectionIndex, int sectionCount) {
        var sectionSize = this.t.length / sectionCount;
        var start = sectionIndex * sectionSize;
        var end = sectionIndex == sectionCount - 1 ? this.t.length : start + sectionSize;
        if (start % 2 == 0) start++;
        for (var i = start; i < end; i += 2) {
            var value = this.t[i];
            if (value instanceof Object[] values) this.chainedForEachValue(values, procedure);
            else if (value != null || this.t[i - 1] != null) procedure.value((V) value);
        }
    }

    @Override
    public void forEachKey(Procedure<? super K> procedure) {
        for (var i = 0; i < this.t.length; i += 2) {
            var cur = this.t[i];
            if (cur == CHAINED_KEY) this.chainedForEachKey((Object[]) this.t[i + 1], procedure);
            else if (cur != null) procedure.value((K) cur);
        }
    }

    private void chainedForEachKey(Object[] chain, Procedure<? super K> procedure) {
        for (var i = 0; i < chain.length; i += 2) {
            var cur = chain[i];
            if (cur == null) return;
            procedure.value((K) cur);
        }
    }

    @Override
    public void forEachValue(Procedure<? super V> procedure) {
        for (var i = 0; i < this.t.length; i += 2) {
            var cur = this.t[i];
            if (cur == CHAINED_KEY) this.chainedForEachValue((Object[]) this.t[i + 1], procedure);
            else if (cur != null) procedure.value((V) this.t[i + 1]);
        }
    }

    private void chainedForEachValue(Object[] chain, Procedure procedure) {
        for (var i = 0; i < chain.length; i += 2) {
            var cur = chain[i];
            if (cur == null) return;
            procedure.value(chain[i + 1]);
        }
    }

    @Override
    public boolean isEmpty() {
        return this.size == 0;
    }

    @Override
    public void putAll(Map<? extends K, ? extends V> map) {
        if (map instanceof UnifriedMap M) this.copyMap(M);
        else if (map instanceof UnsortedMapIterable I) {
            ((UnsortedMapIterable<K, V>) map).forEachKeyValue(this::put);
        } else for (var entry : map.entrySet()) this.put(entry.getKey(), entry.getValue());
    }

    void copyMap(UnifriedMap<K, V> unifiedMap) {
        var t = unifiedMap.t;
        for (var i = 0; i < t.length; i += 2) {
            var cur = t[i];
            if (cur == CHAINED_KEY) this.copyChain((Object[]) t[i + 1]);
            else if (cur != null) this.put((K) cur, (V) t[i + 1]);
        }
    }

    private void copyChain(Object[] chain) {
        var cl = chain.length;
        for (var j = 0; j < cl; j += 2) {
            var cur = chain[j];
            if (cur == null) break;
            this.put((K) cur, (V) chain[j + 1]);
        }
    }

    @Override
    public V remove(Object key) {
        var index = this.index(key);
        var t = this.t;
        var cur = t[index];
        if (cur != null) {
            var val = t[index + 1];
            if (cur == CHAINED_KEY)
                return this.removeFromChain((Object[]) val, (K) key, index);
            if (cur.equals(key)) {
                t[index] = null;
                t[index + 1] = null;
                this.size--;
                return (V) val;
            }
        }
        return null;
    }

    private V removeFromChain(Object[] chain, K key, int index) {
        var cl = chain.length;
        for (var i = 0; i < cl; i += 2) {
            var k = chain[i];
            if (k == null) return null;
            else if (k.equals(key)) {
                var val = chain[i + 1];
                this.overwriteWithLastElementFromChain(chain, index, i);
                return (V)val;
            }
        }
        return null;
    }

    private void overwriteWithLastElementFromChain(Object[] chain, int index, int i) {
        var j = chain.length - 2;
        for (; j > i; j -= 2)
            if (chain[j] != null) {
                chain[i] = chain[j];
                chain[i + 1] = chain[j + 1];
                break;
            }
        chain[j] = null;
        chain[j + 1] = null;
        if (j == 0) {
            t[index] = null;
            t[index + 1] = null;
        }
        this.size--;
    }

    @Override
    public final int size() {
        return this.size;
    }

    @Override
    public Set<Entry<K, V>> entrySet() {
        return new EntrySet();
    }

    @Override
    public Set<K> keySet() {
        return new KeySet();
    }

    @Override
    public Collection<V> values() {
        return new ValuesCollection();
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;

        if (!(object instanceof Map<?, ?> other)) return false;

        if (this.size() != other.size()) return false;

        var t = this.t;
        for (var i = 0; i < t.length; i += 2) {
            var cur = t[i];
            if (cur == CHAINED_KEY) {
                if (!chainedEquals((Object[]) t[i + 1], other)) return false;
            } else if (cur != null) {
                var otherValue = other.get(cur);
                if (otherValue==null || !otherValue.equals(t[i + 1]))
                    return false;
            }
        }

        return true;
    }

    private static boolean chainedEquals(Object[] chain, Map<?, ?> other) {
        var n = chain.length;
        for (var i = 0; i < n; i += 2) {
            var cur = chain[i];
            if (cur == null) return true;
            else if (!other.get(cur).equals(chain[i + 1]))
                return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        var hashCode = 0;
        for (var i = 0; i < this.t.length; i += 2) {
            var cur = this.t[i];
            if (cur == CHAINED_KEY) hashCode += chainedHashCode((Object[]) this.t[i + 1]);
            else if (cur != null) {
                var value = this.t[i + 1];
                hashCode += cur.hashCode() ^ (value == null ? 0 : value.hashCode());
            }
        }
        return hashCode;
    }

    @Override
    public String toString() {
        var b = new StringBuilder();
        b.append('{');
        this.forEachKeyValue(new Procedure2<>() {
            private boolean first = true;

            public void value(K key, V value) {
                if (this.first) this.first = false;
                else b.append(", ");
                b.append(key == UnifriedMap.this ? "(this Map)" : key).append('=').append(value == UnifriedMap.this ? "(this Map)" : value);
            }
        });

        return b.append('}').toString();
    }

    public boolean trimToSize() {
        if (this.t.length <= fastCeil(this.size / this.loadFactor) << 2)
            return false;

        var temp = this.t;
        this.init(fastCeil(this.size / this.loadFactor));
        if (this.isEmpty())
            return true;

        var mask = this.t.length - 1;
        for (var j = 0; j < temp.length; j += 2) {
            var key = temp[j];
            if (key == CHAINED_KEY) {
                var chain = (Object[]) temp[j + 1];
                for (var i = 0; i < chain.length; i += 2) {
                    var cur = chain[i];
                    if (cur != null)
                        this.putForTrim(cur, chain[i + 1], j, mask);
                }
            } else if (key != null)
                this.putForTrim(key, temp[j + 1], j, mask);
        }
        return true;
    }

    private void putForTrim(Object key, Object value, int oldIndex, int mask) {
        var index = oldIndex & mask;
        var t = this.t;
        var cur = t[index];
        if (cur == null) {
            t[index] = key;
            t[index + 1] = value;
            return;
        }
        this.chainedPutForTrim(t, key, index, value);
    }

    private void chainedPutForTrim(Object[] t, Object key, int index, Object value) {
        if (t[index] != CHAINED_KEY) {
            chaining(t, key, index, value);
        } else {
            var chain = (Object[]) t[index + 1];
            var chainLen = chain.length;
            for (var i = 0; i < chainLen; i += 2)
                if (chain[i] == null) {
                    chain[i] = key;
                    chain[i + 1] = value;
                    return;
                }
            var newChain = new Object[chainLen + 4];
            System.arraycopy(chain, 0, newChain, 0, chainLen);
            newChain[chainLen] = key;
            newChain[chainLen + 1] = value;
            this.t[index + 1] = newChain;
        }
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        var size = in.readInt();
        this.loadFactor = in.readFloat();
        this.init(Math.max(
                (int) (size / this.loadFactor) + 1,
                DEFAULT_INITIAL_CAPACITY));
        for (var i = 0; i < size; i++) this.put((K) in.readObject(), (V) in.readObject());
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(this.size());
        out.writeFloat(this.loadFactor);
        for (var i = 0; i < this.t.length; i += 2) {
            var o = this.t[i];
            if (o != null) if (o == CHAINED_KEY) writeExternalChain(out, (Object[]) this.t[i + 1]);
            else {
                out.writeObject(o);
                out.writeObject(this.t[i + 1]);
            }
        }
    }

    private static void writeExternalChain(ObjectOutput out, Object[] chain) throws IOException {
        for (var i = 0; i < chain.length; i += 2) {
            var cur = chain[i];
            if (cur == null) return;
            out.writeObject(cur);
            out.writeObject(chain[i + 1]);
        }
    }

    @Override
    public void forEachWithIndex(ObjectIntProcedure<? super V> objectIntProcedure) {
        var index = 0;
        for (var i = 0; i < this.t.length; i += 2) {
            var cur = this.t[i];
            if (cur == CHAINED_KEY)
                index = this.chainedForEachValueWithIndex((Object[]) this.t[i + 1], objectIntProcedure, index);
            else if (cur != null) objectIntProcedure.value((V) this.t[i + 1], index++);
        }
    }

    private int chainedForEachValueWithIndex(Object[] chain, ObjectIntProcedure<? super V> objectIntProcedure, int index) {
        for (var i = 0; i < chain.length; i += 2) {
            var cur = chain[i];
            if (cur == null) return index;
            objectIntProcedure.value((V) chain[i + 1], index++);
        }
        return index;
    }

    @Override
    public <P> void forEachWith(Procedure2<? super V, ? super P> procedure, P parameter) {
        for (var i = 0; i < this.t.length; i += 2) {
            var cur = this.t[i];
            if (cur == CHAINED_KEY) this.chainedForEachValueWith((Object[]) this.t[i + 1], procedure, parameter);
            else if (cur != null) procedure.value((V) this.t[i + 1], parameter);
        }
    }

    private <P> void chainedForEachValueWith(
            Object[] chain,
            Procedure2<? super V, ? super P> procedure,
            P parameter) {
        for (var i = 0; i < chain.length; i += 2) {
            var cur = chain[i];
            if (cur == null) return;
            procedure.value((V) chain[i + 1], parameter);
        }
    }

    @Override
    public <R> MutableMap<K, R> collectValues(Function2<? super K, ? super V, ? extends R> function) {
        var target = (UnifriedMap<K, R>) this.newEmpty();
        target.loadFactor = this.loadFactor;
        target.size = this.size;
        target.allocate(this.t.length >> 1);

        for (var i = 0; i < target.t.length; i += 2) {
            target.t[i] = this.t[i];

            if (this.t[i] == CHAINED_KEY) {
                var chainedTable = (Object[]) this.t[i + 1];
                var chainedTargetTable = new Object[chainedTable.length];
                for (var j = 0; j < chainedTargetTable.length; j += 2)
                    if (chainedTable[j] != null) {
                        chainedTargetTable[j] = chainedTable[j];
                        chainedTargetTable[j + 1] = function.value((K) chainedTable[j], (V) chainedTable[j + 1]);
                    }
                target.t[i + 1] = chainedTargetTable;
            } else if (this.t[i] != null)
                target.t[i + 1] = function.value((K) this.t[i], (V) this.t[i + 1]);
        }

        return target;
    }

    @Override
    public Pair<K, V> detect(Predicate2<? super K, ? super V> predicate) {
        for (var i = 0; i < this.t.length; i += 2)
            if (this.t[i] == CHAINED_KEY) {
                var chainedTable = (Object[]) this.t[i + 1];
                for (var j = 0; j < chainedTable.length; j += 2)
                    if (chainedTable[j] != null) {
                        var key = (K) chainedTable[j];
                        var value = (V) chainedTable[j + 1];
                        if (predicate.accept(key, value)) return Tuples.pair(key, value);
                    }
            } else if (this.t[i] != null) {
                var key = (K) this.t[i];
                var value = (V) this.t[i + 1];

                if (predicate.accept(key, value)) return Tuples.pair(key, value);
            }

        return null;
    }



    private boolean shortCircuit(
            Predicate/*<? super V>*/ predicate,
            boolean expected,
            boolean onShortCircuit,
            boolean atEnd) {
        var t = this.t;
        for (var i = 0; i < t.length; i += 2) {
            if (t[i] == CHAINED_KEY) {
                var chainedTable = (Object[]) t[i + 1];
                final var ctl = chainedTable.length;
                for (var j = 0; j < ctl; j += 2)
                    if (chainedTable[j] != null) {
                        if (predicate.accept(chainedTable[j + 1]) == expected)
                            return onShortCircuit;
                    }
            } else if (t[i] != null) {
                if (predicate.accept(t[i + 1]) == expected)
                    return onShortCircuit;
            }
        }

        return atEnd;
    }

//    private <P> boolean shortCircuitWith(
//            BiPredicate<? super V, ? super P> predicate,
//            P parameter,
//            boolean expected,
//            boolean onShortCircuit,
//            boolean atEnd) {
//        Object[] t = this.t;
//        for (int i = 0; i < t.length; i += 2)
//            if (t[i] == CHAINED_KEY) {
//                Object[] chainedTable = (Object[]) t[i + 1];
//                for (int j = 0; j < chainedTable.length; j += 2)
//                    if (chainedTable[j] != null) {
//                        if (predicate.test((V) chainedTable[j + 1], parameter) == expected) return onShortCircuit;
//                    }
//            } else if (t[i] != null) {
//                if (predicate.test((V) t[i + 1], parameter) == expected) return onShortCircuit;
//            }
//
//        return atEnd;
//    }

    @Override
    public boolean anySatisfy(Predicate<? super V> predicate) {
        return this.shortCircuit(predicate, true, true, false);
    }

//    @Override
//    public <P> boolean anySatisfyWith(Predicate2<? super V, ? super P> predicate, P parameter) {
//        return this.shortCircuitWith(predicate, parameter, true, true, false);
//    }
//
//    @Override
//    public boolean allSatisfy(Predicate<? super V> predicate) {
//        return this.shortCircuit(predicate, false, false, true);
//    }
//
//    @Override
//    public <P> boolean allSatisfyWith(Predicate2<? super V, ? super P> predicate, P parameter) {
//        return this.shortCircuitWith(predicate, parameter, false, false, true);
//    }

    @Override
    public boolean noneSatisfy(Predicate<? super V> predicate) {
        return this.shortCircuit(predicate, true, false, true);
    }
//
//    @Override
//    public <P> boolean noneSatisfyWith(Predicate2<? super V, ? super P> predicate, P parameter) {
//        return this.shortCircuitWith(predicate, parameter, true, false, true);
//    }

    public void delete() {
        this.t = null;
        this.size = -1;
    }

    static class WeakBoundEntry<K, V> implements Map.Entry<K, V> {
        final K key;
        final WeakReference<UnifriedMap<K, V>> holder;
        V value;

        WeakBoundEntry(K key, V value, WeakReference<UnifriedMap<K, V>> holder) {
            this.key = key;
            this.value = value;
            this.holder = holder;
        }

        @Override
        public K getKey() {
            return this.key;
        }

        @Override
        public V getValue() {
            return this.value;
        }

        @Override
        public V setValue(V value) {
            this.value = value;
            var map = this.holder.get();
            if (map != null && map.containsKey(this.key)) return map.put(this.key, value);
            return null;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof Entry other) {
                var otherKey = (K) other.getKey();
                var otherValue = (V) other.getValue();
                return this.key.equals(otherKey)
                        && this.value.equals(otherValue);
            }
            return false;
        }

        @Override
        public int hashCode() {
            return (this.key == null ? 0 : this.key.hashCode())
                    ^ (this.value == null ? 0 : this.value.hashCode());
        }

        @Override
        public String toString() {
            return this.key + "=" + this.value;
        }
    }

    protected class KeySet implements Set<K>, Serializable, BatchIterable<K> {
//        private static final long serialVersionUID = 1L;

        @Override
        public boolean add(K key) {
            throw new UnsupportedOperationException("Cannot call add() on " + this.getClass().getSimpleName());
        }

        @Override
        public boolean addAll(Collection<? extends K> collection) {
            throw new UnsupportedOperationException("Cannot call addAll() on " + this.getClass().getSimpleName());
        }

        @Override
        public void clear() {
            UnifriedMap.this.clear();
        }

        @Override
        public boolean contains(Object o) {
            return UnifriedMap.this.containsKey(o);
        }

        @Override
        public boolean containsAll(Collection<?> collection) {
            for (var aCollection : collection) if (!UnifriedMap.this.containsKey(aCollection)) return false;
            return true;
        }

        @Override
        public boolean isEmpty() {
            return UnifriedMap.this.isEmpty();
        }

        @Override
        public Iterator<K> iterator() {
            return new KeySetIterator();
        }

        @Override
        public boolean remove(Object key) {
            var oldSize = UnifriedMap.this.size;
            UnifriedMap.this.remove(key);
            return UnifriedMap.this.size != oldSize;
        }

        @Override
        public boolean removeAll(Collection<?> collection) {
            var oldSize = UnifriedMap.this.size;
            for (var object : collection) UnifriedMap.this.remove(object);
            return oldSize != UnifriedMap.this.size;
        }

        void putIfFound(Object key, Map<K, V> other) {
            var index = UnifriedMap.this.index(key);
            var cur = UnifriedMap.this.t[index];
            if (cur != null) {
                var val = UnifriedMap.this.t[index + 1];
                if (cur == CHAINED_KEY) {
                    this.putIfFoundFromChain((Object[]) val, (K) key, other);
                    return;
                }
                if (cur.equals(key))
                    other.put((K) cur, (V) val);
            }
        }

        private void putIfFoundFromChain(Object[] chain, K key, Map<K, V> other) {
            for (var i = 0; i < chain.length; i += 2) {
                var k = chain[i];
                if (k == null) return;
                if (k.equals(key))
                    other.put((K) k, (V) chain[i + 1]);
            }
        }

        @Override
        public boolean retainAll(Collection<?> collection) {
            var retainedSize = collection.size();
            var retainedCopy = (UnifriedMap<K, V>) UnifriedMap.this.newEmpty(retainedSize);
            for (var key : collection) this.putIfFound(key, retainedCopy);
            if (retainedCopy.size() < this.size()) {
                UnifriedMap.this.maxSize = retainedCopy.maxSize;
                UnifriedMap.this.size = retainedCopy.size;
                UnifriedMap.this.t = retainedCopy.t;
                return true;
            }
            return false;
        }

        @Override
        public int size() {
            return UnifriedMap.this.size();
        }

        @Override
        public void forEach(Procedure<? super K> procedure) {
            UnifriedMap.this.forEachKey(procedure);
        }

        @Override
        public int getBatchCount(int batchSize) {
            return UnifriedMap.this.getBatchCount(batchSize);
        }

        @Override
        public void batchForEach(Procedure<? super K> procedure, int sectionIndex, int sectionCount) {
            var map = UnifriedMap.this.t;
            var sectionSize = map.length / sectionCount;
            var start = sectionIndex * sectionSize;
            var end = sectionIndex == sectionCount - 1 ? map.length : start + sectionSize;
            if (start % 2 != 0) start++;
            for (var i = start; i < end; i += 2) {
                var cur = map[i];
                if (cur == CHAINED_KEY) UnifriedMap.this.chainedForEachKey((Object[]) map[i + 1], procedure);
                else if (cur != null) procedure.value((K) cur);
            }
        }

        void copyKeys(Object[] result) {
            var table = UnifriedMap.this.t;
            var count = 0;
            for (var i = 0; i < table.length; i += 2) {
                var x = table[i];
                if (x != null) if (x == CHAINED_KEY) {
                    var chain = (Object[]) table[i + 1];
                    for (var j = 0; j < chain.length; j += 2) {
                        var cur = chain[j];
                        if (cur == null) break;
                        result[count++] = cur;
                    }
                } else result[count++] = x;
            }
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof Set<?> other) {
                if (other.size() == this.size()) return this.containsAll(other);
            }
            return false;
        }

        @Override
        public int hashCode() {
            var hashCode = 0;
            var table = UnifriedMap.this.t;
            for (var i = 0; i < table.length; i += 2) {
                var x = table[i];
                if (x != null) if (x == CHAINED_KEY) {
                    var chain = (Object[]) table[i + 1];
                    for (var j = 0; j < chain.length; j += 2) {
                        var cur = chain[j];
                        if (cur == null) break;
                        hashCode += cur.hashCode();
                    }
                } else hashCode += x.hashCode();
            }
            return hashCode;
        }

        @Override
        public String toString() {
            return Iterate.makeString(this, "[", ", ", "]");
        }

        @Override
        public Object[] toArray() {
            var size = UnifriedMap.this.size();
            var result = new Object[size];
            this.copyKeys(result);
            return result;
        }

        @Override
        public <T> T[] toArray(T[] result) {
            var size = UnifriedMap.this.size();
            if (result.length < size) result = (T[]) Array.newInstance(result.getClass().getComponentType(), size);
            this.copyKeys(result);
            if (size < result.length) result[size] = null;
            return result;
        }

        protected Object writeReplace() {
            UnifiedSet<K> replace = UnifiedSet.newSet(UnifriedMap.this.size());
            for (var i = 0; i < UnifriedMap.this.t.length; i += 2) {
                var cur = UnifriedMap.this.t[i];
                if (cur == CHAINED_KEY) this.chainedAddToSet((Object[]) UnifriedMap.this.t[i + 1], replace);
                else if (cur != null) replace.add((K) cur);
            }
            return replace;
        }

        private void chainedAddToSet(Object[] chain, UnifiedSet<K> replace) {
            for (var i = 0; i < chain.length; i += 2) {
                var cur = chain[i];
                if (cur == null) return;
                replace.add((K) cur);
            }
        }
    }

    abstract class PositionalIterator<T> implements Iterator<T> {
        int count;
        int position;
        int chainPosition;
        boolean lastReturned;

        @Override
        public boolean hasNext() {
            return this.count < UnifriedMap.this.size();
        }

        @Override
        public void remove() {
            if (!this.lastReturned) throw new IllegalStateException("next() must be called as many times as remove()");
            this.count--;
            UnifriedMap.this.size--;

            if (this.chainPosition != 0) {
                this.removeFromChain();
            } else {
                var pos = this.position - 2;
                var t = UnifriedMap.this.t;
                if (t[pos] == CHAINED_KEY) {
                    this.removeLastFromChain((Object[]) t[pos + 1], pos);
                } else {
                    t[pos] = null;
                    t[pos + 1] = null;
                    this.position = pos;
                    this.lastReturned = false;
                }
            }
        }

        void removeFromChain() {
            var chain = (Object[]) UnifriedMap.this.t[this.position + 1];
            var pos = this.chainPosition - 2;
            var replacePos = this.chainPosition;
            while (replacePos < chain.length - 2 && chain[replacePos + 2] != null) replacePos += 2;
            chain[pos] = chain[replacePos];
            chain[pos + 1] = chain[replacePos + 1];
            chain[replacePos] = null;
            chain[replacePos + 1] = null;
            this.chainPosition = pos;
            this.lastReturned = false;
        }

        void removeLastFromChain(Object[] chain, int tableIndex) {
            var pos = chain.length - 2;
            while (chain[pos] == null) pos -= 2;
            if (pos == 0) {
                UnifriedMap.this.t[tableIndex] = null;
                UnifriedMap.this.t[tableIndex + 1] = null;
            } else {
                chain[pos] = null;
                chain[pos + 1] = null;
            }
            this.lastReturned = false;
        }
    }

    class KeySetIterator extends PositionalIterator<K> {
        K nextFromChain() {
            var chain = (Object[]) UnifriedMap.this.t[this.position + 1];
            var cur = chain[this.chainPosition];
            this.chainPosition += 2;
            if (this.chainPosition >= chain.length
                    || chain[this.chainPosition] == null) {
                this.chainPosition = 0;
                this.position += 2;
            }
            this.lastReturned = true;
            return (K) cur;
        }

        @Override
        public K next() {
            if (!this.hasNext()) throw new NoSuchElementException("next() called, but the iterator is exhausted");
            this.count++;
            var table = UnifriedMap.this.t;
            if (this.chainPosition != 0) return this.nextFromChain();
            while (table[this.position] == null) this.position += 2;
            var cur = table[this.position];
            if (cur == CHAINED_KEY) return this.nextFromChain();
            this.position += 2;
            this.lastReturned = true;
            return (K) cur;
        }
    }

    protected class EntrySet implements Set<Entry<K, V>>, Serializable, BatchIterable<Entry<K, V>> {
//        private static final long serialVersionUID = 1L;
        private transient WeakReference<UnifriedMap<K, V>> holder = new WeakReference<>(UnifriedMap.this);

        @Override
        public boolean add(Entry<K, V> entry) {
            throw new UnsupportedOperationException("Cannot call add() on " + this.getClass().getSimpleName());
        }

        @Override
        public boolean addAll(Collection<? extends Entry<K, V>> collection) {
            throw new UnsupportedOperationException("Cannot call addAll() on " + this.getClass().getSimpleName());
        }

        @Override
        public void clear() {
            UnifriedMap.this.clear();
        }

        boolean containsEntry(Entry<?, ?> entry) {
            return this.getEntry(entry) != null;
        }

        private Entry<K, V> getEntry(Entry<?, ?> entry) {
            var key = (K) entry.getKey();
            var value = (V) entry.getValue();
            var index = UnifriedMap.this.index(key);

            var cur = UnifriedMap.this.t[index];
            var curValue = UnifriedMap.this.t[index + 1];
            if (cur == CHAINED_KEY) return this.chainGetEntry((Object[]) curValue, key, value);
            if (cur == null) return null;
            if (cur.equals(key)) if (value.equals(curValue))
                return ImmutableEntry.of((K) cur, (V) curValue);
            return null;
        }

        private Entry<K, V> chainGetEntry(Object[] chain, K key, V value) {
            for (var i = 0; i < chain.length; i += 2) {
                var cur = chain[i];
                if (cur == null) return null;
                if (cur.equals(key)) {
                    var curValue = chain[i + 1];
                    if (value.equals(curValue))
                        return ImmutableEntry.of((K) cur, (V) curValue);
                }
            }
            return null;
        }

        @Override
        public boolean contains(Object o) {
            return o instanceof Entry e && this.containsEntry(e);
        }

        @Override
        public boolean containsAll(Collection<?> collection) {
            for (var obj : collection) if (!this.contains(obj)) return false;
            return true;
        }

        @Override
        public boolean isEmpty() {
            return UnifriedMap.this.isEmpty();
        }

        @Override
        public Iterator<Entry<K, V>> iterator() {
            return new EntrySetIterator(this.holder);
        }

        @Override
        public boolean remove(Object e) {
            if (!(e instanceof Entry<?, ?> entry)) return false;
            var key = (K) entry.getKey();
            var value = (V) entry.getValue();

            var index = UnifriedMap.this.index(key);

            var cur = UnifriedMap.this.t[index];
            if (cur != null) {
                var val = UnifriedMap.this.t[index + 1];
                if (cur == CHAINED_KEY) return this.removeFromChain((Object[]) val, key, value, index);
                if (cur.equals(key) && value.equals(val)) {
                    UnifriedMap.this.t[index] = null;
                    UnifriedMap.this.t[index + 1] = null;
                    UnifriedMap.this.size--;
                    return true;
                }
            }
            return false;
        }

        private boolean removeFromChain(Object[] chain, K key, V value, int index) {
            for (var i = 0; i < chain.length; i += 2) {
                var k = chain[i];
                if (k == null) return false;
                if (k.equals(key)) {
                    var val = (V) chain[i + 1];
                    if (val.equals(value)) {
                        UnifriedMap.this.overwriteWithLastElementFromChain(chain, index, i);
                        return true;
                    }
                }
            }
            return false;
        }

        @Override
        public boolean removeAll(Collection<?> collection) {
            var changed = false;
            for (var obj : collection) if (this.remove(obj)) changed = true;
            return changed;
        }

        @Override
        public boolean retainAll(Collection<?> collection) {
            var retainedSize = collection.size();
            var retainedCopy = (UnifriedMap<K, V>) UnifriedMap.this.newEmpty(retainedSize);

            for (var obj : collection)
                if (obj instanceof Entry<?, ?> otherEntry) {
                    var thisEntry = this.getEntry(otherEntry);
                    if (thisEntry != null) retainedCopy.put(thisEntry.getKey(), thisEntry.getValue());
                }
            if (retainedCopy.size() < this.size()) {
                UnifriedMap.this.maxSize = retainedCopy.maxSize;
                UnifriedMap.this.size = retainedCopy.size;
                UnifriedMap.this.t = retainedCopy.t;
                return true;
            }
            return false;
        }

        @Override
        public int size() {
            return UnifriedMap.this.size();
        }

        @Override
        public void forEach(Procedure<? super Entry<K, V>> procedure) {
            for (var i = 0; i < UnifriedMap.this.t.length; i += 2) {
                var cur = UnifriedMap.this.t[i];
                if (cur == CHAINED_KEY) this.chainedForEachEntry((Object[]) UnifriedMap.this.t[i + 1], procedure);
                else if (cur != null)
                    procedure.value(ImmutableEntry.of((K) cur, (V) t[i + 1]));
            }
        }

        private void chainedForEachEntry(Object[] chain, Procedure<? super Entry<K, V>> procedure) {
            for (var i = 0; i < chain.length; i += 2) {
                var cur = chain[i];
                if (cur == null) return;
                procedure.value(ImmutableEntry.of((K) cur, (V) chain[i + 1]));
            }
        }

        @Override
        public int getBatchCount(int batchSize) {
            return UnifriedMap.this.getBatchCount(batchSize);
        }

        @Override
        public void batchForEach(Procedure<? super Entry<K, V>> procedure, int sectionIndex, int sectionCount) {
            var map = t;
            var sectionSize = map.length / sectionCount;
            var start = sectionIndex * sectionSize;
            var end = sectionIndex == sectionCount - 1 ? map.length : start + sectionSize;
            if (start % 2 != 0) start++;
            for (var i = start; i < end; i += 2) {
                var cur = map[i];
                if (cur == CHAINED_KEY) this.chainedForEachEntry((Object[]) map[i + 1], procedure);
                else if (cur != null)
                    procedure.value(ImmutableEntry.of((K) cur, (V) map[i + 1]));
            }
        }

        void copyEntries(Object[] result) {
            var table = UnifriedMap.this.t;
            var count = 0;
            for (var i = 0; i < table.length; i += 2) {
                var x = table[i];
                if (x != null) if (x == CHAINED_KEY) {
                    var chain = (Object[]) table[i + 1];
                    for (var j = 0; j < chain.length; j += 2) {
                        var cur = chain[j];
                        if (cur == null) break;
                        result[count++] =
                                new WeakBoundEntry<>((K) cur, (V) chain[j + 1], this.holder);
                    }
                } else
                    result[count++] = new WeakBoundEntry<>((K) x, (V) table[i + 1], this.holder);
            }
        }

        @Override
        public Object[] toArray() {
            var result = new Object[UnifriedMap.this.size()];
            this.copyEntries(result);
            return result;
        }

        @Override
        public <T> T[] toArray(T[] result) {
            var size = UnifriedMap.this.size();
            if (result.length < size) result = (T[]) Array.newInstance(result.getClass().getComponentType(), size);
            this.copyEntries(result);
            if (size < result.length) result[size] = null;
            return result;
        }

        private void readObject(ObjectInputStream in)
                throws IOException, ClassNotFoundException {
            in.defaultReadObject();
            this.holder = new WeakReference<>(UnifriedMap.this);
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof Set<?> other) {
                if (other.size() == this.size()) return this.containsAll(other);
            }
            return false;
        }

        @Override
        public int hashCode() {
            return UnifriedMap.this.hashCode();
        }
    }

    class EntrySetIterator extends PositionalIterator<Entry<K, V>> {
        private final WeakReference<UnifriedMap<K, V>> holder;

        EntrySetIterator(WeakReference<UnifriedMap<K, V>> holder) {
            this.holder = holder;
        }

        Entry<K, V> nextFromChain() {
            var chain = (Object[]) UnifriedMap.this.t[this.position + 1];
            var cur = chain[this.chainPosition];
            var value = chain[this.chainPosition + 1];
            this.chainPosition += 2;
            if (this.chainPosition >= chain.length
                    || chain[this.chainPosition] == null) {
                this.chainPosition = 0;
                this.position += 2;
            }
            this.lastReturned = true;
            return new WeakBoundEntry<>((K) cur, (V) value, this.holder);
        }

        @Override
        public Entry<K, V> next() {
            if (!this.hasNext()) throw new NoSuchElementException("next() called, but the iterator is exhausted");
            this.count++;
            var table = UnifriedMap.this.t;
            if (this.chainPosition != 0) return this.nextFromChain();
            while (table[this.position] == null) this.position += 2;
            var cur = table[this.position];
            var value = table[this.position + 1];
            if (cur == CHAINED_KEY) return this.nextFromChain();
            this.position += 2;
            this.lastReturned = true;
            return new WeakBoundEntry<>((K) cur, (V) value, this.holder);
        }
    }

    protected class ValuesCollection extends ValuesCollectionCommon<V>
            implements Serializable, BatchIterable<V> {
        private static final long serialVersionUID = 1L;

        @Override
        public void clear() {
            UnifriedMap.this.clear();
        }

        @Override
        public boolean contains(Object o) {
            return UnifriedMap.this.containsValue(o);
        }

        @Override
        public boolean containsAll(Collection<?> collection) {
            // todo: this is N^2. if c is large, we should copy the values to a set.
            return Iterate.allSatisfy(collection, Predicates.in(this));
        }

        @Override
        public boolean isEmpty() {
            return UnifriedMap.this.isEmpty();
        }

        @Override
        public Iterator<V> iterator() {
            return new ValuesIterator();
        }

        @Override
        public boolean remove(Object o) {
            // this is so slow that the extra overhead of the iterator won't be noticeable
            if (o == null) {
                for (var it = this.iterator(); it.hasNext(); )
                    if (it.next() == null) {
                        it.remove();
                        return true;
                    }
            } else {
                for (var it = this.iterator(); it.hasNext(); ) {
                    var o2 = it.next();
                    if (o == o2 || o2.equals(o)) {
                        it.remove();
                        return true;
                    }
                }
            }
            return false;
        }

        @Override
        public boolean removeAll(Collection<?> collection) {
            // todo: this is N^2. if c is large, we should copy the values to a set.
            var changed = false;

            for (var obj : collection) if (this.remove(obj)) changed = true;
            return changed;
        }

        @Override
        public boolean retainAll(Collection<?> collection) {
            var modified = false;
            var e = this.iterator();
            while (e.hasNext()) if (!collection.contains(e.next())) {
                e.remove();
                modified = true;
            }
            return modified;
        }

        @Override
        public int size() {
            return UnifriedMap.this.size();
        }

        @Override
        public void forEach(Procedure<? super V> procedure) {
            UnifriedMap.this.forEachValue(procedure);
        }

        @Override
        public int getBatchCount(int batchSize) {
            return UnifriedMap.this.getBatchCount(batchSize);
        }

        @Override
        public void batchForEach(Procedure<? super V> procedure, int sectionIndex, int sectionCount) {
            UnifriedMap.this.batchForEach(procedure, sectionIndex, sectionCount);
        }

        void copyValues(Object[] result) {
            var count = 0;
            for (var i = 0; i < UnifriedMap.this.t.length; i += 2) {
                var x = UnifriedMap.this.t[i];
                if (x != null) if (x == CHAINED_KEY) {
                    var chain = (Object[]) UnifriedMap.this.t[i + 1];
                    for (var j = 0; j < chain.length; j += 2) {
                        var cur = chain[j];
                        if (cur == null) break;
                        result[count++] = chain[j + 1];
                    }
                } else result[count++] = UnifriedMap.this.t[i + 1];
            }
        }

        @Override
        public Object[] toArray() {
            var size = UnifriedMap.this.size();
            var result = new Object[size];
            this.copyValues(result);
            return result;
        }

        @Override
        public <T> T[] toArray(T[] result) {
            var size = UnifriedMap.this.size();
            if (result.length < size) result = (T[]) Array.newInstance(result.getClass().getComponentType(), size);
            this.copyValues(result);
            if (size < result.length) result[size] = null;
            return result;
        }

        protected Object writeReplace() {
            FastList<V> replace = FastList.newList(UnifriedMap.this.size());
            for (var i = 0; i < UnifriedMap.this.t.length; i += 2) {
                var cur = UnifriedMap.this.t[i];
                if (cur == CHAINED_KEY) this.chainedAddToList((Object[]) UnifriedMap.this.t[i + 1], replace);
                else if (cur != null) replace.add((V) UnifriedMap.this.t[i + 1]);
            }
            return replace;
        }

        private void chainedAddToList(Object[] chain, FastList<V> replace) {
            for (var i = 0; i < chain.length; i += 2) {
                var cur = chain[i];
                if (cur == null) return;
                replace.add((V) chain[i + 1]);
            }
        }

        @Override
        public String toString() {
            return Iterate.makeString(this, "[", ", ", "]");
        }
    }

    class ValuesIterator extends PositionalIterator<V> {
        V nextFromChain() {
            var chain = (Object[]) UnifriedMap.this.t[this.position + 1];
            var val = chain[this.chainPosition + 1];
            this.chainPosition += 2;
            if (this.chainPosition >= chain.length
                    || chain[this.chainPosition] == null) {
                this.chainPosition = 0;
                this.position += 2;
            }
            this.lastReturned = true;
            return (V)val;
        }

        @Override
        public V next() {
            if (!this.hasNext()) throw new NoSuchElementException("next() called, but the iterator is exhausted");
            this.count++;
            if (this.chainPosition != 0) return this.nextFromChain();

            var p = this.position;
            var table = UnifriedMap.this.t;
            while (table[p] == null) p += 2;
            var cur = table[p];
            this.position = p;
            if (cur == CHAINED_KEY)
                return this.nextFromChain();
            else {
                var val = table[p + 1];
                this.position += 2;
                this.lastReturned = true;
                return (V) val;
            }
        }
    }
}