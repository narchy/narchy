/*
 * OPTIMIZED VERSION
 *
 * Copyright (c) 2022 Goldman Sachs and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v. 1.0 which accompany this distribution.
 * The Eclipse Public License is available at http://www.eclipse.org/legal/epl-v10.html
 * and the Eclipse Distribution License is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 */

package jcog.data.map;

import org.eclipse.collections.api.IntIterable;
import org.eclipse.collections.api.LazyIntIterable;
import org.eclipse.collections.api.LazyIterable;
import org.eclipse.collections.api.RichIterable;
import org.eclipse.collections.api.bag.primitive.MutableIntBag;
import org.eclipse.collections.api.block.function.primitive.*;
import org.eclipse.collections.api.block.predicate.primitive.IntPredicate;
import org.eclipse.collections.api.block.predicate.primitive.ObjectIntPredicate;
import org.eclipse.collections.api.block.procedure.Procedure;
import org.eclipse.collections.api.block.procedure.primitive.IntProcedure;
import org.eclipse.collections.api.block.procedure.primitive.ObjectIntProcedure;
import org.eclipse.collections.api.collection.MutableCollection;
import org.eclipse.collections.api.collection.primitive.ImmutableIntCollection;
import org.eclipse.collections.api.collection.primitive.MutableIntCollection;
import org.eclipse.collections.api.factory.Lists;
import org.eclipse.collections.api.factory.primitive.IntBags;
import org.eclipse.collections.api.factory.primitive.IntLists;
import org.eclipse.collections.api.factory.primitive.IntObjectMaps;
import org.eclipse.collections.api.factory.primitive.IntSets;
import org.eclipse.collections.api.iterator.IntIterator;
import org.eclipse.collections.api.iterator.MutableIntIterator;
import org.eclipse.collections.api.list.MutableList;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.api.map.primitive.MutableIntObjectMap;
import org.eclipse.collections.api.map.primitive.ObjectIntMap;
import org.eclipse.collections.api.set.primitive.IntSet;
import org.eclipse.collections.api.set.primitive.MutableIntSet;
import org.eclipse.collections.api.tuple.primitive.ObjectIntPair;
import org.eclipse.collections.impl.collection.mutable.primitive.SynchronizedIntCollection;
import org.eclipse.collections.impl.collection.mutable.primitive.UnmodifiableIntCollection;
import org.eclipse.collections.impl.lazy.AbstractLazyIterable;
import org.eclipse.collections.impl.lazy.primitive.LazyIntIterableAdapter;
import org.eclipse.collections.impl.tuple.primitive.PrimitiveTuples;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.reflect.Array;
import java.util.*;

/**
 * This file was automatically generated from template file objectPrimitiveHashMap.stg.
 *
 * @since 3.0.
 */
public final class ObjIntHashMap<K> {

    private static final int DEFAULT_INITIAL_CAPACITY = 8;

    private static final Object REMOVED_KEY = new Object() {

        public boolean equals(Object obj) {
            throw new RuntimeException("Possible corruption through unsynchronized concurrent modification.");
        }


        public int hashCode() {
            throw new RuntimeException("Possible corruption through unsynchronized concurrent modification.");
        }


        public String toString() {
            return "ObjectIntHashMap.REMOVED_KEY";
        }
    };

    private Object[] keys;
    private int[] values;

    private int occupiedWithData;
    private int occupiedWithSentinels;

    public ObjIntHashMap() {
        this.allocateTable(DEFAULT_INITIAL_CAPACITY << 1);
    }

    public ObjIntHashMap(int initialCapacity) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("initial capacity cannot be less than 0");
        }
        this.allocateTable(smallestPowerOfTwoGreaterThan(fastCeil(initialCapacity << 1)));
    }

    public ObjIntHashMap(ObjectIntMap<K> map) {
        this(Math.max(map.size(), DEFAULT_INITIAL_CAPACITY));
        this.putAll(map);
    }

    public static <K> ObjIntHashMap<K> newMap() {
        return new ObjIntHashMap<>();
    }

    private static int smallestPowerOfTwoGreaterThan(int n) {
        return n > 1 ? Integer.highestOneBit(n - 1) << 1 : 1;
    }

    private static int fastCeil(float v) {
        var possibleResult = (int) v;
        if (v - possibleResult > 0)
            possibleResult++;
        return possibleResult;
    }

    private static boolean isNonSentinel(Object key) {
        return key != null && key != REMOVED_KEY;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (!(obj instanceof ObjectIntMap)) {
            return false;
        }

        var other = (ObjectIntMap<K>) obj;

        if (this.size() != other.size()) {
            return false;
        }

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && (!other.containsKey(this.keys[i]) || this.values[i] != other.getOrThrow(this.keys[i]))) {
                return false;
            }
        }
        return true;
    }

    public int hashCode() {
        var result = 0;

        var keys = this.keys;
        for (var i = 0; i < keys.length; i++) {
            var ki = keys[i];
            if (isNonSentinel(ki))
                result += ki.hashCode() ^ this.values[i];
        }
        return result;
    }

    public String toString() {
        var a = new StringBuilder();

        a.append("{");

        var first = true;

        for (var i = 0; i < this.keys.length; i++) {
            var key = this.keys[i];
            if (isNonSentinel(key)) {
                if (!first)
                    a.append(", ");
                a.append(key).append("=").append(this.values[i]);
                first = false;
            }
        }
        a.append("}");

        return a.toString();
    }

    public int size() {
        return this.occupiedWithData;
    }

    public boolean isEmpty() {
        return this.size() == 0;
    }

    public boolean notEmpty() {
        return this.size() != 0;
    }

    public String makeString() {
        return this.makeString(", ");
    }

    public String makeString(String separator) {
        return this.makeString("", separator, "");
    }

    public String makeString(String start, String separator, String end) {
        Appendable stringBuilder = new StringBuilder();
        this.appendString(stringBuilder, start, separator, end);
        return stringBuilder.toString();
    }

    public void appendString(Appendable appendable) {
        this.appendString(appendable, ", ");
    }

    public void appendString(Appendable appendable, String separator) {
        this.appendString(appendable, "", separator, "");
    }

    public void appendString(Appendable appendable, String start, String separator, String end) {
        try {
            appendable.append(start);

            var first = true;

            for (var i = 0; i < this.keys.length; i++) {
                var key = this.keys[i];
                if (isNonSentinel(key)) {
                    if (!first) {
                        appendable.append(separator);
                    }
                    appendable.append(String.valueOf(this.values[i]));
                    first = false;
                }
            }
            appendable.append(end);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public MutableIntIterator intIterator() {
        return new InternalIntIterator();
    }

    public int[] toArray() {
        var result = new int[this.size()];
        var index = 0;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                result[index] = this.values[i];
                index++;
            }
        }
        return result;
    }

    public int[] toArray(int[] target) {
        if (target.length < this.size()) {
            target = new int[this.size()];
        }
        var index = 0;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                target[index] = this.values[i];
                index++;
            }
        }
        return target;
    }

    public boolean contains(int value) {
        return this.containsValue(value);
    }

    public boolean containsAll(int... source) {
        for (var item : source) {
            if (!this.containsValue(item)) {
                return false;
            }
        }
        return true;
    }

    public boolean containsAll(IntIterable source) {
        return this.containsAll(source.toArray());
    }

    public void clear() {
        this.occupiedWithData = 0;
        this.occupiedWithSentinels = 0;
        Arrays.fill(this.keys, null);
        //Arrays.fill(this.values, EMPTY_VALUE);
    }

    public void put(K key, int value) {
        var index = this.probe(key);
        var ki = this.keys[index];
        if (isNonSentinel(ki) && ki.equals(key))
            this.values[index] = value; // key already present in map
        else
            this.set(key, value, index);
    }

    private boolean isNonSentinel(int index) {
        return isNonSentinel(keys[index]);
    }

    public void putAll(ObjectIntMap<K> map) {
        map.forEachKeyValue(this::put);
    }

    public void updateValues(ObjectIntToIntFunction<K> function) {
        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                this.values[i] = function.valueOf((K) this.keys[i], this.values[i]);
            }
        }
    }

    public void remove(K key) {
        this.removeKeyAtIndex(key, this.probe(key));
    }

    private void removeKeyAtIndex(K key, int index) {
        //var ki = keys[index]; if (isNonSentinel(ki) && eqNonSentinel(key, ki)) {
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            this.keys[index] = REMOVED_KEY;
            //this.values[index] = EMPTY_VALUE;
            this.occupiedWithData--;
            this.occupiedWithSentinels++;
        }
    }


    public int removeKeyIfAbsent(K key, int value) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            this.keys[index] = REMOVED_KEY;
            var oldValue = this.values[index];
            //this.values[index] = EMPTY_VALUE;
            this.occupiedWithData--;
            this.occupiedWithSentinels++;

            return oldValue;
        }
        return value;
    }


    public int getIfAbsentPut(K key, int value) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            return this.values[index];
        } else {
            this.set(key, value, index);
            return value;
        }
    }


    public int getAndPut(K key, int putValue, int defaultValue) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            var v = this.values;
            var existingValue = v[index];
            v[index] = putValue;
            return existingValue;
        } else {
            this.set(key, putValue, index);
            return defaultValue;
        }
    }


    public int getIfAbsentPut(K key, IntFunction0 function) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            return this.values[index];
        } else {
            var value = function.value();
            this.set(key, value, index);
            return value;
        }
    }


    public <P> int getIfAbsentPutWith(K key, IntFunction<P> function, P parameter) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            return this.values[index];
        } else {
            var value = function.intValueOf(parameter);
            this.set(key, value, index);
            return value;
        }
    }


    public int getIfAbsentPutWithKey(K key, IntFunction<K> function) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            return this.values[index];
        } else {
            var value = function.intValueOf(key);
            this.set(key, value, index);
            return value;
        }
    }


    public int updateValue(K key, int initialValueIfAbsent, IntToIntFunction function) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key)) {
            return this.values[index] = function.valueOf(this.values[index]);
        } else {
            var value = function.valueOf(initialValueIfAbsent);
            this.set(key, value, index);
            return value;
        }
    }

    private void set(K key, int value, int index) {
        var k = this.keys;
        if (k[index] == REMOVED_KEY)
            --this.occupiedWithSentinels;

        k[index] = key;
        this.values[index] = value;
        if ((++this.occupiedWithData) + this.occupiedWithSentinels > this.maxOccupiedWithData())
            this.rehashAndGrow();
    }

    public final int increment(K key) {
        return addToValue(key, 1);
    }

    public int addToValue(K key, int inc) {
        var index = this.probe(key);
        var ki = keys[index];
        if (isNonSentinel(ki) && ki.equals(key))
            return (this.values[index] += inc);
        else {
            this.set(key, inc, index);
            return inc;
        }
    }

    public ObjIntHashMap<K> withoutKey(K key) {
        this.remove(key);
        return this;
    }

    public ObjIntHashMap<K> withoutAllKeys(Iterable<K> keys) {
        for (var key : keys)
            this.remove(key);
        return this;
    }

    public int getOrThrow(K key) {
        var index = this.probe(key);
        if (isNonSentinel(index))
            return this.values[index];

        throw new IllegalStateException("Key " + key + " not present.");
    }

    public int getIfAbsent(Object key, int ifAbsent) {
        var index = probe(key);
        var ki = keys[index];
        return isNonSentinel(ki) && ki.equals(key) ? this.values[index] : ifAbsent;
    }

    public boolean containsKey(K key) {
        var index = this.probe(key);
        var ki = this.keys[index];
        return isNonSentinel(ki) && ki.equals(key);
    }

    public boolean containsValue(int value) {
        var v = this.values;
        var n = v.length;
        for (var i = 0; i < n; i++) {
            if (isNonSentinel(i) && v[i] == value)
                return true;
        }
        return false;
    }

    public void each(IntProcedure procedure) {
        this.forEachValue(procedure);
    }

    public void forEachValue(IntProcedure procedure) {
        var v = this.values;
        var k = this.keys;
        var n = k.length;
        for (var i = 0; i < n; i++) {
            if (isNonSentinel(i))
                procedure.value(v[i]);
        }
    }

    public void forEachKey(Procedure<K> procedure) {
        for (var key : this.keys) {
            if (isNonSentinel(key))
                procedure.value((K) key);
        }
    }


    public void forEachKeyValue(ObjectIntProcedure<K> procedure) {
        var k = this.keys;
        var n = k.length;
        var v = this.values;
        for (var i = 0; i < n; i++) {
            var ki = k[i];
            if (isNonSentinel(ki))
                procedure.value((K) ki, v[i]);
        }
    }


    public ObjIntHashMap<K> select(ObjectIntPredicate<K> predicate) {
        ObjIntHashMap<K> result = newMap();

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && predicate.accept((K) this.keys[i], this.values[i])) {
                result.put((K) this.keys[i], this.values[i]);
            }
        }
        return result;
    }


    public ObjIntHashMap<K> reject(ObjectIntPredicate<K> predicate) {
        ObjIntHashMap<K> result = newMap();

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && !predicate.accept((K) this.keys[i], this.values[i])) {
                result.put((K) this.keys[i], this.values[i]);
            }
        }
        return result;
    }

    public MutableIntCollection select(IntPredicate predicate) {
        var result = IntLists.mutable.empty();

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && predicate.accept(this.values[i])) {
                result.add(this.values[i]);
            }
        }
        return result;
    }


    public MutableIntCollection reject(IntPredicate predicate) {
        var result = IntLists.mutable.empty();

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && !predicate.accept(this.values[i])) {
                result.add(this.values[i]);
            }
        }
        return result;
    }

    public int detectIfNone(IntPredicate predicate, int ifNone) {
        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && predicate.accept(this.values[i])) {
                return this.values[i];
            }
        }
        return ifNone;
    }

    public <V> MutableCollection<V> collect(IntToObjectFunction<V> function) {
        MutableList<V> result = Lists.mutable.withInitialCapacity(this.size());
        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                result.add(function.valueOf(this.values[i]));
            }
        }
        return result;
    }

    public int count(IntPredicate predicate) {
        var count = 0;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && predicate.accept(this.values[i])) {
                count++;
            }
        }
        return count;
    }

    public boolean anySatisfy(IntPredicate predicate) {
        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && predicate.accept(this.values[i])) {
                return true;
            }
        }
        return false;
    }

    public boolean allSatisfy(IntPredicate predicate) {
        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && !predicate.accept(this.values[i])) {
                return false;
            }
        }
        return true;
    }

    public boolean noneSatisfy(IntPredicate predicate) {
        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && predicate.accept(this.values[i])) {
                return false;
            }
        }
        return true;
    }

    public <V> V injectInto(V injectedValue, ObjectIntToObjectFunction<V, V> function) {
        var result = injectedValue;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                result = function.valueOf(result, this.values[i]);
            }
        }

        return result;
    }


    public RichIterable<IntIterable> chunk(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Size for groups must be positive but was: " + size);
        }
        MutableList<IntIterable> result = Lists.mutable.empty();
        if (this.notEmpty()) {
            IntIterator iterator = this.intIterator();
            while (iterator.hasNext()) {
                var batch = IntBags.mutable.empty();
                for (var i = 0; i < size && iterator.hasNext(); i++) {
                    batch.add(iterator.next());
                }
                result.add(batch);
            }
        }
        return result;
    }


    public long sum() {
        var result = 0L;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                result += this.values[i];
            }
        }
        return result;
    }


    public int max() {
        if (this.isEmpty()) {
            throw new NoSuchElementException();
        }
        var max = 0;
        var isMaxSet = false;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && (!isMaxSet || max < this.values[i])) {
                max = this.values[i];
                isMaxSet = true;
            }
        }
        return max;
    }


    public int min() {
        if (this.isEmpty()) {
            throw new NoSuchElementException();
        }
        var min = 0;
        var isMinSet = false;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && (!isMinSet || this.values[i] < min)) {
                min = this.values[i];
                isMinSet = true;
            }
        }
        return min;
    }


    public int maxIfEmpty(int defaultValue) {
        if (this.isEmpty()) {
            return defaultValue;
        }
        var max = 0;
        var isMaxSet = false;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && (!isMaxSet || max < this.values[i])) {
                max = this.values[i];
                isMaxSet = true;
            }
        }
        return max;
    }


    public int minIfEmpty(int defaultValue) {
        if (this.isEmpty()) {
            return defaultValue;
        }
        var min = 0;
        var isMinSet = false;

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i) && (!isMinSet || this.values[i] < min)) {
                min = this.values[i];
                isMinSet = true;
            }
        }
        return min;
    }


    public double average() {
        if (this.isEmpty()) {
            throw new ArithmeticException();
        }
        return (double) this.sum() / this.size();
    }


    public double median() {
        if (this.isEmpty()) {
            throw new ArithmeticException();
        }
        var sortedArray = this.toSortedArray();
        var middleIndex = sortedArray.length >> 1;
        if (sortedArray.length > 1 && (sortedArray.length & 1) == 0) {
            var first = sortedArray[middleIndex];
            var second = sortedArray[middleIndex - 1];
            return ((double) first + second) / 2.0;
        }
        return sortedArray[middleIndex];
    }


    public MutableIntList toList() {
        var result = IntLists.mutable.withInitialCapacity(this.size());

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                result.add(this.values[i]);
            }
        }
        return result;
    }


    public MutableIntSet toSet() {
        var result = IntSets.mutable.empty();

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                result.add(this.values[i]);
            }
        }
        return result;
    }


    public MutableIntBag toBag() {
        var result = IntBags.mutable.empty();

        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                result.add(this.values[i]);
            }
        }
        return result;
    }

    public int[] toSortedArray() {
        var array = this.toArray();
        Arrays.sort(array);
        return array;
    }

    public MutableIntList toSortedList() {
        return this.toList().sortThis();
    }


    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(this.size());
        for (var i = 0; i < this.keys.length; i++) {
            if (isNonSentinel(i)) {
                out.writeObject(this.keys[i]);
                out.writeInt(this.values[i]);
            }
        }
    }


    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        var size = in.readInt();
        this.allocateTable(smallestPowerOfTwoGreaterThan(fastCeil(size << 1)));

        for (var i = 0; i < size; i++) {
            this.put((K) in.readObject(), in.readInt());
        }
    }


    public LazyIterable<K> keysView() {
        return new KeysView();
    }


    public RichIterable<ObjectIntPair<K>> keyValuesView() {
        return new KeyValuesView();
    }


    public MutableIntObjectMap<K> flipUniqueValues() {
        MutableIntObjectMap<K> result = IntObjectMaps.mutable.empty();
        this.forEachKeyValue((key, value) ->
        {
            var oldKey = result.put(value, key);
            if (oldKey != null) {
                throw new IllegalStateException("Duplicate value: " + value + " found at key: " + oldKey + " and key: " + key);
            }
        });
        return result;
    }

    /**
     * @since 12.0
     */
    public boolean trimToSize() {
        var newCapacity = smallestPowerOfTwoGreaterThan(this.size());
        if (this.keys.length > newCapacity) {
            this.rehash(newCapacity);
            return true;
        }
        return false;
    }

    /**
     * Rehashes every element in the set into a new backing table of the smallest possible size and eliminating removed sentinels.
     *
     * @deprecated since 12.0 - Use {@link #trimToSize()} instead
     */
    @Deprecated
    public void compact() {
        this.rehash(smallestPowerOfTwoGreaterThan(this.size()));
    }

    private void rehashAndGrow() {
        var max = this.maxOccupiedWithData();
        var newCapacity = Math.max(max, smallestPowerOfTwoGreaterThan((this.occupiedWithData + 1) << 1));
        if (this.occupiedWithSentinels > 0 && (max >> 1) + (max >> 2) < this.occupiedWithData) {
            newCapacity <<= 1;
        }
        this.rehash(newCapacity);
    }

    private void rehash(int newCapacity) {
        var oldLength = this.keys.length;
        var old = this.keys;
        var oldValues = this.values;
        this.allocateTable(newCapacity);
        this.occupiedWithData = 0;
        this.occupiedWithSentinels = 0;

        for (var i = 0; i < oldLength; i++)
            if (isNonSentinel(old[i]))
                this.put((K) old[i], oldValues[i]);
    }

    private int probe(Object element) {
        var index = this.spread(element);
        var removedIndex = -1;
        var probe = 17;
        var keys = this.keys;
        var mask = keys.length - 1;

        while (true) {
            var ki = keys[index];

            if (ki == null)
                return removedIndex == -1 ? index : removedIndex;
            else if (ki == REMOVED_KEY) {
                if (removedIndex == -1)
                    removedIndex = index;
            } else if (ki.equals(element))
                return index;

            // Probe algorithm: 17*n*(n+1)/2 where n = no. of collisions
            index = (index + probe) & mask;
            probe += 17;
        }
    }

    private int spread(Object x) {
        var h = x.hashCode();
        // This function ensures that hashCodes that differ only by
        // constant multiples at each bit position have a bounded
        // number of collisions (approximately 8 at default load factor).
        h ^= h >>> 20 ^ h >>> 12;
        h ^= h >>> 7 ^ h >>> 4;

        //XOR mixing
//        h ^= (h >>> 16);
//        h *= 0x3243f6a9; // Multiplication by prime number for better distribution
//        h ^= (h >>> 16);

        return h & (keys.length - 1);
    }

    private void allocateTable(int sizeToAllocate) {
        this.keys = new Object[sizeToAllocate];
        this.values = new int[sizeToAllocate];
    }

    private int maxOccupiedWithData() {
        var capacity = this.keys.length;
        // need at least one free slot for open addressing
        return Math.min(capacity - 1, capacity >> 1);
    }


    public Set<K> keySet() {
        return new KeySet();
    }


    public MutableIntCollection values() {
        return new ValuesCollection();
    }

    private class InternalIntIterator implements MutableIntIterator {
        private int count;
        private int position;


        public boolean hasNext() {
            return this.count != ObjIntHashMap.this.size();
        }


        public int next() {
            if (!this.hasNext()) {
                throw new NoSuchElementException();
            }

            var keys = ObjIntHashMap.this.keys;
            while (!isNonSentinel(keys[this.position])) {
                this.position++;
            }
            var result = ObjIntHashMap.this.values[this.position];
            this.count++;
            this.position++;
            return result;
        }


        public void remove() {
            if (this.position == 0 || !isNonSentinel(ObjIntHashMap.this.keys[this.position - 1])) {
                throw new IllegalStateException();
            }
            ObjIntHashMap.this.remove((K) ObjIntHashMap.this.keys[this.position - 1]);
            this.count--;
        }
    }

    private class KeySet implements Set<K> {

        public boolean equals(Object obj) {
            if (obj instanceof Set<?> other) {
                if (other.size() == this.size()) {
                    return this.containsAll(other);
                }
            }
            return false;
        }


        public int hashCode() {
            var hashCode = 0;
            var table = ObjIntHashMap.this.keys;
            for (var key : table) {
                if (isNonSentinel(key)) {
                    var nonSentinelKey = (K) key;
                    hashCode += nonSentinelKey == null ? 0 : nonSentinelKey.hashCode();
                }
            }
            return hashCode;
        }


        public int size() {
            return ObjIntHashMap.this.size();
        }


        public boolean isEmpty() {
            return ObjIntHashMap.this.isEmpty();
        }


        public boolean contains(Object o) {
            return ObjIntHashMap.this.containsKey((K) o);
        }


        public Object[] toArray() {
            var size = ObjIntHashMap.this.size();
            var result = new Object[size];
            this.copyKeys(result);
            return result;
        }


        public <T> T[] toArray(T[] result) {
            var size = ObjIntHashMap.this.size();
            if (result.length < size) {
                result = (T[]) Array.newInstance(result.getClass().getComponentType(), size);
            }
            this.copyKeys(result);
            if (size < result.length) {
                result[size] = null;
            }
            return result;
        }


        public boolean add(K key) {
            throw new UnsupportedOperationException("Cannot call add() on " + this.getClass().getSimpleName());
        }


        public boolean remove(Object key) {
            var oldSize = ObjIntHashMap.this.size();
            ObjIntHashMap.this.remove((K) key);
            return oldSize != ObjIntHashMap.this.size();
        }


        public boolean containsAll(Collection collection) {
            for (var aCollection : collection) {
                if (!ObjIntHashMap.this.containsKey((K) aCollection)) {
                    return false;
                }
            }
            return true;
        }


        public boolean addAll(Collection<? extends K> collection) {
            throw new UnsupportedOperationException("Cannot call addAll() on " + this.getClass().getSimpleName());
        }


        public boolean retainAll(Collection<?> collection) {
            var oldSize = ObjIntHashMap.this.size();
            this.removeIf(next -> !collection.contains(next));
            return oldSize != ObjIntHashMap.this.size();
        }


        public boolean removeAll(Collection<?> collection) {
            var oldSize = ObjIntHashMap.this.size();
            for (var object : collection) {
                ObjIntHashMap.this.remove((K) object);
            }
            return oldSize != ObjIntHashMap.this.size();
        }


        public void clear() {
            ObjIntHashMap.this.clear();
        }


        public Iterator<K> iterator() {
            return new KeySetIterator();
        }

        private void copyKeys(Object[] result) {
            var count = 0;
            for (var key : ObjIntHashMap.this.keys) {
                if (isNonSentinel(key)) {
                    result[count++] = key;
                }
            }
        }
    }

    private class KeySetIterator implements Iterator<K> {
        private int count;
        private int position;
        private K currentKey;
        private boolean isCurrentKeySet;


        public boolean hasNext() {
            return this.count < ObjIntHashMap.this.size();
        }


        public K next() {
            if (!this.hasNext()) {
                throw new NoSuchElementException();
            }
            this.count++;
            var keys = ObjIntHashMap.this.keys;
            while (!isNonSentinel(keys[this.position])) {
                this.position++;
            }
            this.currentKey = (K) ObjIntHashMap.this.keys[this.position];
            this.isCurrentKeySet = true;
            this.position++;
            return this.currentKey;
        }


        public void remove() {
            if (!this.isCurrentKeySet) {
                throw new IllegalStateException();
            }

            this.isCurrentKeySet = false;
            this.count--;

            if (isNonSentinel(this.currentKey)) {
                var index = this.position - 1;
                ObjIntHashMap.this.removeKeyAtIndex(this.currentKey, index);
            } else {
                ObjIntHashMap.this.remove(this.currentKey);
            }
        }
    }

    private class ValuesCollection implements MutableIntCollection {
        @Override
        public void each(IntProcedure procedure) {
            throw new UnsupportedOperationException();
        }

        public int size() {
            return ObjIntHashMap.this.size();
        }


        public boolean isEmpty() {
            return ObjIntHashMap.this.isEmpty();
        }


        public boolean notEmpty() {
            return ObjIntHashMap.this.notEmpty();
        }


        public String makeString() {
            return this.makeString(", ");
        }


        public String makeString(String separator) {
            return this.makeString("", separator, "");
        }


        public String makeString(String start, String separator, String end) {
            Appendable stringBuilder = new StringBuilder();
            this.appendString(stringBuilder, start, separator, end);
            return stringBuilder.toString();
        }


        public void appendString(Appendable appendable) {
            this.appendString(appendable, ", ");
        }


        public void appendString(Appendable appendable, String separator) {
            this.appendString(appendable, "", separator, "");
        }


        public void appendString(Appendable appendable, String start, String separator, String end) {
            try {
                appendable.append(start);

                var first = true;

                for (var i = 0; i < ObjIntHashMap.this.keys.length; i++) {
                    var key = ObjIntHashMap.this.keys[i];
                    if (isNonSentinel(key)) {
                        if (!first) {
                            appendable.append(separator);
                        }
                        appendable.append(String.valueOf(ObjIntHashMap.this.values[i]));
                        first = false;
                    }
                }
                appendable.append(end);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }


        public boolean add(int element) {
            throw new UnsupportedOperationException("Cannot call add() on " + this.getClass().getSimpleName());
        }


        public boolean addAll(int... source) {
            throw new UnsupportedOperationException("Cannot call addAll() on " + this.getClass().getSimpleName());
        }


        public boolean addAll(IntIterable source) {
            throw new UnsupportedOperationException("Cannot call addAll() on " + this.getClass().getSimpleName());
        }


        public boolean remove(int item) {
            var oldSize = ObjIntHashMap.this.size();

            for (var i = 0; i < ObjIntHashMap.this.keys.length; i++) {
                if (isNonSentinel(ObjIntHashMap.this.keys[i]) && item == ObjIntHashMap.this.values[i]) {
                    ObjIntHashMap.this.remove((K) ObjIntHashMap.this.keys[i]);
                }
            }
            return oldSize != ObjIntHashMap.this.size();
        }


        public boolean removeAll(IntIterable source) {
            var oldSize = ObjIntHashMap.this.size();

            var iterator = source.intIterator();
            while (iterator.hasNext()) {
                this.remove(iterator.next());
            }
            return oldSize != ObjIntHashMap.this.size();
        }


        public boolean removeAll(int... source) {
            var oldSize = ObjIntHashMap.this.size();

            for (var item : source) {
                this.remove(item);
            }
            return oldSize != ObjIntHashMap.this.size();
        }


        public boolean retainAll(IntIterable source) {
            var oldSize = ObjIntHashMap.this.size();
            final var sourceSet = source instanceof IntSet ? (IntSet) source : source.toSet();
            var retained = ObjIntHashMap.this.select((K object, int value) -> sourceSet.contains(value));
            if (retained.size() != oldSize) {
                ObjIntHashMap.this.keys = retained.keys;
                ObjIntHashMap.this.values = retained.values;
                ObjIntHashMap.this.occupiedWithData = retained.occupiedWithData;
                ObjIntHashMap.this.occupiedWithSentinels = retained.occupiedWithSentinels;
                return true;
            }
            return false;
        }


        public boolean retainAll(int... source) {
            return this.retainAll(IntSets.mutable.with(source));
        }


        public void clear() {
            ObjIntHashMap.this.clear();
        }


        public MutableIntCollection with(int element) {
            throw new UnsupportedOperationException("Cannot call with() on " + this.getClass().getSimpleName());
        }


        public MutableIntCollection without(int element) {
            throw new UnsupportedOperationException("Cannot call without() on " + this.getClass().getSimpleName());
        }


        public MutableIntCollection withAll(IntIterable elements) {
            throw new UnsupportedOperationException("Cannot call withAll() on " + this.getClass().getSimpleName());
        }


        public MutableIntCollection withoutAll(IntIterable elements) {
            throw new UnsupportedOperationException("Cannot call withoutAll() on " + this.getClass().getSimpleName());
        }


        public MutableIntCollection asUnmodifiable() {
            return UnmodifiableIntCollection.of(this);
        }


        public MutableIntCollection asSynchronized() {
            return SynchronizedIntCollection.of(this);
        }


        public ImmutableIntCollection toImmutable() {
            return IntLists.immutable.withAll(this);
        }


        public MutableIntIterator intIterator() {
            return ObjIntHashMap.this.intIterator();
        }


        public int[] toArray() {
            return ObjIntHashMap.this.toArray();
        }


        public int[] toArray(int[] target) {
            return ObjIntHashMap.this.toArray(target);
        }


        public boolean contains(int value) {
            return ObjIntHashMap.this.containsValue(value);
        }


        public boolean containsAll(int... source) {
            return ObjIntHashMap.this.containsAll(source);
        }


        public boolean containsAll(IntIterable source) {
            return ObjIntHashMap.this.containsAll(source);
        }


        public MutableIntCollection select(IntPredicate predicate) {
            return ObjIntHashMap.this.select(predicate);
        }


        public MutableIntCollection reject(IntPredicate predicate) {
            return ObjIntHashMap.this.reject(predicate);
        }


        public RichIterable<IntIterable> chunk(int size) {
            return ObjIntHashMap.this.chunk(size);
        }


        public int detectIfNone(IntPredicate predicate, int ifNone) {
            return ObjIntHashMap.this.detectIfNone(predicate, ifNone);
        }


        public int count(IntPredicate predicate) {
            return ObjIntHashMap.this.count(predicate);
        }


        public boolean anySatisfy(IntPredicate predicate) {
            return ObjIntHashMap.this.anySatisfy(predicate);
        }


        public boolean allSatisfy(IntPredicate predicate) {
            return ObjIntHashMap.this.allSatisfy(predicate);
        }


        public boolean noneSatisfy(IntPredicate predicate) {
            return ObjIntHashMap.this.noneSatisfy(predicate);
        }


        public MutableIntList toList() {
            return ObjIntHashMap.this.toList();
        }


        public MutableIntSet toSet() {
            return ObjIntHashMap.this.toSet();
        }


        public MutableIntBag toBag() {
            return ObjIntHashMap.this.toBag();
        }


        public LazyIntIterable asLazy() {
            return new LazyIntIterableAdapter(this);
        }

        @Override
        public <T> T injectInto(T injectedValue, ObjectIntToObjectFunction<? super T, ? extends T> function) {
            throw new UnsupportedOperationException();
        }


        public int[] toSortedArray() {
            return ObjIntHashMap.this.toSortedArray();
        }


        public MutableIntList toSortedList() {
            return ObjIntHashMap.this.toSortedList();
        }


        public long sum() {
            return ObjIntHashMap.this.sum();
        }


        public int max() {
            return ObjIntHashMap.this.max();
        }


        public int maxIfEmpty(int defaultValue) {
            return ObjIntHashMap.this.maxIfEmpty(defaultValue);
        }


        public int min() {
            return ObjIntHashMap.this.min();
        }


        public int minIfEmpty(int defaultValue) {
            return ObjIntHashMap.this.minIfEmpty(defaultValue);
        }


        public double average() {
            return ObjIntHashMap.this.average();
        }


        public double median() {
            return ObjIntHashMap.this.median();
        }

        /**
         * @since 9.2.
         */

        public MutableIntCollection newEmpty() {
            return IntBags.mutable.empty();
        }

        @Override
        public <V> MutableCollection<V> collect(IntToObjectFunction<? extends V> function) {
            throw new UnsupportedOperationException();
        }
    }

    private class KeysView extends AbstractLazyIterable<K> {

        public void each(Procedure<? super K> procedure) {
            throw new UnsupportedOperationException();
            //ObjIntHashMap.this.forEachKey(procedure);
        }


//        public void forEachWithIndex(ObjectIntProcedure<K> objectIntProcedure) {
//            int index = 0;
//            for (Object key : ObjIntHashMap.this.keys) {
//                if (isNonSentinel(key)) {
//                    objectIntProcedure.value((K) key, index);
//                    index++;
//                }
//            }
//        }
//        public <P> void forEachWith(Procedure2<K, P> procedure, P parameter) {
//            for (Object key : ObjIntHashMap.this.keys) {
//                if (isNonSentinel(key)) {
//                    procedure.value((K) key, parameter);
//                }
//            }
//        }


        public Iterator<K> iterator() {
            return new InternalKeysViewIterator();
        }

        public class InternalKeysViewIterator implements Iterator<K> {
            private int count;
            private int position;


            public K next() {
                if (!this.hasNext()) {
                    throw new NoSuchElementException();
                }

                var keys = ObjIntHashMap.this.keys;
                while (!isNonSentinel(keys[this.position])) {
                    this.position++;
                }
                var result = (K) ObjIntHashMap.this.keys[this.position];
                this.count++;
                this.position++;
                return result;
            }


            public void remove() {
                throw new UnsupportedOperationException("Cannot call remove() on " + this.getClass().getSimpleName());
            }


            public boolean hasNext() {
                return this.count != ObjIntHashMap.this.size();
            }
        }
    }

    private class KeyValuesView extends AbstractLazyIterable<ObjectIntPair<K>> {

        public void each(Procedure<? super ObjectIntPair<K>> procedure) {
            for (var i = 0; i < ObjIntHashMap.this.keys.length; i++) {
                if (isNonSentinel(ObjIntHashMap.this.keys[i])) {
                    procedure.value(PrimitiveTuples.pair((K) ObjIntHashMap.this.keys[i], ObjIntHashMap.this.values[i]));
                }
            }
        }

//
//        public void forEachWithIndex(ObjectIntProcedure<ObjectIntPair<K>> objectIntProcedure) {
//            int index = 0;
//            for (int i = 0; i < ObjIntHashMap.this.keys.length; i++) {
//                if (isNonSentinel(ObjIntHashMap.this.keys[i])) {
//                    objectIntProcedure.value(PrimitiveTuples.pair((K) ObjIntHashMap.this.keys[i], ObjIntHashMap.this.values[i]), index);
//                    index++;
//                }
//            }
//        }


//        public <P> void forEachWith(Procedure2<ObjectIntPair<K>, P> procedure, P parameter) {
//            for (int i = 0; i < ObjIntHashMap.this.keys.length; i++) {
//                if (isNonSentinel(ObjIntHashMap.this.keys[i])) {
//                    procedure.value(PrimitiveTuples.pair((K) ObjIntHashMap.this.keys[i], ObjIntHashMap.this.values[i]), parameter);
//                }
//            }
//        }


        public Iterator<ObjectIntPair<K>> iterator() {
            return new InternalKeyValuesIterator();
        }

        public class InternalKeyValuesIterator implements Iterator<ObjectIntPair<K>> {
            private int count;
            private int position;


            public ObjectIntPair<K> next() {
                if (!this.hasNext()) {
                    throw new NoSuchElementException();
                }

                var keys = ObjIntHashMap.this.keys;
                while (!isNonSentinel(keys[this.position])) {
                    this.position++;
                }
                var result = PrimitiveTuples.pair((K) ObjIntHashMap.this.keys[this.position], ObjIntHashMap.this.values[this.position]);
                this.count++;
                this.position++;
                return result;
            }


            public void remove() {
                throw new UnsupportedOperationException("Cannot call remove() on " + this.getClass().getSimpleName());
            }


            public boolean hasNext() {
                return this.count != ObjIntHashMap.this.size();
            }
        }
    }
}
