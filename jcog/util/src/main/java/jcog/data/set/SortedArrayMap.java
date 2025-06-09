package jcog.data.set;

import jcog.util.ArrayUtil;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by andrey on 18.03.2017.
 * https://github.com/andrey-stekov/SortedArrayMap/blob/master/src/main/java/org/andrey/collections/SortedArrayMap.java
 * TODO test
 */
public class SortedArrayMap<K, V> implements NavigableMap<K, V> {
    public static final int DEFAULT_CAPACITY = 10;
    public static final float INCREASE_COEF = 1.5f;

    private Entry<K, V>[] array;
    private Comparator<Entry<K, V>> comparator;
    private Comparator<K> keyComparator;
    private int size = 0;

    private static final SortedArrayMap EMPTY = new SortedArrayMap(null, null);

    public SortedArrayMap(Comparator<K> keyComparator) {
        this(keyComparator, DEFAULT_CAPACITY);
    }

    public SortedArrayMap(Comparator<K> keyComparator, int capacity) {
        array = new Entry[capacity];
        this.keyComparator = keyComparator;
        this.comparator = (a, b) -> keyComparator.compare(a.getKey(), b.getKey());
    }

    private SortedArrayMap(Comparator<K> keyComparator, Entry<K, V>[] array) {
        this.keyComparator = keyComparator;
        this.comparator = (a, b) -> keyComparator.compare(a.getKey(), b.getKey());
        this.array = array;
        this.size = array.length;
    }

    private int binarySearch(Entry<K, V> key) {
        int low = 0;
        int high = size - 1;

        while (low <= high) {
            int mid = (low + high) >>> 1;
            Entry<K, V> midVal = array[mid];
            int cmp = comparator.compare(midVal, key);
            if (cmp < 0)
                low = mid + 1;
            else if (cmp > 0)
                high = mid - 1;
            else
                return mid; // key found
        }
        return -(low + 1);  // key not found.
    }

    public Entry<K, V> lowerEntry(K key) {
        return null;
    }

    public K lowerKey(K key) {
        return null;
    }

    public Entry<K, V> floorEntry(K key) {
        return null;
    }

    public K floorKey(K key) {
        return null;
    }

    public Entry<K, V> ceilingEntry(K key) {
        return null;
    }

    public K ceilingKey(K key) {
        return null;
    }

    public Entry<K, V> higherEntry(K key) {
        return null;
    }

    public K higherKey(K key) {
        return null;
    }

    public Entry<K, V> firstEntry() {
        return size > 0 ? array[0] : null;
    }

    public Entry<K, V> lastEntry() {
        return size > 0 ? array[size - 1] : null;
    }

    public Entry<K, V> pollFirstEntry() {
        if (size == 0) {
            return null;
        }

        Entry<K, V> entry = array[0];
        remove(entry.key);
        return entry;
    }

    public Entry<K, V> pollLastEntry() {
        if (size == 0)
            return null;

        Entry<K, V> entry = array[size - 1];
        remove(entry.key);
        return entry;
    }

    public NavigableMap<K, V> descendingMap() {
        Entry<K, V>[] newArray = Arrays.copyOf(array, size);
        Arrays.sort(array, (a, b)-> -comparator.compare(a, b));
        return new SortedArrayMap<>((a, b)-> -keyComparator.compare(a, b), newArray);
    }

    public NavigableSet<K> navigableKeySet() {
        return new KeySet<>(this);
    }

    public NavigableSet<K> descendingKeySet() {
        return new KeySet<>((SortedArrayMap<K, V>) descendingMap());
    }

    public NavigableMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive) {
        if (size == 0) return EMPTY;

        int fromIndex = binarySearch(fromKey);
        int toIndex = binarySearch(toKey);

        fromIndex = fromIndex < 0 ? -fromIndex - 1 : fromIndex;
        toIndex = toIndex < 0 ? -toIndex - 1 : toIndex;

        fromIndex = fromInclusive ? fromIndex : fromIndex + 1;
        toIndex = toInclusive ? toIndex : toIndex - 1;

        boolean rangeIsCorrect = fromIndex < toIndex && fromIndex >= 0 && toIndex < size;
        if (!rangeIsCorrect) {
            throw new IllegalArgumentException("Incorrect range");
        }

        int newSize = toIndex - fromIndex + 1;

        if (newSize == 0)
            return EMPTY;


        Entry<K, V>[] newArray = new Entry[newSize];
        System.arraycopy(array, fromIndex, newArray, 0, newSize);
        return new SortedArrayMap<>(keyComparator, newArray);
    }

    public NavigableMap<K, V> headMap(K toKey, boolean inclusive) {
        if (size == 0) return EMPTY;

        return subMap(array[0].key, true, toKey, inclusive);
    }

    public NavigableMap<K, V> tailMap(K fromKey, boolean inclusive) {
        if (size == 0) return EMPTY;

        return subMap(fromKey, inclusive, array[size - 1].key, inclusive);
    }

    public Comparator<? super K> comparator() {
        return keyComparator;
    }

    public SortedMap<K, V> subMap(K fromKey, K toKey) {
        return subMap(fromKey, true, toKey, false);
    }

    public SortedMap<K, V> headMap(K toKey) {
        return headMap(toKey, false);
    }

    public SortedMap<K, V> tailMap(K fromKey) {
        return tailMap(fromKey, true);
    }

    public K firstKey() {
        return size > 0 ? array[0].key : null;
    }

    public K lastKey() {
        return size > 0 ? array[size - 1].key : null;
    }

    public int size() {
        return size;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public boolean containsKey(Object key) {
        return binarySearch((K) key) >= 0;
    }

    public boolean containsValue(Object value) {
        return ArrayUtil.indexOf(array, z -> Objects.equals(z.value, value), 0, size)!=-1;
//        return Arrays
//                .stream(array)
//                .filter((e)->e.value == null && value == null || e.value != null && e.value.equals(value))
//                .findAny()
//                .isPresent();
    }

    public V get(Object key) {
        int index = binarySearch((K) key);
        return index >= 0 ? array[index].value : null;
    }

    private void ensureSize(int req) {
        if (req > array.length) {
            array = Arrays.copyOf(array, 1 + (int)(size * INCREASE_COEF));
        }
    }

    public V put(K key, V value) {
        Entry<K, V> entity = new Entry<>(key, value);
        int index = binarySearch(entity);
        int target = index < 0 ? -index - 1 : index;

        if (index >= 0) {
            Entry<K, V> old = array[index];
            array[index].value = value;
            return old.value;
        } else {

            ensureSize(size + 1);

            if (target == size) {
                array[size] = entity;
            } else {
                System.arraycopy(array, target, array, target + 1, size - target);
                array[target] = entity;
            }
            size++;
            return null;
        }
    }

    private int binarySearch(K key) {
        return binarySearch(new Entry<>(key));
    }

    public V remove(Object key) {
        int index = binarySearch((K)key);

        if (index < 0)
            return null;

        V old = array[index].value;
        if (index == size - 1) {
            array[index] = null;
        } else {
            System.arraycopy(array, index + 1, array, index, size - index - 1);
        }
        size--;
        return old;
    }

    public void putAll(Map<? extends K, ? extends V> map) {
        map.entrySet().forEach(e ->
            put(e.getKey(), e.getValue())
        );
    }

    public void clear() {
        Arrays.fill(array, null);  //this.array = new Entry[DEFAULT_CAPACITY];
        this.size = 0;
    }

    public Set<K> keySet() {
        return Arrays.stream(array)
                .map(Entry::getKey)
                .collect(Collectors.toSet());
    }

    public Collection<V> values() {
        return Arrays.stream(array)
                .map(Entry::getValue)
                .toList();
                //.collect(Collectors.toSet());
    }

    public Set<Map.Entry<K, V>> entrySet() {
        return Arrays.stream(array)
                .collect(Collectors.toSet());
    }

    public static class KeySet<K, V> extends AbstractSet<K> implements NavigableSet<K> {
        private SortedArrayMap<K, V> map;

        KeySet(SortedArrayMap<K, V> map) {
            this.map = map;
        }

        @Override
        public K lower(K k) {
            return map.lowerKey(k);
        }

        @Override
        public K floor(K k) {
            return map.floorKey(k);
        }

        @Override
        public K ceiling(K k) {
            return map.ceilingKey(k);
        }

        @Override
        public K higher(K k) {
            return map.higherKey(k);
        }

        @Override
        public K pollFirst() {
            return map.pollFirstEntry().getKey();
        }

        @Override
        public K pollLast() {
            return map.pollLastEntry().getKey();
        }


        @Override
        public Iterator<K> iterator() {
            return map.keySet().iterator();
        }

        @Override
        public NavigableSet<K> descendingSet() {
            return new KeySet<>((SortedArrayMap<K, V>) map.descendingMap());
        }

        @Override
        public Iterator<K> descendingIterator() {
            return descendingSet().iterator();
        }

        @Override
        public NavigableSet<K> subSet(K fromElement, boolean fromInclusive, K toElement, boolean toInclusive) {
            return new KeySet<>((SortedArrayMap<K, V>) map.subMap(fromElement, fromInclusive, toElement, toInclusive));
        }

        @Override
        public NavigableSet<K> headSet(K toElement, boolean inclusive) {
            return new KeySet<>((SortedArrayMap<K, V>) map.headMap(toElement, inclusive));
        }

        @Override
        public NavigableSet<K> tailSet(K fromElement, boolean inclusive) {
            return new KeySet<>((SortedArrayMap<K, V>) map.tailMap(fromElement, inclusive));
        }

        @Override
        public Comparator<? super K> comparator() {
            return map.keyComparator;
        }

        @Override
        public SortedSet<K> subSet(K fromElement, K toElement) {
            return new KeySet<>((SortedArrayMap<K, V>) map.subMap(fromElement, toElement));
        }

        @Override
        public SortedSet<K> headSet(K toElement) {
            return new KeySet<>((SortedArrayMap<K, V>) map.headMap(toElement));
        }

        @Override
        public SortedSet<K> tailSet(K fromElement) {
            return new KeySet<>((SortedArrayMap<K, V>) map.tailMap(fromElement));
        }

        @Override
        public K first() {
            return map.firstKey();
        }

        @Override
        public K last() {
            return map.lastKey();
        }

        @Override
        public int size() {
            return map.size();
        }
    }
    
    public static class Entry<K, V> implements Map.Entry<K, V> {
        private final K key;
        private V value;

        public Entry(K key) {
            this.key = key;
        }

        public Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }

        public V setValue(V value) {
            V old = this.value;
            this.value = value;
            return old;
        }

        @Override
        public String toString() {
            return key + " = " + value;
        }
    }
}
