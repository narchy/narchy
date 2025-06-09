package jcog.sort;

import jcog.data.bit.MetalBitSet;
import jcog.data.list.Lst;
import jcog.data.map.ObjIntHashMap;
import org.eclipse.collections.api.block.function.primitive.DoubleToDoubleFunction;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.random.RandomGenerator;

public class PrioritySet<X> implements AutoCloseable {
    private final ObjIntHashMap<X> itemToIndex;
    private final Object[] items;
    private final double[] pri;

    /** used indices */
    private final MetalBitSet used;

    private int size;
    public final int capacity;
    private double priTotal;

    private int priMinIndex, priMaxIndex;

    public PrioritySet(int capacity) {
        if (capacity <= 0)
            throw new IllegalArgumentException("Capacity must be greater than 0");

        this.capacity = capacity;
        this.itemToIndex = new ObjIntHashMap<>(capacity);
        this.items = new Object[capacity];
        this.pri = new double[capacity];
        this.used = MetalBitSet.bits(capacity);
        clear();
    }

    public void clear() {
        if (size > 0)
            clearItems();
        priTotal = 0;
        priMaxIndex = priMinIndex = -1;
    }

    private void clearItems() {
        Arrays.fill(items, 0, size, null);
        Arrays.fill(pri, 0, size, 0);
        itemToIndex.clear();
        used.clear();
        size = 0;
    }

    public double pri(X item) {
        int index = itemToIndex.getIfAbsent(item, -1);
        return index == -1 ? Double.NaN : pri[index];
    }

    public boolean put(X item, double priority) {
        validPri(priority);
        if (item == null) throw new NullPointerException();
        
        int existing = itemToIndex.getIfAbsent(item, -1);
        if (existing!=-1) {
            pri(existing, priority);
            return true;
        } else
            return insert(item, priority);
    }

    public void forEach(Consumer<X> x) {
        int c = capacity;
        var ii = items;
        for (int i = used.next(true, 0, c); i >= 0; i = used.next(true, i + 1, c))
            x.accept((X) ii[i]);
    }


    private boolean insert(X item, double priority) {
        if (size >= capacity) {
            if (priority < priLowest())
                return false;
            removeLowestItem();
        }

        _insert(item, priority);
        return true;
    }

    private void _insert(X item, double p) {
        int index = used.next(false, 0, capacity);
        items[index] = item;
        pri[index] = p;
        itemToIndex.put(item, index);
        used.set(index);
        size++;
        priTotal += p;
        updatePriIndex(index, p);
    }


    public boolean remove(X item) {
        if (item == null) return false;
        int index = itemToIndex.removeKeyIfAbsent(item, -1);
        if (index == -1) return false;
        priTotal = Math.max(priTotal - pri[index], 0);
        pri[index] = 0;
        items[index] = null;
        used.clear(index);
        size--;
        if (index == priMinIndex) priMinIndex = lowestPriIndex();
        if (index == priMaxIndex) priMaxIndex = highestPriIndex();
        return true;
    }

    /** TODO optimize and avoid pri calls */
    public boolean priMul(X item, double f) {
        return pri(item, p -> p * f);
    }

    public boolean pri(X item, DoubleToDoubleFunction update) {
        int index = itemToIndex.getIfAbsent(item, -1);
        if (index != -1) {
            pri(index, update.valueOf(pri[index]));
            return true;
        } else {
            return false;
        }
    }

    public boolean pri(X item, double newPriority) {
        validPri(newPriority);

        int index = itemToIndex.getIfAbsent(item, -1);
        if (index != -1) {
            pri(index, newPriority);
            return true;
        } else {
            return false;
        }
    }

    private static void validPri(double newPriority) {
        if (newPriority <= 0)
            throw new IllegalArgumentException("Priority must be greater than 0");
    }

    private void pri(int index, double newPriority) {
        var delta = newPriority - pri[index];
        if (delta!=0) {
            priTotal += delta;
            updatePriIndex(index, pri[index] = newPriority);
        }
    }

    /** roulette select (sample) */
    public int getIndex(RandomGenerator rng) {
        int s = size;
        if (s == 0) return -1;
        double randomValue = rng.nextFloat() * priTotal;
        double cumulativeProb = 0;
        int last = -1;
        int c = capacity;
        for (int i = used.next(true, 0, c); i >= 0; i = used.next(true, i + 1, c)) {
            cumulativeProb += pri[i];
            if (randomValue <= cumulativeProb)
                return i;
            last = i;
        }
        return last;
    }

    public X get(RandomGenerator rng) {
        return itemSafe(getIndex(rng));
    }

    public X item(int i) {
        return (X) items[i];
    }

    public boolean removeHighestItem() {
        var i = itemHighest();
        if (i!=null) { remove(i); return true; }
        else return false;
    }

    public boolean removeLowestItem() {
        var i = itemLowest();
        if (i!=null) { remove(i); return true; }
        else return false;
    }

    private @Nullable X itemSafe(int m) {
        return m < 0 ? null : item(m);
    }

    @Nullable public X itemHighest() {
        return itemSafe(priMaxIndex);
    }

    @Nullable public X itemLowest() {
        return itemSafe(priMinIndex);
    }

    private void updatePriIndex(int index, double p) {
        if (priMinIndex == -1 || p < pri[priMinIndex])
            priMinIndex = index;
        if (priMaxIndex == -1 || p > pri[priMaxIndex])
            priMaxIndex = index;
    }

    public double priHighest() {
        return pri[priMaxIndex];
    }

    public double priLowest() {
        return pri[priMinIndex];
    }

    private int lowestPriIndex() {
        int priMinIndex = -1;
        double priMin = Double.POSITIVE_INFINITY;
        int c = capacity;
        for (int i = used.next(true, 0, c); i >= 0; i = used.next(true, i + 1, c)) {
            double pi = pri[i];
            if (pi < priMin) {
                priMin = pi;
                priMinIndex = i;
            }
        }
        return priMinIndex;
    }

    private int highestPriIndex() {
        int priMaxIndex = -1;
        double priMax = Double.NEGATIVE_INFINITY;
        int c = capacity;
        for (int i = used.next(true, 0, c); i >= 0; i = used.next(true, i + 1, c)) {
            double pi = pri[i];
            if (pi > priMax) {
                priMax = pi;
                priMaxIndex = i;
            }
        }
        return priMaxIndex;
    }

    public int size() { return size; }
    public boolean isEmpty() { return size == 0; }
    public boolean contains(X item) { return itemToIndex.containsKey(item); }

    public List<X> toList() {
        Lst<X> l = new Lst<>(size);
        forEach(l::addFast);
        return l;
    }

    public X removeLast() {
        var x = itemLowest();
        remove(x);
        return x;
    }

    @Override
    public void close() {
        clear();
    }

}