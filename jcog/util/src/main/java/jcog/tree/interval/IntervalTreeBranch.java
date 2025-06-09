package jcog.tree.interval;

import org.jetbrains.annotations.Nullable;

import java.util.Collection;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class IntervalTreeBranch<K extends Comparable<? super K>, V> implements
        IntervalTreeNode<K, V> {

    private IntervalTreeNode<K, V> left;
    private IntervalTreeNode<K, V> right;
    private Between<K> key;

    public IntervalTreeBranch(IntervalTreeNode<K, V> left, IntervalTreeNode<K, V> right) {
        this.left = left;
        this.right = right;
        updateKeyRange();
    }

    private void updateKeyRange() {
		key = left != null ? new Between<>(left.getLow(), (right == null || left.getHigh().compareTo(right.getHigh()) > 0 ? left : right).getHigh()) : new Between<>(right.getLow(), right.getHigh());
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public @Nullable IntervalTreeNode<K, V> getLeft() {
        return left;
    }

    @Override
    public @Nullable IntervalTreeNode<K, V> getRight() {
        return right;
    }

    void setLeft(IntervalTreeNode<K, V> left) {
        this.left = left;
    }

    void setRight(IntervalTreeNode<K, V> right) {
        this.right = right;
    }

    @Override
    public boolean contains(K point) {
        return key.contains(point);
    }

    @Override
    public boolean contains(Between<K> interval) {
        return key.contains(interval);
    }

    @Override
    public boolean overlaps(K low, K high) {
        return key.overlaps(low, high);
    }

    @Override
    public boolean overlaps(Between<K> interval) {
        return key.overlaps(interval);
    }

    @Override
    public  K getLow() {
        return key.getLow();
    }

    @Override
    public  K getHigh() {
        return key.getHigh();
    }

    @Override
    public  V getValue() {
        throw new UnsupportedOperationException();
    }

    @Override
    public  IntervalTreeNode<K, V> put(Between<K> key, V value) {
        if (right == null) {
            if (left.getLow().compareTo(key.getLow()) < 0) {
                right = left;
                left = new IntervalTreeLeaf<>(key, value);
            } else {
                right = new IntervalTreeLeaf<>(key, value);
            }
        } else {
            if (right.getLow().compareTo(key.getLow()) < 0) {
                right = right.put(key, value);
            } else {
                left = left == null ? new IntervalTreeLeaf<>(key, value) : left.put(key, value);
            }
        }
        updateKeyRange();
        return this;
    }

    @Override
    public void getOverlap(Between<K> range, Consumer<V> accumulator) {
        if (left != null && left.overlaps(range)) {
            left.getOverlap(range, accumulator);
        }
        if (right != null && right.overlaps(range)) {
            right.getOverlap(range, accumulator);
        }
    }

    @Override
    public void getOverlap(Between<K> range, Collection<V> accumulator) {
        if (left != null && left.overlaps(range)) {
            left.getOverlap(range, accumulator);
        }
        if (right != null && right.overlaps(range)) {
            right.getOverlap(range, accumulator);
        }
    }

    @Override
    public @Nullable V getEqual(Between<K> range) {
        if (left != null && left.overlaps(range)) {
            return left.getEqual(range);
        }
        if (right != null && right.overlaps(range)) {
            return right.getEqual(range);
        }
        return null;
    }

    @Override
    public @Nullable V getContain(Between<K> range) {
        if (left != null && left.overlaps(range)) {
            return left.getContain(range);
        }
        if (right != null && right.overlaps(range)) {
            return right.getContain(range);
        }
        return null;
    }

    @Override
    public void getContain(Between<K> range, Collection<V> accumulator) {
        if (left != null && left.contains(range)) {
            left.getContain(range, accumulator);
        }
        if (right != null && right.contains(range)) {
            right.getContain(range, accumulator);
        }
    }
    @Override
    public void forEachContainedBy(Between<K> range, BiConsumer<Between<K>,V> accumulator) {
        if (left != null && left.overlaps(range)) {
            left.forEachContainedBy(range, accumulator);
        }
        if (right != null && right.overlaps(range)) {
            right.forEachContainedBy(range, accumulator);
        }
    }
    @Override
    public void searchContainedBy(Between<K> range, Collection<V> accumulator) {
        if (left != null && left.overlaps(range)) {
            left.searchContainedBy(range, accumulator);
        }
        if (right != null && right.overlaps(range)) {
            right.searchContainedBy(range, accumulator);
        }
    }

    @Override
    public int size() {
        return 1 + (left != null ? left.size() : 0) + (right != null ? right.size() : 0);
    }

    @Override
    public void values(Collection<V> accumulator) {
        if (left != null) {
            left.values(accumulator);
        }
        if (right != null) {
            right.values(accumulator);
        }
    }

    @Override
    public @Nullable IntervalTreeNode<K, V> remove(V value) {
        if (left != null) {
            left = left.remove(value);
        }
        if (right != null) {
            right = right.remove(value);
        }
        return removeCleanup();
    }

    private IntervalTreeNode<K, V> removeCleanup() {
        if (left == null) {
            return right;
        }
        if (right == null) {
            return left;
        }
        updateKeyRange();
        return this;
    }

    @Override
    public void entrySet(Set<Entry<Between<K>, V>> accumulator) {
        if (left != null) {
            left.entrySet(accumulator);
        }
        if (right != null) {
            right.entrySet(accumulator);
        }
    }

    @Override
    public boolean containsValue(V value) {
        return (left != null && left.containsValue(value)) || (right != null && right.containsValue(value));
    }

    @Override
    public void keySet(Set<Between<K>> accumulator) {
        if (left != null) {
            left.keySet(accumulator);
        }
        if (right != null) {
            right.keySet(accumulator);
        }
    }

    @Override
    public boolean containedBy(Between<K> interval) {
        return interval.contains(key);
    }



    @Override
    public final Between<K> getRange() {
        return key;
    }

    @Override
    public int maxHeight() {
		return (left != null ? right != null ? Math.max(left.maxHeight(), right.maxHeight()) : left.maxHeight() : right.maxHeight()) + 1;
    }

    @Override
    public void averageHeight(Collection<Integer> heights, int currentHeight) {
        if (left != null) {
            left.averageHeight(heights, currentHeight + 1);
        }
        if (right != null) {
            right.averageHeight(heights, currentHeight + 1);
        }
    }

    @Override
    public @Nullable IntervalTreeNode<K, V> removeAll(Collection<V> values) {
        if (left != null) {
            left = left.removeAll(values);
        }
        if (right != null) {
            right = right.removeAll(values);
        }
        if (left == null && right == null) {
            return null;
        }
        updateKeyRange();
        return this;
    }

    @Override
    public @Nullable IntervalTreeNode<K, V> removeOverlapping(Between<K> range) {
        if (left != null && left.overlaps(range)) {
            left = left.removeOverlapping(range);
        }
        if (right != null && right.overlaps(range)) {
            right = right.removeOverlapping(range);
        }
        return removeCleanup();
    }

    @Override
    public @Nullable IntervalTreeNode<K, V> removeContaining(Between<K> range) {
        if (left != null && left.contains(range)) {
            left = left.removeContaining(range);
        }
        if (right != null && right.contains(range)) {
            right = right.removeContaining(range);
        }
        return removeCleanup();
    }

    @Override
    public @Nullable IntervalTreeNode<K, V> removeContainedBy(Between<K> range) {
        if (left != null && left.overlaps(range)) {
            left = left.removeContainedBy(range);
        }
        if (right != null && right.overlaps(range)) {
            right = right.removeContainedBy(range);
        }
        return removeCleanup();
    }

}