package jcog.data.list.table;

import org.jetbrains.annotations.Nullable;

import java.util.function.Consumer;

/**
 * more abstract than a bag, nearly a Map
 */
public interface Baglike<K, V> extends Iterable<V> {


    @Nullable V get(Object key);

    @Nullable V remove(K key);

    int size();

    int capacity();

    void clear();

    default boolean isFull() {
        return size() >= capacity();
    }

    /**
     * iterates in sorted order
     */
    void forEachKey(Consumer<? super K> each);

}
