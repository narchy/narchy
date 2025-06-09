package jcog.data.map;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * most recently used cache based on (non-thread_safe) LinkedHashMap
 */
public class MRUMap<K, V> extends LinkedHashMap<K, V> {

    protected int capacity;

    private static final float LOAD_FACTOR_DEFAULT =
        0.75f;
        //0.85f;
        //1;

    public MRUMap(int capacity) {
        this(capacity, LOAD_FACTOR_DEFAULT);
    }

    public MRUMap(int capacity, float loadFactor) {
        this(capacity, 0, loadFactor);
    }

    public MRUMap(int capacity, int initialCapacity, float loadFactor) {
        super(initialCapacity, loadFactor, true);
        setCapacity(capacity);
    }

    public final void setCapacity(int capacity) {
        this.capacity = capacity;
    }

    public final int capacity() {
        return capacity;
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<K, V> entry) {
        if (this.size() > capacity) {
            onEvict(entry);
//            if (entry.getValue() instanceof AutoCloseable a) {
//                try {
//                    a.close();
//                } catch (Exception e) {
//                    throw new RuntimeException(e);
//                }
//            }
            return true;
        } else
            return false;
    }

    protected void onEvict(Map.Entry<K, V> entry) {

    }
}