package jcog.sort;

import jcog.Util;
import jcog.util.ArrayUtil;
import org.eclipse.collections.api.block.function.primitive.FloatFunction;
import org.jetbrains.annotations.Nullable;

/** caches item rank values for fast entry comparison as each entry
 * TODO maybe make autocloseable to enforce .clear (aka .close()) for returning Ranked to pools
 * */
public class RankedN<X> extends TopN<X> {

    /** cached rank/strength/weight/value table; maintained to be synchronized with the items array.
     *  stored negated for use by SortedArray
     *  lazy-init in insert
     */
    @Nullable private float[] value;

    public RankedN(X[] items) {
        super(items);
    }

//    @Override
//    public void clear() {
//        if (size > 0) {
//            Arrays.fill(value, 0, size, 0);
//            super.clear();
//        }
//    }

    @Override
    public void delete() {
        super.delete();
        rank = null;
        value = null;
    }

    public RankedN(X[] buffer, FloatFunction<X> ranking) {
        this(buffer, FloatRank.the(ranking));
    }

    public RankedN(X[] buffer, FloatRank<X> ranking) {
        this(buffer);
        rank(ranking);
    }

    public RankedN<X> rank(FloatRank<X> rank) {
        if (rank!=this.rank) {
            this.rank = rank;
            rerank();
        }
        return this;
    }

    private void rerank() {
        var s = this.size; if (s <= 0) return;

        var items = this.items;
        var value = this.value;
        var nulls = false;
        for (var i = 0; i < s; i++) {
            var vi = floatValueOf(items[i]);
            if (vi == Float.POSITIVE_INFINITY) {
                items[i] = null;
                nulls = true;
            }
            value[i] = vi;
        }

        if (nulls) {
            removeNulls();
            s = size;
        }

        if (s > 1)
            sort(s);

        commit();
    }

    private void sort(int s) {
        QuickSort.quickSort(0, s, this::valueComparator, this::swap);
    }

    @Override
    protected int updateEnd(X x, float itemRank) {
        return insertValueIndex(super.updateEnd(x, itemRank), itemRank);
    }

    @Override
    protected int updateInternal(int index, X x, float itemRank, int sizeBefore) {
        return insertValueIndex(super.updateInternal(index, x, itemRank, sizeBefore), itemRank);
    }

    private int insertValueIndex(int i, float itemRank) {
        if (i == -1) return -1;

        var v = this.value;
        if (v == null)
            v = this.value = new float[capacity()]; //lazy-init

        var shift = size - 1 - i;
        if (shift > 0)
            System.arraycopy(v, i, v, i + 1, shift );

        v[i] = itemRank;
        return i;
    }

    @Override
    protected final float value(int item) {
        return -value[item];
    }

    /** only for use by sorted Array */
    @Override protected final float valueAt(int item, FloatFunction<X> cmp) {
        return value[item];
    }


    @Override
    public X remove(int index) {
        var totalOffset = this.size - index - 1;
        if (totalOffset < 0)
            return null;

        var list = this.items;
        var previous = list[index];
        if (totalOffset > 0) {
            var value = this.value;
            System.arraycopy(value, index + 1, value, index, totalOffset);
            System.arraycopy(list, index + 1, list, index, totalOffset);
        }
        --size;
        value[size] = 0;
        list[size] = null;
        commit();
        return previous;
    }


    @Override
    protected final void rejectExisting(X e) {
    }

    @Override
    protected final void rejectOnEntry(X e) {

    }

    private void swap(int a, int b) {
        ArrayUtil.swapObjFloat(items, value, a, b);
    }

    private int valueComparator(int a, int b) {
        if (a == b)
            return 0;
        var v = this.value;
        return Util.compare(v[b], v[a]);
        //return Float.compare(v[b], v[a]);
    }

//    @Nullable
//    public X getRoulette(FloatSupplier rng, Predicate<X> filter, boolean cached) {
//        int n = size();
//        if (n == 0)
//            return null;
//        if (n == 1)
//            return get(0);
//
//        IntToFloatFunction select = i -> filter.test(get(i)) ? (cached ? rankCached(i) : rank.rank(get(i))) : Float.NaN;
//        return get( //n < 8 ?
//                this instanceof RankedN ?
//                        Roulette.selectRoulette(n, select, rng) : //RankedTopN acts as the cache
//                        Roulette.selectRouletteCached(n, select, rng) //must be cached for consistency
//        );
//
//    }
}