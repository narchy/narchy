package jcog.pri.bag.impl;

import jcog.Util;
import jcog.data.list.Lst;
import jcog.pri.PriMap;
import jcog.pri.Prioritizable;
import jcog.pri.Prioritized;
import jcog.pri.bag.Bag;
import jcog.pri.bag.util.ProxyBag;
import jcog.pri.op.PriMerge;
import jcog.signal.NumberX;
import org.eclipse.collections.api.block.function.primitive.FloatFunction;
import org.eclipse.collections.api.block.procedure.primitive.FloatProcedure;
import org.jetbrains.annotations.Nullable;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * concurrent buffering bag wrapper
 * TODO optional normalize results by a scaling factor to target mass
 */
public class BufferedBag<X, Y extends Prioritized & Prioritizable> extends ProxyBag<X, Y> {

    /**
     * pre-bag accumulating buffer
     */
    public final PriMap<Y> pre;
    private final AtomicBoolean busy = new AtomicBoolean(false);


    public BufferedBag(Bag<X, Y> bag) {
        this(bag, new PriMap<>());
    }

    public BufferedBag(Bag<X, Y> bag, PriMap<Y> pre) {
        super(bag);
        super.merge(bag.merge()); //by default.  changing this later will set pre and bag's merges
        this.pre = pre;
        this.bagPut = bag::put;
    }

    @Override
    public void clear() {
        pre.clear();
        super.clear();
    }

    @Override public boolean isEmpty() {
        return pre.isEmpty() && super.isEmpty();
    }
    @Override public int size() {
        return Math.max(pre.size(),super.size());
    }

    private final Lst<Y> sortBuffer = new Lst<>();

    @Override
    public final void commit(@Nullable Consumer<? super Y> update) {
        if (Util.enterAlone(busy)) {
            try {

                bag.commit(update);

                //pre.drain(bagPut, null);
                pre.drainSorted(bagPut, null, sortBuffer, priSort);

                if (postCommit)
                    bag.commit(null);

            } finally {
                Util.exitAlone(busy);
            }
        }

    }

    @Override
    public final Y put(Y y, @Nullable NumberX overflowingIgnored) {
        return pre.put(y, merge(), pressurize);
    }

    public Bag<X,Y> merge(PriMerge nextMerge) {
        super.merge(nextMerge);
        bag.merge(nextMerge);
        return this;
    }

    private static final boolean highOrLowFirst = true;
    /** highOrLowFirst==true: opportunity for weaks, false: retain strong, elitism */
    private static final FloatFunction<Prioritized> priSort = highOrLowFirst ?
            p -> -p.priElseZero()  :
            p -> +p.priElseZero();
    private final Consumer<Y> bagPut;
    private static final boolean pressureOnInput = false, postCommit = false;
    private final FloatProcedure pressurize = pressureOnInput ? this::pressurize : null;

}