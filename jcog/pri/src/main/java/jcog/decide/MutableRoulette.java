package jcog.decide;

import com.google.common.collect.AbstractIterator;
import jcog.Is;
import jcog.TODO;
import jcog.Util;
import jcog.data.list.Lst;
import jcog.pri.Prioritizable;
import jcog.util.ArrayUtil;
import org.eclipse.collections.api.block.function.primitive.FloatToFloatFunction;
import org.eclipse.collections.api.block.function.primitive.IntToFloatFunction;
import org.eclipse.collections.api.block.function.primitive.IntToObjectFunction;
import org.jetbrains.annotations.Nullable;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;
import java.util.random.RandomGenerator;

/**
 * efficient embeddable roulette decision executor.
 * takes a weight-update function applied to the current
 * choice after it has been selected in order to influence
 * its probability of being selected again.
 * <p>
 * if the update function returns zero, then the choice is
 * deactivated and can not be selected again.
 * <p>
 * when no choices remain, it exits automatically.
 */
@Is("Fitness_proportionate_selection") public class MutableRoulette {

    private static final float EPSILON = Prioritizable.EPSILON;

    private int n;
    /**
     * weights of each choice
     */
    private float[] w;
    private final RandomGenerator rng;
    /**
     * weight update function applied between selections to the last selected index's weight
     */
    @Nullable private final FloatToFloatFunction weigh;
    /**
     * current index (roulette ball position)
     */
    private int i;
    /**
     * current weight sum
     */
    private float weightSum;
    /**
     * current # entries remaining above epsilon threshold
     */
    private int remaining;
    private boolean direction;

    /**
     * with no weight modification
     */
    public MutableRoulette(int count, IntToFloatFunction initialWeights, RandomGenerator rng) {
        this(count, initialWeights, null, rng);
    }

    private MutableRoulette(int count, IntToFloatFunction initialWeights, @Nullable FloatToFloatFunction weigh, RandomGenerator rng) {
        this(Util.floatArrayOf(initialWeights, count), count, weigh, rng);
    }

    private MutableRoulette(float[] w, @Nullable FloatToFloatFunction weigh, RandomGenerator rng) {
        this(w, w.length, weigh, rng);
    }

    public MutableRoulette(@Nullable FloatToFloatFunction weigh, RandomGenerator rng) {
        this(ArrayUtil.EMPTY_FLOAT_ARRAY, 0, weigh, rng);
    }

    private MutableRoulette(float[] w, int n, @Nullable FloatToFloatFunction weigh, RandomGenerator rng) {
        this.rng = rng;
        this.weigh = weigh;
        reset(w, n);
    }

    /** constructs and runs entirely in constructor */
    private MutableRoulette(float[] weights, int n, FloatToFloatFunction weigh, RandomGenerator rng, IntPredicate choose) {
        this(weights, n, weigh, rng);
        chooseWhile(choose);
    }

    public static <X> List<X> roulette(Map<X, Double> map, int limit, Random rng) {
        if (map.isEmpty() || limit < 1)
            throw new UnsupportedOperationException();
        var l = new Lst<X>(limit);
        var scoreList = map.entrySet().stream().toList();
        var r = new MutableRoulette(map.size(), s -> scoreList.get(s).getValue().floatValue(), rng);
        for (int i = 0; i < limit; i++)
            l.add(scoreList.get(r.next()).getKey());
        return l;
    }

    private MutableRoulette chooseWhile(IntPredicate choose) {
        while (next(choose)) { }
        return this;
    }

    public MutableRoulette reset(float[] w) {
        return reset(w, w.length);
    }

    public MutableRoulette reset(float[] w, int n) {
        this.w = w;
        this.n = n;
        reweigh();
        return this;
    }

    private static void realloc(int newSize) {
        throw new TODO();
    }

    private MutableRoulette reweigh(IntToFloatFunction initializer) {
        return reweigh(n, initializer);
    }

    public MutableRoulette reweigh(int n, IntToFloatFunction initializer) {
        assert(n>0);

        if (n!=this.n)
            realloc(n);

        float[] w = this.w;
        for (int i = 0; i < n; i++)
            w[i] = initializer.valueOf(i);

        return reweigh();
    }

    public MutableRoulette reweigh() {
        int remaining = this.n;
        float s = 0;

        int nn = remaining;
        float[] w = this.w;
        for (int i = 0; i < nn; i++) {
            float wi = w[i];
            if (wi < 0 || !Float.isFinite(wi))
                throw new RuntimeException("invalid weight: " + wi);

            if (wi < EPSILON) {
                w[i] = 0;
                remaining--;
            } else {
                s += wi;
            }
        }

        if (remaining == 0 || s < remaining * EPSILON) {
//            //flat, choose all equally
//            Arrays.fill(w, Prioritized.EPSILON);
//            s = Prioritized.EPSILON * remaining;
//            this.remaining = this.n;
            this.remaining = 0;
        } else {
            this.remaining = remaining;
        }

        this.weightSum = s;

        int n = this.n;
        if (n > 1 && remaining > 1) {
            int r = rng.nextInt(); //using only one RNG call
            this.direction = r >= 0;
            this.i = (r & 0b01111111111111111111111111111111) % n;
        } else {
            this.direction = true;
            this.i = 0;
        }

        return this;
    }

    private boolean next(IntPredicate select) {
        int n = next();
        return n >= 0 && select.test(n) && remaining > 0;
    }

    private boolean next(IntConsumer select) {
        int n = next();
        if (n == -1)
            return false;
        else {
            select.accept(n);
            return remaining > 0;
        }
    }

    /** returns -1 if none remain */
    public final int next() {
        return switch (remaining) {
            case 0 -> -1;
            case 1 -> next1();
            default -> nextN();
        };
    }

    private int nextN() {
        float distance = Math.max(EPSILON, rng.nextFloat() * weightSum);

        float[] w = this.w;
        int i = this.i;
        float wi;
        int count = this.n;

        int idle = count; //to count extra step
        do {

            wi = w[i = Util.next(i, direction, count)];

            if ((distance -= wi) <= 0)
                break;

            if (idle-- < 0)
                return -1; //HACK emergency bailout

        } while (true);


        if (weigh !=null) {
            float nextWeight = weigh.valueOf(wi);
            if (!validWeight(nextWeight)) {
                w[i] = 0;
                weightSum -= wi;
                remaining--;
            } else if (nextWeight != wi) {
                w[i] = nextWeight;
                weightSum += nextWeight - wi;
            }
        }

        return this.i = i;
    }

    private int next1() {
        float[] w = this.w;
        int count = n;
        for (int i = 0; i < count; i++) {
            float wi = w[i];
            if (wi > EPSILON) {
                if (weigh != null) {
                    float wxNext;
                    float nextWeight = weigh.valueOf(wi);
                    if (validWeight(nextWeight)) {
                        weightSum = wxNext = nextWeight;
                    } else {
                        weightSum = wxNext = 0;
                        remaining = 0;
                    }
                    w[i] = wxNext;
                }
                return i;
            }
        }

        return -1;
        //throw new RuntimeException();
    }

    private static boolean validWeight(float nextWeight) {
        return nextWeight==nextWeight /*!NaN*/ && nextWeight >= EPSILON;
    }

    public float weightSum() {
        return weightSum;
    }

    public int size() {
        return n;
    }

    public void zero(int n) {
        w[n] = 0;
        reweigh();
    }

    public <X> Iterator<X> iterator(IntToObjectFunction<X> o) {
        return new RouletteIterator(this, o);
    }

    private static final class RouletteIterator<X> extends AbstractIterator<X> {
        private final MutableRoulette r;
        private final IntToObjectFunction<X> objGet;

        RouletteIterator(MutableRoulette r, IntToObjectFunction<X> objGet) {
            this.r = r;
            this.objGet = objGet;
        }

        @Nullable
        @Override protected X computeNext() {

            int N = r.next();
            if (N < 0) {
                this.endOfData();
                return null;
            }

            r.zero(N);

            var n = objGet.apply(N);
//                if (n == null)
//                    throw new NullPointerException(); //TODO shouldnt happen
            return n;
        }
    }

}