package jcog.decide;

import jcog.Is;
import jcog.Util;
import jcog.pri.Prioritized;
import org.eclipse.collections.api.block.function.primitive.IntToFloatFunction;

import java.util.random.RandomGenerator;

/**
 * roulette select
 * TODO immutable 'CompiledRoulette' output which stores a 'prefix sum' that can be binary searched more quickly
 */
@Is("Fitness_proportionate_selection") public enum Roulette { ;

    public static int selectRoulette(RandomGenerator rng, float... w) {
        return selectRoulette(w, w.length, rng);
    }

    /** returns -1 if no option (not any weight==NaN, or non-positive) */
    public static int selectRoulette(float[] w, int n, RandomGenerator rng) {
        double sum = 0;
        int only = -1, count = 0;
        for (int j = 0; j < n; j++) {
            float wj = w[j];
            if (valid(wj)) {
                sum += wj;
                count++;
                only = j;
            }
        }
        return switch (count) {
            case 0  -> rng.nextInt(n); //flat
            case 1  -> only;
            default -> selectN(w, n, rng, sum);
        };
    }

    private static int selectN(float[] w, int n, RandomGenerator rng, double sum) {
        var p = rng.nextFloat();
        return select(w, n, sum, p);
    }

    public static int select(float[] w, int n, double sum, float p) {
        double pos = p * sum;
        int i;
        n--;
        for (i = 0; i < n; i++) {
            if ((pos -= w[i]) <= 0) break;
        }
        return i;
    }

//    /** rng is either RandomGenerator or a FloatSupplier */
//    private static double random(Object rng) {
//        return rng instanceof RandomGenerator r ? r.nextFloat() :
//                ((FloatSupplier)rng).asFloat();
//    }

    /**
     * Returns the selected index based on the weights(probabilities)
     */
    public static int selectRoulette(int weightCount, IntToFloatFunction weight, RandomGenerator rng) {
        return switch (weightCount) {
            case 1 -> 0;//valid(weight.valueOf(0)) ? 0 : -1;
            case 2 -> select2(weight.valueOf(0), weight.valueOf(1), rng);
            default -> selectN(weightCount, weight, rng);
        };
    }

    private static int selectN(int weightCount, IntToFloatFunction weight, RandomGenerator rng) {
        assert (weightCount > 0);
        double weightSum = Util.sumIfPositive(weightCount, weight);
        return weightSum < Prioritized.EPSILON ?
                selectFlat(weightCount, rng) :
                selectRouletteUnidirectionally(weightCount, weight, weightSum, rng);
    }

    private static int selectFlat(int weightCount, RandomGenerator rng) {
        return Util.bin(rng.nextFloat(), weightCount);
    }

    /** returns -1 if no option (not any weight==NaN, or non-positive) */
    public static int selectRouletteCached(int weightCount, IntToFloatFunction weight, RandomGenerator rng) {
//        return selectRouletteCached(weightCount, weight, ((FloatSupplier)(rng::nextFloat)));
//    }
//    public static int selectRouletteCached(int weightCount, IntToFloatFunction weight, RandomGenerator rng) {
        return switch (weightCount) {
            case 1  -> valid(weight.valueOf(0)) ? 0 : -1;
            case 2  -> select2(weight.valueOf(0), weight.valueOf(1), rng);
            default -> selectNcached(weightCount, weight, rng);
        };
    }

    private static int selectNcached(int weightCount, IntToFloatFunction weight, RandomGenerator rng) {
        float[] w = new float[weightCount];
//        int lastValid = -1;
        for (int i = 0; i < weightCount; i++) {
            float wi = weight.valueOf(i);
//            if (valid(wi)) {
                w[i] = wi;
//                lastValid = lastValid == -1 ? i : -2; //first, or > 1
//            }
        }

//        if (lastValid == -1)
//            return -1;
//        else if (lastValid != -2)
//            return lastValid;
//        else
            return selectRoulette(rng, w);
    }

    private static int select2(float rx, float ry, RandomGenerator rng) {
        boolean bx = valid(rx), by = valid(ry);
        if (bx && by) return _select2(rx, ry, rng);
        else if (!bx && !by) return -1;
        else if (bx /*&& !by*/) return 0;
        else return +1;
    }

    private static int _select2(float rx, float ry, RandomGenerator rng) {
        return rng.nextFloat() <=
            (Util.equals(rx, ry, Float.MIN_NORMAL) ?
                0.5f : (((double) rx) / (rx + ry))
            ) ? 0 : 1;
    }

    private static boolean valid(float w) {
        return w > 0;
    }

    private static int selectRouletteUnidirectionally(int n, IntToFloatFunction weight, double weightSum, RandomGenerator rng) {
        double distance = rng.nextFloat() * weightSum;

        int i = selectFlat(n, rng);
        float y = weight.valueOf(i);
        for (int limit = n; (distance -= Math.max((float) 0, y)) > Float.MIN_NORMAL && --limit > 0; ) {

            if (++i == n) i = 0; //wrap-around

        }
        //assert(safetyLimit>=0);  //if (safetyLimit<0) throw new WTF();
        return i;
    }


//    /** not sure if this offers any improvement over the simpler unidirectional iieration.
//     * might also be biased to the edges or middle because it doesnt start at random index though this can be tried */
//    private static int selectRouletteBidirectionally(int count, IntToFloatFunction weight, float weight_sum, FloatSupplier rng) {
//        float x = rng.asFloat();
//        int i;
//        boolean dir;
//        if (x <= 0.5f) {
//            dir = true;
//            i = 0; //count up
//        } else {
//            dir = false;
//            i = count - 1; //count down
//            x = 1 - x;
//        }
//
//        float distance = x * weight_sum;
//
//        int limit = count;
//        while ((distance -= weight.valueOf(i)) > Float.MIN_NORMAL) {
//            if (dir) {
//                if (++i == count) i = 0;
//            } else {
//                if (--i == -1) i = count - 1;
//            }
//            if (--limit==0)
//                throw new WTF();
//        }
//
//        return i;
//    }

}