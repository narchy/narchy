package jcog.tensor;

import jcog.Is;
import jcog.TODO;

import java.util.Random;
import java.util.function.UnaryOperator;

/** Map-like interface for learning associations of numeric vectors */
@Is("Autoassociative_memory")
public interface Predictor extends UnaryOperator<double[]>  {

    /*synchronized*/ double[] put(double[] x, double[] y, float pri);

    /** predict: analogous to Map.get */
    double[] get(double[] x);

    void clear(Random rng);

    @Override
    default /* final */ double[] apply(double[] x) { return get(x); }

    /** LERP toward another predictor, assuming it's the same model
     *  @param rate 0..1
     */
    default void copyLerp(Predictor p, float rate) {
        throw new TODO();
    }

//    public final double[] put(Tensor x, double[] y, float pri) {
//        return put(x.doubleArrayShared(), y, pri);
//    }

}