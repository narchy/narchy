//package jcog.tensor.rl.dqn;
//
//import jcog.Util;
//import jcog.decide.Decide;
//import jcog.decide.DecideSoftmax;
//import jcog.random.XoRoShiRo128PlusRandom;
//import jcog.tensor.Predictor;
//import org.eclipse.collections.api.block.function.primitive.IntIntToObjectFunction;
//import org.jetbrains.annotations.Nullable;
//
//import java.util.Random;
//
///** discretized 1D action space */
//public class QPolicy1D implements Policy {
//
//    final int inputs;
//
//    public final QPolicy q;
//
//    final int actionDiscretization;
//
//    public QPolicy1D(int inputs, int actionDiscretization, IntIntToObjectFunction<Predictor> p) {
//        this.inputs = inputs;
//        this.actionDiscretization = actionDiscretization;
//        this.q = new QPolicy(()->p.value(inputs, actionDiscretization));
//    }
//
//    @Override
//    public double[] learn(@Nullable double[] xPrev, double[] actionPrev, double reward, double[] x, float pri) {
//        double a = actionPrev[0];
//        double[] A = discretize(a);
//        double[] Y = q.learn(xPrev, A, reward, x, pri);
//        double y = undiscretize(Y);
//        return new double[] { y };
//    }
//
//    final Decide decide = new DecideSoftmax(0.5f, new XoRoShiRo128PlusRandom());
//
//    private double undiscretize(double[] y) {
//        float which = decide.applyAsInt(y);
//        return which / (actionDiscretization-1);
//    }
//
//    private double[] discretize(double a) {
//        double[] A = new double[actionDiscretization];
//        A[Util.bin((float)a, actionDiscretization)] = 1;
//        return A;
//    }
//
//    @Override
//    public void clear(Random rng) {
//        q.clear(rng);
//    }
//
//}