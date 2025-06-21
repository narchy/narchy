//package jcog.tensor.rl.dqn;
//
//import jcog.tensor.Predictor;
//import org.eclipse.collections.api.block.function.primitive.IntIntToObjectFunction;
//import org.jetbrains.annotations.Nullable;
//
//import java.util.Random;
//
///**
// * implements something like the IDQ in:
// *    https://arxiv.org/pdf/1711.08946.pdf
// */
//public class BranchedPolicy implements Policy {
//
//    final int inputs;
//    public final Policy[] actions;
//
//    public BranchedPolicy(int inputs, int actions, int actionDiscretization, IntIntToObjectFunction<Predictor> p) {
//        this.inputs = inputs;
//        this.actions = new Policy[actions];
//        for (int a = 0; a < actions; a++) {
//            QPolicy1D qa = new QPolicy1D(
//                inputs, actionDiscretization, p
//            );
////            QPolicySimul qa = new QPolicySimul( //???
////                inputs, 1, p
////            );
////            qa.q.plan.set( //HACK
////                //0.05f
////                0.9f
////                //0.99f
////            );
//            this.actions[a] = qa;
//        }
//    }
//
//    @Override
//    public void clear(Random rng) {
//        for (var a : actions)
//            a.clear(rng);
//    }
//
//    @Override
//    public double[] learn(@Nullable double[] xPrev, double[] actionPrev, double reward, double[] x, float pri) {
//        double[] y = new double[actionPrev.length];
//        for (int i = 0, actionsLength = actions.length; i < actionsLength; i++) {
//            Policy a = actions[i];
//            y[i] = a.learn(xPrev, new double[] { actionPrev[i] }, reward, x, pri)[0];
//        }
//        return y;
//    }
//}
