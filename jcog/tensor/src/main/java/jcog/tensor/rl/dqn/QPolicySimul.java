package jcog.tensor.rl.dqn;

import jcog.Util;
import jcog.data.DistanceFunction;
import jcog.decide.Decide;
import jcog.decide.DecideSoftmax;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Predictor;
import org.eclipse.collections.api.block.function.primitive.IntIntToObjectFunction;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

/** https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py
 *  https://www.reddit.com/r/reinforcementlearning/search/?q=multiple%20actions&restrict_sr=1&sr_nsfw=&include_over_18=1
 *  https://www.reddit.com/r/reinforcementlearning/comments/gwxc5c/dqns_and_action_spaces/
 * */
public class QPolicySimul implements Policy {

    final int inputs, actions;
    public final QPolicy q;

    /** TODO return by ActionEncoder method  */
    @Deprecated private final int actionDiscretization = 2;

    /** TODO return by ActionEncoder method  */
    @Deprecated private final int actionsInternal;

    /** maps action vector to a distribution of virtual d^n basis vectors */
    interface ActionEncoder {
        double[] actionEncode(double[] x, int actionsInternal);
    }
    interface ActionDecoder {
        double[] actionDecode(double[] z, int actions);
    }

    private ActionEncoder ae =
        new DistanceActionEncoder();
        //new FuzzyDistanceActionEncoder();
        //new BinaryActionEncoder();

    private ActionDecoder ad =
        new SoftDistanceActionDecoder(8);
        //new LinearCombinationActionDecoder();
        //new FuzzyDistanceActionDecoder();
        //new BinaryActionDecoder(new DecideSoftmax(0.1f, new XoRoShiRo128PlusRandom()));
        //new NoisyLinearCombinationActionDecoder();


    public QPolicySimul(int inputs, int actions, IntIntToObjectFunction<Predictor> p) {
        this.inputs = inputs;
        this.actions = actions;
        this.actionsInternal = (int)Math.pow(actionDiscretization, actions);
        Supplier<Predictor> ps = () -> p.value(inputs, actionsInternal);
        this.q = new QPolicy(ps, null);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + q.p + ")";
    }

    @Override
    public void clear(Random rng) {
        q.clear(rng);
    }

    @Override
    public double[] learn(double[] xPrev, double[] actionPrev, double reward, double[] x, float pri) {
        return ad.actionDecode(q.learn(xPrev,
                ae.actionEncode(actionPrev, actionsInternal),
                reward, x, pri), actions);
    }


    /** HACK 2-ary thresholding */
    public static class BinaryActionEncoder implements ActionEncoder {

        private final Decide decide =
                new DecideSoftmax(0.1f, new XoRoShiRo128PlusRandom());
        //new DecideRoulette(new XoRoShiRo128PlusRandom());


        @Override
        public double[] actionEncode(double[] x, int actionsInternal) {
            //assert (actionDiscretization == 2);
            double[] z = new double[actionsInternal];
            int actions = x.length;
            int Z = 0;
            for (int i = 0; i < actions; i++) {
                double a = x[i];
                if (a >= 0.5)
                    Z |= 1 << i;
            }
            z[Z] = 1;
            return z;
        }
    }

    public static class BinaryActionDecoder implements ActionDecoder {
        private final Decide decide;

        public BinaryActionDecoder(Decide decide) {
            this.decide = decide;
        }

        @Override public double[] actionDecode(double[] z, int actions) {
            //System.out.println(n2(z));
            //HACK 2-ary thresholding
            //assert (actionDiscretization == 2);
            return actionDecodeDecide(z, actions, decide);
        }


    }
    private static double[] actionDecodeDecide(double[] z, int actionsExternal, Decide decide) {
        int Z = decide.applyAsInt(z);
        double[] y = new double[actionsExternal];
        for (int i = 0; i < actionsExternal; i++) {
            if ((Z & (1 << i)) != 0)
                y[i] = 1;
        }
        return y;
    }

    /** HACK 2-ary thresholding */
    public static class DistanceActionEncoder implements ActionEncoder {

        //private final float decodeSpecificity = 1;

        private final float temperature =
            0.5f;
            //1;
            //0.25f;
            //2;
            //0.1f;

        private boolean normalizeManhattanOrCartesian = true;
        @Override
        public double[] actionEncode(double[] x, int actionsInternal) {
            //assert (actionDiscretization == 2);
            double[] z = new double[actionsInternal];
            int actions = x.length;
            double distMax = 1; //z.length;
            //TODO refine
            for (int i = 0; i < actionsInternal; i++) {
                double d = dist(x, idealDecode(i, actions));
//                double weight =
//                    actionsInternal == 2 ? Math.max(0, 1-d) /* clean */ : 1 / sqr(1 + d * actionsInternal); /*blur intense*/
//                    //(distMax - d)/distMax;
//                    //Math.max(0, 1-d); //clean
//                    //1 / (1 + d * actionsInternal); //blur
//                    //1 / sqr(1 + d * actionsInternal); //blur intense
//                    //1 / (1 + d * actions);
//                    //Math.max(0, 1-d/actions);
//                    //sqr(Math.max(0, 1-d/actions));
//                    //Math.max(0, 1-sqr(d/actions));
//                    //Math.max(0, 1-d*2);
//                    //1 / (1 + d);
//                    //1 / sqr(1 + d);
//                    //1 / sqr(1 + d * actions);
////                z[i] = weight;
//                zSum += weight;
                z[i] = d;
            }

//            double zMax = Util.max(z), zMin = Util.min(z);
            for (int i = 0; i < z.length; i++) {
//                z[i] = Util.normalize(z[i], zMin, zMax);
                z[i] = 1.0 / (1.0 + Math.pow(z[i] * actionsInternal, 2));
            }

//            /* filter equal-and-opposites */
//            for (int i = 0; i < actionsInternal; i++) {
//                if (z[i] <= Float.MIN_NORMAL) continue;
//
//                next: for (int j = i+1; j < actionsInternal; j++) {
//                    float complete = 1 - Float.MIN_NORMAL;
//                    boolean
//                        domI = z[i] >= complete && z[j] < complete,
//                        domJ = z[j] >= complete && z[i] < complete,
//                        equal = Util.equals(z[i], z[j], Float.MIN_NORMAL);
//                    if (equal || domI || domJ) {
//                        double[]
//                            I = idealDecode(i, actions),
//                            J = idealDecode(j, actions);
//                        for (int k = 0; k < I.length; k++) {
//                            if (!Util.equals(I[k], 1 - J[k])) continue next;
//                        }
//                        if (equal)
//                            z[i] = z[j] = 0; //equal and opposite: zero
//                        else if (domI)
//                            z[j] = 0; //I dominates J
//                        else if (domJ)
//                            z[i] = 0; //J dominates I
//
//                    }
//                }
//            }



            double zSum = Util.sum(z);
            if (zSum > Float.MIN_NORMAL) {
                if (normalizeManhattanOrCartesian)
                    Util.mul(z, 1 / zSum);
                else {
                    Util.normalizeCartesian(z, z.length, Double.MIN_NORMAL);
                }
            }

//            double diff = DistanceFunction.distanceManhattan(x, actionDecode(z, actions));
//            System.out.println("diff: " + diff);

            //System.out.println(n4(x) + "\t->\t" + n4(z) + "\t=\t" + n4(actionDecode(z, actions)));
            return z;
        }

//        /** experimental */
//        private double[] actionEncode0(double[] x, int actionsInternal) {
//            //assert (actionDiscretization == 2);
//            int actionsExternal = x.length;
//            double[] z = new double[actionsInternal];
//            int actions = x.length;
////            double zSum = 0;
//            for (int i = 0; i < actionsInternal; i++) {
//                double d = dist(x, idealDecode(i, actions));
//                z[i] = d;
////                double weight =
////                        //Math.max(0, 1 - d); //clean
////                        //1 / (1 + d * actionsInternal); //blur
////                        1 / sqr(1 + d * actionsInternal); //blur intense
////                        //1 / (1 + d * actions);
////                        //Math.max(0, 1-d/actions);
////                        //sqr(Math.max(0, 1-d/actions));
////                        //Math.max(0, 1-sqr(d/actions));
////                        //Math.max(0, 1-d*2);
////                        //1 / (1 + d);
////                        //1 / sqr(1 + d);
////                        //1 / sqr(1 + d * actions);
////
////                z[i] = weight;
////                zSum += weight;
//            }
//
//            Util.normalizeUnit(z);
//            for (int i = 0; i < z.length; i++)
//                z[i] = 1 - z[i]; //dist -> weight
//            Util.normalizeHamming(z, Double.MIN_NORMAL);
//
//            //double zMin = Util.min(z), zMax = Util.max(z);
//
////            if (zSum > Float.MIN_NORMAL) {
////                Util.mul(1 / zSum, z);
////
//////                //0..1 -> -1..+1
//////                for (int i = 0; i < z.length; i++)
//////                    z[i] = Fuzzy.polarize(z[i]);
////            } else {
////                Arrays.fill(z, 1.0 / actionsInternal);
////            }
//
//            return z;
//        }

        /**
         * TODO abstract for distance function parameter
         */
        private double dist(double[] x, double[] y) {
            //return DistanceFunction.distanceManhattan(x, y);
            return DistanceFunction.distanceCartesian(x,y); //may be too far to reach max(0, 1-x) for all-in-between points
        }


//        /** TODO still not perfect */
//        public double[] actionEncode2(double[] x, int actionsInternal) {
//            int actions = x.length;
//            double[] z = new double[actionsInternal];
//            double[] xDelta = x.clone();
//            double[] xSum = new double[x.length];
//
//            var rng = ThreadLocalRandom.current();
//
//            int jMax = actionsInternal;
//            for (int j = 0; j < jMax; j++) {
//                int best = -1; double bestDist = Double.POSITIVE_INFINITY;
//                int iOffset = rng.nextInt(actionsInternal); //for fairness
//                for (int _i = 0; _i < actionsInternal; _i++) {
//                    int i = (iOffset + _i)%actionsInternal;
//                    if (z[i] > 0) continue; //already added
//
//                    double[] yi = idealDecode(i, actions);
//                    double dist = DistanceFunction.distanceCartesian(xSum, yi);
//                    if (dist < bestDist) {
//                        bestDist = dist; best = i;
//                    }
//                }
//                if (best < 0) break; //HACK
//
//                double yScale;
//                double[] yi = idealDecode(best, actions);
//                if (j == 0) {
//                    yScale = 1;
//                } else {
//                    yScale =
//                            //Util.max(xx);
//                            vectorProject(xDelta, yi);
//                }
//                if (yScale!=yScale)
//                    break; //HACK
//                z[best] = yScale;
//                Util.mul(yScale, yi);
//
//                for (int i = 0; i < actions; i++) {
//                    xDelta[i] -= yi[i];
//                    xSum[i] += yi[i];
//                }
//                if (len(xDelta) < Float.MIN_NORMAL)
//                    break;
//            }
//
//            //Util.normalizeCartesian(z, z.length, Double.MIN_NORMAL);
//            double zSum = Util.sumAbs(z); Util.mul(1/zSum, z); //Normalize Manhattan
//
//            double diff = DistanceFunction.distanceManhattan(x, actionDecode(z, actions));
//            if (diff!=diff)
//                throw new WTF();
//            System.out.println("diff: " + diff);
//
//            //System.out.println(n4(x) + "\t->\t" + n4(z) + "\t=\t" + n4(actionDecode(z, actions)));
//
//            return z;
//        }
//
//        public static double dotProduct(double[] x, double[] y) {
//            assert(x.length == y.length);
//            double s = 0;
//            for (int i = 0; i < x.length; i++)
//                s += (x[i]*y[i]);
//            return s;
//        }
//
//        public static double len(double[] x) {
//            double s = 0;
//            for (int i = 0; i < x.length; i++)
//                s += sqr(x[i]);
//            return Math.sqrt(s);
//        }
//
//        private double vectorProject(double[] a, double[] b) {
//            return dotProduct(a, b) / len(b);
//        }


    }

    public static class LinearCombinationActionDecoder implements ActionDecoder {

        @Override public double[] actionDecode(double[] z, int actions) {

            z = Util.normalizeUnit(z.clone());

            double[] y = new double[actions];
            for (int i = 0; i < z.length; i++) {
                double[] ideal = idealDecode(i, actions);
                double zi = Util.unitizeSafe(z[i]);
                for (int a = 0; a <actions; a++)
                    y[a] += zi * ideal[a];
            }

            return y;
        }

    }

    public static class NoisyLinearCombinationActionDecoder extends LinearCombinationActionDecoder {

        final double noiseAmp =
            0.5f;
            //1;

        @Override public double[] actionDecode(double[] z, int actions) {
            double noise = noise(z);
            //System.out.println(noise);

            double[] y = super.actionDecode(z, actions);

            //            boolean noise = true;
            var rng = ThreadLocalRandom.current();
            //System.out.println(s);
            for (int a = 0; a < actions; a++) {
                y[a] = Util.unitizeSafe(y[a] +
                    noise * rng.nextGaussian()
                    //noise*rng.nextDouble(-1, +1)
                );
            }
            return y;
        }

        private double noise(double[] z) {
            return
                Util.sqr(
                    Util.unitize(Math.abs(1 - Util.sumAbs(z)))
                )
                * noiseAmp;
        }
    }

    private static double[] idealDecode(int Z, int actions) {
        double[] z = new double[actions];
        for (int i = 0; i < actions; i++) {
            if ((Z & (1 << i)) != 0)
                z[i] = 1;
        }
        return z;
    }

//    public static class FuzzyDistanceActionEncoder extends DistanceActionEncoder {
//    }
    public static class FuzzyDistanceActionDecoder extends LinearCombinationActionDecoder {
        @Override
        public double[] actionDecode(double[] z, int actions) {

            //TODO refine

            int zArgMax = Util.argmax(z);
            double zMax = z[zArgMax];
            double zOtherSum = 0;
            for (int i = 0; i < z.length; i++) {
                if (i!=zArgMax)
                    zOtherSum += z[i] / zMax;
            }
            double uncertainty = zOtherSum / (z.length - 1);
            //double uncertainty = 1 - (Util.max(z) - Util.min(z));
            //System.out.println(uncertainty + " " + n2(z));
            Random rng = new XoRoShiRo128PlusRandom();
//            for (int i = 0; i < z.length; i++)
//                z[i] = Util.unitizeSafe( z[i] + uncertainty * ((rng.nextFloat()-0.5f)*2f) /* TODO gaussian */ );

            double[] y = super.actionDecode(z, actions);

            addNoise(y, uncertainty, rng);

            return y;
        }

        private static void addNoise(double[] y, double uncertainty, Random rng) {
            for (int i = 0; i < y.length; i++)
                y[i] = Util.unitizeSafe( y[i] + uncertainty *
                        rng.nextGaussian()
                        // (rng.nextFloat()-0.5f)*2
                );
        }
    }

    public static class SoftDistanceActionDecoder implements ActionDecoder {
        private final Random rng = new XoRoShiRo128PlusRandom();

        private final Decide decide =
            //new DecideRoulette(rng);
            new DecideSoftmax(0.25f, new XoRoShiRo128PlusRandom());

        /** samples precision */
        int iterations;

        public SoftDistanceActionDecoder(int iterations) {
            this.iterations = iterations;
        }
        //4;
            //1;
            //0.5f;

        @Override
        public double[] actionDecode(double[] z, int actionsExternal) {

//            //remove floor
//            if (z.length>2) {
//                z = z.clone();
//                double zMin = Util.min(z);
//                for (int i = 0; i < z.length; i++)
//                    z[i] -= zMin;
//            }

            //multisampled softmax:
            int actionsInternal = z.length;
            int iterations =
                this.iterations;
                //(int)Math.ceil(this.iterations * actionsExternal);
            double[] y = new double[actionsExternal];
            for (int i = 0; i < iterations; i++) {
                double[] yi = actionDecodeDecide(z, actionsExternal, decide);
                for (int j = 0; j < actionsExternal; j++)
                    y[j] += yi[j];
            }
            Util.mul(y, 1f/iterations);

            //System.out.println(n2(z) + " -> " + n2(y));

            return y;
        }
    }

}