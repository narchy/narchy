package jcog.tensor.rl.misc;

import jcog.Log;
import jcog.Util;
import jcog.activation.ReluActivation;
import jcog.activation.SigmoidActivation;
import jcog.nn.RecurrentNetwork;
import jcog.optimize.MyAsyncCMAESOptimizer;
import jcog.optimize.MyCMAESOptimizer;
import jcog.signal.FloatRange;
import jcog.tensor.rl.dqn.Policy;
import org.hipparchus.optim.nonlinear.scalar.GoalType;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/** non-gradient policy optimization */
public class PopulationPolicy implements Policy {

    private static final Logger logger = Log.log(PopulationPolicy.class);

    public final Population pop;
    private RecurrentNetwork fn;

    /** explore vs. exploit rate.  0..1 */
    public final FloatRange exploit = FloatRange.unit(
        //0.5f
        0
    );

    /** how many iterations to try each individual for
     *  TODO tune this in proportion to aggregate reward variance, starting with a small value
     * */
    int period =
        //4;
        8;
        //16;
        //64;
        //128;

    /** # of episode epochs to accumulate before learning & reset */
    int repeats = 4;

    /** determines # hidden neurons */
    float complexity = 1/5f;

    /** population size */
    final int populationSize =
        8;
        //16;
        //64;

    final float weightRange =
        Util.PHIf;
        //Float.NaN; //unbounded
        //Util.PHIf * 4;
        //Util.PHIf * 2;
        //Util.PHIf / 2;

    /** reward accumulator per individual */
    private transient double[] individualRewards = null;

    private transient int iteration = 0, individualCurrent = 0;
    private transient double[] piBest = null;
    private transient boolean exploring = false;
    private transient int timeUntilExplore = 0;
    private transient int repeatsRemain;


    /** current policy being tesed */
    private double[] policy;

    public PopulationPolicy(Population p) {
        pop = p;
    }

    public double[] current() {
        return policy;
    }

    @Override
    public void clear(Random rng) {
        iteration = 0;
        individualCurrent = 0;
        individualRewards = new double[populationSize];
        repeatsRemain = repeats;
    }

    @Override
    public double[] learn(@Nullable double[] xPrev_ignored, double[] actionPrev, double reward, double[] x, float pri) {

        int actions = actionPrev.length;
        if (!pop.initialized) {
            int parameters = fn(x.length, actions);
            pop.init(parameters, populationSize, -weightRange, +weightRange);
            pop.initialized = true;
        }

        if (exploring || piBest == null || timeUntilExplore-- <= 0) {
            explore(reward);
        } else {
            policy = piBest;
        }

        return act(x, policy);
    }

    private void explore(double reward) {
        exploring = true;

        //accumulate reward for the current individual
        individualRewards[individualCurrent] += reward /* * pri ? */; //TODO -reward if minimize?

        if (((iteration++) + 1) % period == 0) {
            //accumulate rewards for the population individual
            int individualNext = (individualCurrent + 1) % populationSize;

            /* after evaluating the entire population (upon returning to the individual #0),
               commit accumulated rewards and iterate */
            if (individualNext == 0 && --repeatsRemain == 0) {
                learn();
                reset();
            }
            individualCurrent = individualNext;
        }

        policy = pop.get(individualCurrent);
    }

    private void reset() {
        exploring = false;
        timeUntilExplore = (int) Math.ceil(exploit.getAsDouble() * (period * repeats * populationSize));
        repeatsRemain = repeats;
        Arrays.fill(individualRewards, 0); //reset
    }

    private void learn() {
        Util.mul(individualRewards, 1f/(period * repeats)); //normalize to reward per iteration

        //logger.info("{} {{}..{}}", n4(Util.mean(individualRewards)), n4(Util.min(individualRewards)), n4(Util.max(individualRewards)));

        pop.commit(individualRewards); //finished batch
        piBest = pop.best();
    }


    /** returns # of parameters */
    private int fn(int inputs, int actions) {
//        if (direct)
//            return actions; //DIRECT

        if (fn == null) {
            boolean recurrent = false; /* 0 to disable */
            boolean inputsDirectToOutput = false;
            int hidden = inputsDirectToOutput ? 0 :
                Math.round((inputs + actions) * complexity)
                //Math.round(actions * complexity)
                //actions + 1;
                //actions;
                //Fuzzy.mean(inputs/2, actions);
                //inputs + actions;
            ;
            int hops = (hidden==0 ? 1 : recurrent ? 4 : 2);

            if (hidden == 0) {
                if (recurrent || !inputsDirectToOutput)
                    throw new UnsupportedOperationException();
            }

            this.fn = new RecurrentNetwork(inputs, actions, hidden, hops);

            fn.activationFn(
                ReluActivation.the,
                SigmoidActivation.the
                //LeakyReluActivation.the,
                //SigLinearActivation.the
                //TanhActivation.the;
                //ReluActivation.the;
            );

            //HACK to calculate weightsEnabled subset, establish full connectivity
            if (!recurrent) {
                fn.connect[1][0] = fn.connect[1][1] = 0;
                fn.connect[2][0] = fn.connect[2][1] = fn.connect[2][2] = 0;
            }

            fn.connect[0][2] = inputsDirectToOutput? 1 : 0;

            //connection prob=1 unles zero:
            for (int i = 0; i < fn.connect.length; i++)
                for (int j = 0; j < fn.connect[0].length; j++) {
                    if (fn.connect[i][j] > 0)
                        fn.connect[i][j] = 1;
                }

            fn.clear(ThreadLocalRandom.current());
            logger.info("{}\tins={}\touts={}\thidden_neurons={} hidden_layers={}\trecurrent={}", fn.getClass(), inputs, actions, hidden, (hops-1), recurrent);
        }
        return fn.weights.weightsEnabled();
    }


    /**
     * determine an action vector for the state applied to a given policy
     */
    private double[] act(double[] x /* state */, double[] policy) {
        return fn.weights(policy).get(x); //function of state
    }

    public abstract static class Population {

        public boolean initialized = false;

        public abstract void init(int parameters, int populationSize, float boundMin, float boundMax);

        public abstract double[] best();

//        /** weighted sum of populations by rewards */
//        public double[] bestComposite() {
//            throw new TODO();
//        }

        public abstract void commit(double[] individualRewards);

        public abstract double[] get(int individual);
    }

//    /** TODO */
//    abstract public static class NEATPopulation extends Population {
//
//    }

//    /** TODO
//     *  https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb
//     */
//    abstract public static class CrossEntropyMethodPopulation extends Population {
//
//    }

    public static class CMAESPopulation extends Population {
        private MyAsyncCMAESOptimizer opt = null;
        private MyCMAESOptimizer.FitEval iter;

        /**
         * population, representing policies
         */
        private double[][] pi = null;

        /**
         * Standard deviation for all parameters (all parameters must be scaled accordingly).
         * Defines the search space as the std dev from an initial x0.
         * Larger values will sample a initially wider Gaussian.
         * TODO dynamic control/re-control
         * TODO tune
         */
        private final FloatRange sigma = new FloatRange(0.25f, 0.0001f, 4);
        private final FloatRange sigmaMin = new FloatRange(sigma.floatValue()/32, 0.0001f, 4);

        private static final Logger logger = Log.log(CMAESPopulation.class);
        private int parameters;


        /** current policy */
        @Deprecated private double[] policy;

        @Override
        public void init(int parameters, int populationSize, float boundMin, float boundMax) {
            this.parameters = parameters;
            logger.info("CMAES+NEAT: {} parameters", parameters);

            double[] sigma = new double[parameters];
            Arrays.fill(sigma, this.sigma.floatValue());

            opt = new MyAsyncCMAESOptimizer(populationSize, sigma) {
                @Override
                protected boolean apply(double[][] p) {
                    pi = p;
                    return true;
                }
            };
            opt.sigmaMin = sigmaMin.floatValue();

            if (boundMin==boundMin) {
                double[] min = new double[parameters]; Arrays.fill(min, boundMin);
                double[] max = new double[parameters]; Arrays.fill(max, boundMax);
                iter = opt.iterator(GoalType.MAXIMIZE, min, max);
            } else
                iter = opt.iterator(GoalType.MAXIMIZE, parameters);
        }

        @Override
        public double[] get(int individual) {
            return policy = pi[individual];
        }

        @Override
        public double[] best() {
            return opt.best().clone();
        }

        @Override
        public void commit(double[] individualRewards) {
            iter.next(individualRewards);
        }

        public int parameters() {
            return parameters;
        }

        public double[] policy() {
            return policy;
        }

        /** for setting an individual, ex: to finetune */
        public void set(double[] i, int individual) {
            System.arraycopy(i, 0, pi[individual], 0, i.length);
            opt.arx.setColumn(individual, i);
        }
    }
}
