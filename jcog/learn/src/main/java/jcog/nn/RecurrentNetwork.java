package jcog.nn;

import jcog.Util;
import jcog.activation.DiffableFunction;
import jcog.activation.LinearActivation;
import jcog.activation.ReluActivation;
import jcog.activation.SigLinearActivation;
import jcog.data.bit.MetalBitSet;
import jcog.tensor.DeltaPredictor;
import jcog.tensor.Predictor;
import org.hipparchus.linear.Array2DRowRealMatrix;
import org.hipparchus.linear.RealMatrix;

import java.util.Arrays;
import java.util.Random;

import static java.lang.System.arraycopy;
import static jcog.nn.BackpropRecurrentNetwork.wClamp;

/**
 * TODO rename FreeformNetwork
 * TODO mutable hidden neuron 'centroid's and connectivity matrix
 * TODO basic double[] state impl
 * TODO sparse edge list state impl
 *
 * TODO  ?? forward value state buffering for use during specific backprop iter
 *
 */
public class RecurrentNetwork extends DeltaPredictor {

    /**
     * -weightRange..+weightRange
     * TODO share weight initialization with MLP
     */
    @Deprecated
    final float initWeightRange;

    public final int inputs, outputs, hiddens;

    /**
     * neuron activation TODO make this recycled or threadlocal
     */
    final double[] A;
    /**
     * neuron activation (double buffer)
     */
    final double[] Anext;



    /**
     * analogous to expected # of MLP hidden layers
     */
    private static final int iterationsForward_default = 3;
    final int iterationsForward;

    /** TODO use SGDLayer or abstract sibling */
    @Deprecated public final WeightState weights;
    private final MetalBitSet weightsEnabled;

    public DiffableFunction activationFnHidden =
        ReluActivation.the;
        //LeakyReluActivation.the;
        //SigmoidActivation.the;
        //new SigLinearActivation();
        //TanhActivation.the;

    public DiffableFunction activationFnOutput =
        SigLinearActivation.the;
        //SigmoidActivation.the;
        //ReluActivation.the;

    /*
        { input + bias, hidden, output } x { input + bias, hidden, output }
     */

    final boolean inputsConstant = true;
    final boolean outputsTerminal = false;
    final boolean selfConnections = false;

    public final float[][] connect = new float[3][];
    {
        //inputs  -> ...
        connect[0] = new float[]{0, 3/4f, 0};

        //hiddens -> ...
        connect[1] = new float[]{0 /* inputsConstant? */, 1/4f, 3/4f};

        //outputs -> ...
        connect[2] = new float[]{
                0 /* inputsConstant? */,
                /* TODO valid for iterations >= 4: */ outputsTerminal ? 0 : 1/4f,
                /* TODO this is valid only for iterations >= 3 */ outputsTerminal ? 0 : 1/4f};
    }

    public RecurrentNetwork(int inputs, int outputs, int hiddens, int iterations) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.hiddens = hiddens;
        this.iterationsForward = iterations;

        /* Neuron ordinal map:
                       (bias)               */
        int n = inputs + 1 + hiddens + outputs;

        initWeightRange =
            //1f/n;
            (float) (1/Math.sqrt(n));

        this.A = new double[n /* bias */];
        this.Anext = new double[n /* bias */];

        this.weights = new WeightState(n);
        this.weightsEnabled = MetalBitSet.bits(weights.weights());;
    }

    @Override
    public void copyLerp(Predictor p, float rate) {
        weights.copyLerp(((RecurrentNetwork)p).weights, rate);
    }

    public final boolean isBias(int f) {
        return f == inputs;
    }

    public RecurrentNetwork clear(float weightRange, Random rng) {
        int n = n();

        for (int from = 0; from < n; from++) {
            for (int to = 0; to < n; to++) {
                boolean enabled, active = false;
                float cft = 0;
                if (inputsConstant && to < inputs)
                    enabled = false;
                else if (isBias(to))
                    enabled = false;
                else if (outputsTerminal && from >= n - outputs)
                    enabled = false;
                else if (!selfConnections && from == to)
                    enabled = false;
                else {

                    int F = neuronClass(from), T = neuronClass(to);
                    cft = this.connect[F][T];


//                    if (F==1 && T==1) {
//                        //recurrent: scale by id distance
//                        //more likely to recur within neighborhood
//                        float proximity = 1 - (( 1 + Math.abs(from - to))/((float)hiddens));
//                        //cft *= 2 /* on avg */ * proximity;
//                        cft *= Math.sqrt(2) /* on avg TODO check */ * Util.sqr(proximity);
//                    }

                    active = cft >= 1 || rng.nextFloat() < cft;
                    enabled =
                        active; //sparse
                        //true; //full
                        //cft > 0; //dense
                }

                double w;
                if (active)
                    w =
                        //weightRange * Fuzzy.polarize(rng.nextFloat());
                        //weightRange * rng.nextGaussian();

                        //guassian-like:
                        weightRange * (rng.nextBoolean() ? +1 : -1)
                            * Math.pow(rng.nextDouble(), 1 + Math.sqrt(n * cft)) //TODO use randomizeHe gaussian
                        ;
                else
                    w = 0;

                weights.weightSet(from, to, w);
                weights.weightsEnabled.set(from * n + to, enabled);
            }
        }

        return this;
    }

    public final int neuronClass(int n) {
        int inputLimit = inputs + 1;
        if (n < inputLimit) return 0;
        else if (n < inputLimit + hiddens) return 1;
        else return 2;
    }

    @Override
    public double[] get(double[] x) {
        /* initialize non-input, non-bias neurons */

        double[] value = this.A, valueNext = this.Anext;

        int n = n();

        //set inputs
        arraycopy(x, 0, value, 0, inputs);

        //set bias
        value[/* bias index: */ inputs] = 1; //constant bias input

        //set hidden and outputs to 0
        Arrays.fill(value, inputs + 1, n, 0);

        arraycopy(value, 0, valueNext, 0, value.length);

        for (int i = 0; i < iterationsForward; i++) {
            if (!iteration(i))
                break;
        }


        //return outputs
        return Arrays.copyOfRange(value, n - outputs, n);
    }

    /**
     * returns true if not stable
     *  TODO optional simultaneous pre/post cycle separation for increased precision
     */
    protected boolean iteration(int iteration) {
        boolean changed = iteration < iterationsForward-1; //if last iteration, dont bother checking

        double[] v = this.A, vNext = this.Anext;
        int vv = v.length;
        RealMatrix weights = this.weights.getWeights();

        int iStart = inputsConstant ? inputs + 1 : 0;
        for (int to = iStart; to < vv; to++) {

            double activationSum = 0;
            for (int from = 0; from < vv; from++) {
                double x = v[from];
                if (x!=0) {
                    double w = this.weights.weight(from, to);
                    if (w != 0)
                        activationSum = Util.fma(x, w, activationSum);
                }
            }

            double nextActivation = activationFn(to).valueOf(activationSum);

            if (!changed)
                changed = !Util.equals(v[to], nextActivation, stableThreshold);

//            if (to < outputStart)
             vNext[to] = nextActivation;
//            else {
//                //accumulate activation in outputs
//                vNext[to] += nextActivation;
//            }

        }

        if (changed) {
            //apply double-buffer
            arraycopy(vNext, iStart, v, iStart, inputsConstant ? hiddens + outputs : this.n());
        }

        return changed;
    }

    /**
     * activation function
     * TODO abstract for per-neuron variety
     *
     * @return
     */
    protected DiffableFunction activationFn(int neuron) {
        if (neuron >= inputs + 1 + hiddens)
            return activationFnOutput;
        else if (neuron >= inputs + 1)
            return activationFnHidden;
        else
            return LinearActivation.the;
    }

    public final RecurrentNetwork activationFn(DiffableFunction hidden, DiffableFunction out) {
        this.activationFnHidden = hidden;
        this.activationFnOutput = out;
        return this;
    }

    int outputNeuron(int o) {
        return inputs + 1/*bias*/ + hiddens + o;
    }

    @Override
    public void clear(Random rng) {
        clear(initWeightRange, rng);
    }

    @Override
    @Deprecated public void putDelta(double[] d, float pri) {
        throw new UnsupportedOperationException("TODO");
    }

    public int n() {
        return A.length;
    }

    public double value(int neuron /*, TODO bptt: int when */) {
        return A[neuron];
    }

    /** loads a weight vector */
    public RecurrentNetwork weights(double[] weights) {
        this.weights.load(weights);
        return this;
    }

    /** default dense matrix impl */
    public static class WeightState {

        //private static final double weightThresh = 1.0/(weightClamp*weightClamp);

        /**
         * synapse weights
         *  TODO optional sparse matrix impls
         */
        private final Array2DRowRealMatrix weights;
        private final double[][] weightsData;

        final MetalBitSet weightsEnabled;

        public WeightState(int n) {
            this.weights =
                    new Array2DRowRealMatrix(n, n);
                    //new BlockRealMatrix(n, n);
            weightsData = weights.getDataRef();

            weightsEnabled = MetalBitSet.bits(weights());
        }

        @Deprecated public RealMatrix getWeights() {
            return weights;
        }


        public double weight(int i) {
            int n = weightsData[0].length;
            return weight(i%n, i/n);
        }

        public final double weight(int f, int t) {
            return weightsData[f][t];
        }

        public void weightAdd(int f, int t, double dw) {
//            weights.setEntry(f, t,
//                    weights.getEntry(f,t) + dw);
            weightSet(f, t, weight(f, t) + dw);
        }

        public double dotColumn(int f, double[] x) {
            int n = weightsData.length;
            double s = 0;
            for (int t = 0; t < n; t++) {
                //s += state.weight(f, t) * delta[t];
                s = Util.fma(weight(f, t), x[t], s);
            }
            return s;
        }

        public void weightSet(int f, int t, double next) {

            double v;
//            if (next == 0)
//                v = 0;
//            else if (next > 0 && next < weightThresh)
//                v = +weightThresh;
//            else if (next < 0 && next > -weightThresh)
//                v = -weightThresh;
//            else
                v = Util.clampSafe(next, -wClamp, +wClamp);

            weightsData[f][t] = v;
        }

        /** total weight count */
        public int weights() {
            return Util.sqr(weightsData.length);
        }

        public int weightsEnabled() {
            return weightsEnabled.cardinality();
        }

        public double weightL1() {
            return Util.sumAbs(weightsData);
            //return weights.walkInOptimizedOrder(new L1Norm());
        }

        public void load(double[] weights) {
            int n = weightsData.length, k = 0, l = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double w = weightsEnabled.test(k) ? weights[l++] : 0;
                    weightsData[i][j] = w;
                    k++;
                }
        }

        /** "soft update" */
        public void copyLerp(WeightState weights, float rate) {
            double[][] y = weights.weightsData;
            double[][] x = this.weightsData;
            for (int i = 0; i < x.length; i++)
                for (int j = 0; j < x[0].length; j++)
                    x[i][j] = Util.lerpSafe(rate, x[i][j], y[i][j]);
        }

//        public double weightL2() {
//            return weights.walkInOptimizedOrder(new L2Norm());
//        }

//        private static class L1Norm implements RealMatrixPreservingVisitor {
//
//            private double sum;
//
//            @Override
//            public void start(final int rows, final int columns,
//                              final int startRow, final int endRow,
//                              final int startColumn, final int endColumn) {
//                sum = 0;
//            }
//
//            @Override
//            public void visit(final int row, final int column, final double value) {
//                sum += abs(value);
//            }
//
//            @Override
//            public double end() {
//                return sum;
//            }
//        }

//        private static class L2Norm implements RealMatrixPreservingVisitor {
//
//            private double sumSq;
//
//            @Override
//            public void start(final int rows, final int columns,
//                              final int startRow, final int endRow,
//                              final int startColumn, final int endColumn) {
//                sumSq = 0;
//            }
//
//            @Override
//            public void visit(final int row, final int column, final double value) {
//                sumSq = Util.fma(value,value, sumSq);
//            }
//
//            @Override
//            public double end() {
//                return Math.sqrt(sumSq);
//            }
//        }
    }

    protected static final double stableThreshold = NEAT.stableThreshold;

}