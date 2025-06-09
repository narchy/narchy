package jcog.tensor.deprtensor;

import jcog.data.bit.MetalBitSet;
import jcog.data.list.Lst;
import jcog.math.FloatSupplier;
import jcog.math.NumberException;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;

import java.util.List;
import java.util.function.Function;

import static jcog.Util.fma;
import static jcog.tensor.deprtensor.Tens0r.scaleNorm;

public abstract class Optimize {
    public boolean gradientNorm = true;
    public boolean gradientNormPreClip = false;

    public static boolean clipGradBeforeBackward = true;

    public static double gradientClip =
        //0.25;
        //0.1;
        //16;
        //64;
        1;
        //0.5f;
        //Double.POSITIVE_INFINITY; //disabled


    /**
     * determines the 'dynamic range'
     */
    public double weightClip =
        //1;
        //2;
        //4;
        //16;
        //64;
        //256;
        //1024;
        //2048;
        128 * 1024;
        //Double.POSITIVE_INFINITY; //disabled

    public FloatSupplier learningRate;

    protected Optimize(FloatSupplier learningRate) {
        this.learningRate = learningRate;
    }

    public static void clip(double[] x) {
        Tens0r.clip(x, gradientClip);
    }
    public static void clip(double[][] x) {
        Tens0r.clip(x, gradientClip);
    }

    public abstract void step(List<Tens0r> parameters);

    public final void run(TensorFn.Layers l) {
        run(l.params);
    }

    public final void run(List<Tens0r> params) {
        try {
            if (gradientClip != Double.POSITIVE_INFINITY) {
                if (gradientNorm) {
                    if (gradientNormPreClip)
                        clipGradients(params);

                    scaleNorm(params, gradientClip);
                } else {
                    clipGradients(params);
                }
            }


        } catch (NumberException e) {
            System.err.println("gradient pre: " + e.getMessage());
            for (var p : params)
                p.fillData(0); //TODO only the failing param, not all
        }

        step(params);

        if (weightClip != Double.POSITIVE_INFINITY) {

//            double dataMaxAbs = 0;
            for (var p : params) {
//                dataMaxAbs = Math.max(dataMaxAbs, p.dataMaxAbs());
                p.clipData(weightClip);
            }
//            System.out.println(dataMaxAbs);
        }

        for (var p : params)
            p.zeroGrad();
    }

    private static void clipGradients(List<Tens0r> params) {
        for (var p : params)
            p.clipGradients(gradientClip);
    }


    public static class DenseLayer extends TensorFn.AbstractLayer {
        public final Tens0r weights;
        public final Tens0r bias;
        final Function<Tens0r, Tens0r> activationFn;
        final RandomBits rand = new RandomBits(new XoRoShiRo128PlusRandom());
        private final TensorFn.Init init;
        protected TensorFn output;  // Last output, stored for backpropagation
        protected Tens0r input;     // Last input, stored for backpropagation
        MetalBitSet active;
        private TensorFn tensor;
        private float dropoutRate; // Default dropout rate is 0 (no dropout)

        /** clip grad after layer backward */


        public DenseLayer(int i, int o, TensorFn.Activate act) {
            this(i, o, act.fn, act.init);
        }

        public DenseLayer(int i, int o, Function<Tens0r, Tens0r> activationFn, TensorFn.Init init) {
            this.activationFn = activationFn;
            this.init = init;
            this.weights = new Tens0r(new double[i][o]);
            this.bias = new Tens0r(new double[o]);
            reset();
        }

        public void setDropoutRate(float dropoutRate) {
            this.dropoutRate = dropoutRate;
        }

        @Override
        public void addParams(Lst<Tens0r> target) {
            target.add(weights);
            target.add(bias);
        }

        @Override
        public Tens0r forward(Tens0r input) {
            this.input = input;

            if (output == null)
                this.output = build();
            this.tensor.x = new Tens0r[]{input};

            _forward();

            this.output.forward();

            return this.output;
        }

        @Override
        public void zeroGrad() {
            if (tensor != null)
                tensor.zeroGrad();
            output.zeroGrad();
        }

        private TensorFn build() {
            var y = new TensorFn.TensorOpFn(new double[input.data.length][weights.data[0].length], TensorOp.DENSE, input) {

                @Override
                public void backward() {
                    if (x == null)
                        x = new Tens0r[]{input};  //return;

                    DenseLayer.this.backward(input);

                    input.backward();
                }

            };
            this.tensor = y;
            return (TensorFn) activationFn.apply(y);
        }

        @Override public void backward(Tens0r X /* parent */) {
            double[][] w = weights.data;
            int I = w.length, O = w[0].length;

            double[][] x =  X.data;
            double[][] gx = X.grad;

            double[]   gb = bias.grad[0];
            double[][] gw = weights.grad;

            double[][] gp = tensor.grad; //incoming grad from post/output (read-only)
            @Deprecated int G = gp.length;

            for (int g = 0; g < G; g++) {
                double[] gg = gp[g];
                for (int o = 0; o < O; o++) {
                    if (!active.test(o))
                        continue; // Skip if neuron was dropped
                    double ggo = gg[o];
                    gb[o] += ggo;
                    for (int i = 0; i < I; i++) {
                        gw[i][o] = fma(ggo, x[g][i], gw[i][o]);
                        gx[g][i] = fma(ggo, w[i][o], gx[g][i]);
                    }
                }
            }
        }

        private static boolean hasNaN(double[] gg) {
            for (double x : gg)
                if (x!=x) return true;
            return false;
        }

        private void _forward() {
            var weights = this.weights.data;
            var bias = this.bias.data[0];
            double[][] input = this.input.data;
            int I = input.length;
            int J = weights[0].length;
            int K = weights.length;

            double[][] y = tensor.data;

            if (active == null)
                active = MetalBitSet.bits(J);

            int inactives = 0;
            for (int j = 0; j < J; j++) {
                boolean inactive = rand.nextBooleanFast8(dropoutRate);
                active.set(j, !inactive);
                if (inactive) inactives++;
            }
            double dropoutRescale = 1.0 / (1.0 - dropoutRate/*Effective*/);

            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++) {
                    double yij;
                    if (active.test(j)) {
                        yij = bias[j];

                        for (int k = 0; k < K; k++)
                            yij = fma(input[i][k], weights[k][j], yij); //yij += input[i][k] * weights[k][j];

                        yij *= dropoutRescale;  // Scale the active neurons' outputs
                    } else {
                        yij = 0; // Apply dropout
                    }
                    y[i][j] = yij;
                }
            }
        }

        public void reset() {
            init.apply(weights);
        }

        @Override
        public DenseLayer clone() {
            return new DenseLayer(weights.h(), weights.w(), activationFn, init);
        }
    }
}
