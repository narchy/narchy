package jcog.nn.optimizer;

import jcog.Util;
import jcog.activation.DiffableFunction;
import jcog.data.bit.MetalBitSet;
import jcog.nn.layer.LinearLayer;

import java.util.Arrays;

import static jcog.Util.fma;

public abstract class BatchWeightUpdater implements WeightUpdater {
    int iteration = -1;
    protected int minibatches = 1;

    @Override
    public void reset(int weights, float alpha) {
        iteration++;
    }

    @Override public final void update(LinearLayer l, double[] dOut) {

        updateGrad(l, dOut);

        if (iteration % minibatches == 0) {
            commitGrad(l, l.dW);
            Arrays.fill(l.dW, 0);
        }
    }

    public void commitGrad(LinearLayer l, double[] dW) {
        updateWeights(l, dW, l.dWPrev, l.W);
    }

    private void updateGrad(LinearLayer l, double[] dOut) {
        int ii = l.ins(), oo = l.outs();

        double[] dW = l.dW;
        double[] W = l.W;
        double[] out = l.out;
        double[] dIn = l.delta;

        double[] in = l.in;
        DiffableFunction act = l.activation;

        boolean dropping = l.dropping;
        MetalBitSet e = dropping ? l.enabled : null;


        //update gradients
        for (int o = 0, io = 0; o < oo; o++) {

            double dOutO = dOut[o] * (act != null ? act/*TODO act[o]*/.derivative(out[o]) : 1);
            if (!Double.isFinite(dOutO))
                continue; //skip

            for (int i = 0; i < ii; i++, io++) {

                double inI;
                if (dOutO == 0 || (dropping && !e.test(io)) || !Double.isFinite(inI=in[i]))
                    continue; //skip

                dIn[i] = fma(W[io], dOutO, dIn[i]);

                dW[io] = grad(fma(inI, dOutO, dW[io]));
            }
        }
    }

    private transient boolean clipping = false;
    private double gradMin, gradMax;

    protected double grad(double x) {
        double y = clipping ? Util.clampSafe(x, gradMin, gradMax) : x;
//        if (x!=y)
//            Util.nop();
        return y;
    }

    public final BatchWeightUpdater gradClamp(double abs) {
        clipping = true; gradMin = -abs; gradMax = +abs;
        return this;
    }

    protected abstract void updateWeights(LinearLayer l, double[] dW, double[] dWPrev, double[] w);


    public WeightUpdater minibatches(int i) {
        minibatches = i;
        return this;
    }
}