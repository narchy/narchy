package jcog.nndepr.optimizer;

import jcog.Util;
import jcog.nndepr.layer.LinearLayer;
import jcog.signal.FloatRange;

/**
 * https://github.com/google/automl/blob/master/lion/lion_pytorch.py
 * https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
 */
public class LionOptimizer extends BatchWeightUpdater {

    float subStepping =
        //100;
        200;
        //1000;

    public final FloatRange
        beta1 = FloatRange.unit(0.9f),
        beta2 = FloatRange.unit(0.99f),
        weight_decay = FloatRange.unit(0);

    /** Exponential moving average of gradient values */
    double[] exp_avg = null;
    protected float alpha;


    /*
          def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
            Args:
              params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
              lr (float, optional): learning rate (default: 1e-4)
              betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.99))
              weight_decay (float, optional): weight decay coefficient (default: 0)
         */
    @Override
    protected void updateWeights(LinearLayer l, double[] dW, double[] dWPrev, double[] w) {
        int n = w.length;
        /*
        stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])
        */
        float weight_decay = this.weight_decay.asFloat();
        double f = 1 - alpha * weight_decay;

        double beta1 = this.beta1.doubleValue(), beta2 = this.beta2.doubleValue();
        double oneMinBeta1 = 1 - beta1;
        double[] exp_avg = this.exp_avg;
        double alpha = this.alpha;

        for (int i = 0; i < n; i++) {
            double dwi = dW[i];
            double expAvgI = exp_avg[i];

            /*
            weight update:
              update = exp_avg * beta1 + grad * (1 - beta1)
              p.add_(torch.sign(update), alpha=-group['lr'])
            */
            double update = expAvgI * beta1 + dwi * oneMinBeta1;
            //w[i] = (w[i] * f) + Math.signum(update) * alpha;
            w[i] = Util.fma(Math.signum(update), alpha, (w[i] * f));

            /*
              Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
            */
            exp_avg[i] = Util.lerpSafe(beta2, dwi, expAvgI);
        }
    }

    @Override
    public void reset(int weights, float alpha) {
        super.reset(weights, alpha);
        if (exp_avg == null)
            exp_avg = new double[weights];
        this.alpha = alpha/subStepping;
    }
}
