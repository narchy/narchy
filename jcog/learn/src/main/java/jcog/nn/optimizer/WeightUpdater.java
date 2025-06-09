package jcog.nn.optimizer;

import jcog.nn.layer.LinearLayer;

public interface WeightUpdater {
    void reset(int weights, float alpha);

    /** @param deltaIn delta coming from next layer, backwards */
    void update(LinearLayer l, double[] deltaIn);
}