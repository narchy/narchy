package jcog.nn.spiking0.learn;

import jcog.Util;
import jcog.nn.spiking0.AbstractNeuron;
import jcog.nn.spiking0.AbstractSynapse;
import jcog.nn.spiking0.neuron.SpikingNeuron;

public abstract class SpikeSynapseLearning implements SynapseLearning {
    final double weightMin = 0.01f;
    final double weightMax =
        1;
        //2;

    final double decayRate = 0.98f;

    @Override
    public void update(AbstractSynapse s) {
        AbstractNeuron f = s.source, t = s.target;
        double nextWeight;
        if (f instanceof SpikingNeuron F && t instanceof SpikingNeuron T) {
            double delta = update(F, T);
            nextWeight = (decayRate * s.weight) + delta;
        } else
            nextWeight = s.weight;

        s.weight =
            nextWeight >= 0 ?
                    Util.clampSafe(nextWeight, +weightMin, +weightMax) :
                    Util.clampSafe(nextWeight, -weightMax, -weightMin);
            //Util.clampSafe(nextWeight, -weightMax, weightMax);
    }

    protected abstract double update(SpikingNeuron a, SpikingNeuron b);
}
