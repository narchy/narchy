package jcog.nndepr.spiking0.neuron;

import jcog.data.list.Lst;
import jcog.nndepr.spiking0.AbstractBrain;
import jcog.nndepr.spiking0.AbstractNeuron;
import jcog.nndepr.spiking0.AbstractSynapse;
import jcog.nndepr.spiking0.learn.SpikeSynapseLearning;

import java.util.List;
import java.util.stream.Stream;

public class RealtimeBrain extends AbstractBrain {

    public final List<AbstractNeuron> neurons = new Lst<>();

    /** bias, ambient, temperature */
    double thalamicInput =
        //0;
        //0.1;
        0.01;

    public SpikeSynapseLearning synapseUpdate = null;

    //0.02;
        //0.001;


    @Override
    public void forward() {
        for (var n : neurons) {
            if (n instanceof SpikingNeuron sn)
                sn.update(thalamicInput, this);
        }

        for (var n : neurons) {
            if (n instanceof SpikingNeuron sn)
                sn.commit();
        }

        if (synapseUpdate!=null)
            synapseStream().forEach(synapseUpdate::update);

        t += timeStep;
    }

    public Stream<AbstractSynapse> synapseStream() {
        return neurons.stream().flatMap(z -> z instanceof SpikingNeuron zz ? zz.synapses.stream() : Stream.empty());
    }
}
