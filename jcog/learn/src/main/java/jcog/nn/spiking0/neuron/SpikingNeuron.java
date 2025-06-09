package jcog.nn.spiking0.neuron;

import jcog.data.list.Lst;
import jcog.nn.spiking0.AbstractNeuron;
import jcog.nn.spiking0.AbstractSynapse;

import java.util.List;

abstract public class SpikingNeuron implements AbstractNeuron {
    public final List<AbstractSynapse<AbstractNeuron>> synapses = new Lst<>();
    public boolean firing = false, firingNext = false;
    public double timeFired = Double.NEGATIVE_INFINITY;    // last round the neuron fired

    /**
     * Compute the activation the neuron gets from its neighbours
     */
    protected double activation() {
        double a = 0;

        for (var s : synapses)
            a += s.getInput() * s.weight;

        return a;
    }


    abstract public void update(double ambientActivation, RealtimeBrain b);

    public void commit() {
        this.firing = firingNext;
    }

}
