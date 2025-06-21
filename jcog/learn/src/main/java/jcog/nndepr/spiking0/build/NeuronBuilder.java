package jcog.nndepr.spiking0.build;

import jcog.nndepr.spiking0.OutputNeuron;
import jcog.nndepr.spiking0.critterding.CritterdingNeuron;

import java.util.LinkedList;
import java.util.List;

/**
 * see archneuronz.h
 */
public class NeuronBuilder {

    // inhibitory neuron by flag
    public boolean isInhibitory;
    // Consistent Synapses flag
    //boolean hasConsistentSynapses;
    // inhibitory synapses flag
    boolean hasInhibitorySynapses;
    // neuron firing potential

    // motor neuron ability (excititatory only) flag
    //boolean isMotor; //isMotor if motor!=null
    // function
    public OutputNeuron motor;
    // synaptic plasticity by flag
    public boolean isPlastic;
    // factors
    public final List<SynapseBuilder> synapseBuilders = new LinkedList();

    public CritterdingNeuron newNeuron() {
        CritterdingNeuron n = new CritterdingNeuron();
        n.isPlastic = isPlastic;
        n.isInhibitory = isInhibitory;
        n.motor = motor;
        return n;
    }
}
