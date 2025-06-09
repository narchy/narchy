package jcog.nn.spiking0;

public abstract class InterNeuron extends OutputNeuron {

    public double potential, output, nextOutput;

    public abstract void forward(AbstractSynapse<AbstractNeuron>[] synapses, double alpha, double potentialFactor, double plasticStrengthen, double firingThreshold);

    /**
     * every connection do it's influence on the neuron's total potential
     */
    protected double incomingPotential(AbstractSynapse<AbstractNeuron>[] synapses, double alpha) {
        if (synapses == null)
            return 0;

        double dPot = 0;
        for (final AbstractSynapse<AbstractNeuron> s : synapses) {

//            // decay synaptic weights
//            if (isPlastic) {
////                if (Math.abs(s.weight) > weakestWeight)
//                s.weight = lerpSafe(alpha, s.weight, s.weight * plasticityWeakenFactor);
//            }

            dPot += s.getInput() * s.weight;
        }
        return dPot;
    }

    public double getPotential() {
        return potential;
    }

    @Override
    public double getOutput() {
        return output;
    }
}
