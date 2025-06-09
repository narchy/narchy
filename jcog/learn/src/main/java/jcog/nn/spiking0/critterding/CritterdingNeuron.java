package jcog.nn.spiking0.critterding;

import jcog.Util;
import jcog.nn.spiking0.AbstractNeuron;
import jcog.nn.spiking0.AbstractSynapse;
import jcog.nn.spiking0.InterNeuron;
import jcog.nn.spiking0.OutputNeuron;
import org.jetbrains.annotations.Nullable;

public class CritterdingNeuron extends InterNeuron {

//    @Deprecated static final float weakestWeight =
//        0;
//        //0.01f;

    static final double maxSynapseWeight =
        1;
        //2;

    public boolean isInhibitory;
    public boolean isPlastic = true;
    @Nullable
    public OutputNeuron motor = null;

    public CritterdingNeuron() {
        super();
    }

    @Override
    public void forward(AbstractSynapse<AbstractNeuron>[] synapses, double alpha, double potentialFactor, double plasticStrengthen, double firingThreshold) {

        potential = Util.fma(potential, potentialFactor,
                incomingPotential(synapses, 1 /*alpha*/));

        if (isInhibitory)
            forwardInhibitory(synapses, alpha, plasticStrengthen, firingThreshold);
        else
            forwardExhibitory(synapses, alpha, plasticStrengthen, firingThreshold);
    }

    protected void forwardInhibitory(final AbstractSynapse<AbstractNeuron>[] synapses, double alpha, double plasticityStrengthen, double firingThreshold) {
        if (potential <= -firingThreshold /* * synapses.length*/)
            fireInhibitory(synapses, plasticityStrengthen, alpha);
        else
            clearInhibitory();
    }

    protected void forwardExhibitory(final AbstractSynapse<AbstractNeuron>[] synapses, double alpha, double plasticityStrengthen, double firingThreshold) {
        if (potential >= +firingThreshold /* * synapses.length*/)
            fireExhibitory(synapses, plasticityStrengthen, alpha);
        else
            clearExhibitory();
    }


    /** non-fired */
    private void clearInhibitory() {
        nextOutput = 0;
        if (potential > 0) potential = 0;
    }

    /** non-fired */
    private void clearExhibitory() {
        nextOutput = 0;
        if (potential < 0) potential = 0;
    }

    private void fireInhibitory(AbstractSynapse<AbstractNeuron>[] synapses, double plasticityStrengthen, double alpha) {
        double o = this.potential;
        potential = 0; // reset neural potential
        nextOutput = -1; // fire the neuron

        // PLASTICITY: if neuron & synapse fire together, the synapse strenghtens
        if (isPlastic) {
            for (final AbstractSynapse<AbstractNeuron> s : synapses) {
                double i = s.curInput, w = s.weight;
                ////boolean amp = ((i < 0 && w > 0) || (i > 0 && w < 0));
                double amp =
                        //similar(w, -i);
                        //similarPos(w, -i);
                        //similar(w * i, o);
                        similar(i * Math.signum(w), o);
                s.weight = clampWeight(w + amp * Math.signum(w) * alpha * plasticityStrengthen);
            }
        }
    }

    private void fireExhibitory(AbstractSynapse<AbstractNeuron>[] synapses, double plasticityStrengthen, double alpha) {
        double o = this.potential;
        potential = 0; // reset neural potential
        nextOutput = 1; // fire the neuron

        // PLASTICITY: if neuron & synapse fire together, the synapse strenghtens
        if (isPlastic) {
            for (final AbstractSynapse<AbstractNeuron> s : synapses) {
                double i = s.curInput, w = s.weight;
                //boolean amp = ((i > 0 && w > 0) || (i < 0 && w < 0));
                double amp =
                        //similar(w, i);
                        //similarPos(w, i);
                        //similar(w * i, o);
                        similar(i * Math.signum(w), o);
                s.weight = clampWeight(w + amp * Math.signum(w) * alpha * plasticityStrengthen);
            }
        }
    }

    private static double similarPos(double x, double y) {
        return Math.max(0, similar(x, y));
    }

    /** cosine similarity */
    private static double similar(double x, double y) {
        x = Util.clamp(x, -1, +1);
        y = Util.clamp(y, -1, +1);
        double d = Math.abs(x) * Math.abs(y);
        return d > Double.MIN_NORMAL ? x * y / d : 0;
    }

    private double clampWeight(double w) {
        return Util.clampSafePolar(w, maxSynapseWeight);
    }

    public boolean isInhibitory() {
        return isInhibitory;
    }

    public OutputNeuron getMotor() {
        return motor;
    }

    public void setMotor(OutputNeuron motor) {
        this.motor = motor;
    }

    public void setPlastic(boolean isPlastic) {
        this.isPlastic = isPlastic;
    }

    @Override
    public void clear() {
        //super.clear();
        potential = 0;
    }
}
