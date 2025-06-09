package jcog.nn.spiking0.neuron;

/**
 * integrate-and-fire neuron
 */
public class IFNeuron extends SpikingNeuron {

    public boolean positive;

    public double potential = 0;

    private static final double firingThreshold =
            //1;
            //2;
            16;

    public IFNeuron(boolean positive) {
        this.positive = positive;
    }

    @Override
    public double getOutput() {
        return firing ? (positive ? +1 : -1) : 0;
    }

    @Override
    public void update(double ambientActivation, RealtimeBrain b) {
        potential += (positive ? ambientActivation : -ambientActivation) + activation();

        boolean f;
        if (positive) {
            if (potential < 0) potential = 0;
            f = potential >= +firingThreshold;
        } else {
            if (potential > 0) potential = 0;
            f = potential <= -firingThreshold;
        }

        if (f) {
            firingNext = true;
            timeFired = b.t;
            potential = 0;
        } else {
            firingNext = false;
        }
    }
}
