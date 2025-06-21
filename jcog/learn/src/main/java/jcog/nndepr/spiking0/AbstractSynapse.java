package jcog.nndepr.spiking0;

public class AbstractSynapse<T extends AbstractNeuron> {
    public final T source, target;

    public double weight;
    public double curInput;
    public boolean invalid = true;

    public AbstractSynapse(T source, T target, double weight) {
        this.source = source;
        this.target = target;
        this.weight = weight;
    }

    public final double getInput() {
        if (invalid) {
            curInput = source.getOutput();
            invalid = false;
        }
        return curInput;
    }

    public void weightAdd(double dw) {
        weight += dw;
    }
}
