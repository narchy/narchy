package jcog.nn.spiking0;

public class InputNeuron implements AbstractNeuron {
    
    private double senseInput;

    public final InputNeuron setInput(double x) {
        this.senseInput = x; return this;
    }

    @Override
    public final double getOutput() {
        return senseInput;
    }


}
