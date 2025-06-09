package jcog.nn.spiking0;


public class OutputNeuron implements AbstractNeuron {
    private double activation;

    public void activate(double x) {
        activation += x;
    }
    
    @Override
    public double getOutput() {
        return activation;
    }

    public void clear() {
        activation = 0;
    }
}

