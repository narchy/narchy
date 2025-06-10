package jcog.tensor.spike;

import jcog.Util;
import jcog.tensor.Tensor;
import org.ejml.simple.SimpleMatrix;

import java.util.BitSet;
import java.util.List;
import java.util.function.UnaryOperator;


/**
 * adapter for embedding spiking network in a Tensor graph
 */
class SpikeTensor implements UnaryOperator<Tensor> {
    private final SpikingNetwork net;
    private final int steps;
    private double t, dt;

    private Tensor lastInput;
    private BitSet lastSpikes;

    @Deprecated private final float learningRate = 0.5f;

    public SpikeTensor(SpikingNetwork net, double t, double dt, int steps) {
        this.net = net;
        this.t = t;
        this.dt = dt;
        this.steps = steps;
    }

    @Override public Tensor apply(Tensor input) {
        var o = net.outputCount();
        var output = new Tensor(1, o, input.hasGrad());
        var oo = output.array();
        var spikes = forward(input.array());
        for (var i = 0; i < o; i++)
            oo[i] = spikes.get(i) ? 1 : 0;

        if (output.hasGrad()) {
            this.lastInput = input;
            this.lastSpikes = spikes;
            output.op = new Tensor.TensorOp(input) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    backpropagateGradient(grad);
                }
            };
        }

        return output;
    }

    protected BitSet forward(double[] inputs) {
        if (inputs.length!=net.inputCount())
            throw new UnsupportedOperationException("network input size mismatch");

        var spikes = new BitSet(net.outputCount());
        for (int j = 0; j < steps; j++) {
            spikes.or(net.forward(inputs, t));
            t += dt;
        }
        return spikes;
    }

    protected SimpleMatrix gradient(SimpleMatrix outputGrad) {
        var inputGrad = new SimpleMatrix(lastInput.rows(), lastInput.cols());
        var surrogateDeriv = computeSurrogateDerivative();

        for (int i = 0; i < net.outputCount(); i++) {
            if (lastSpikes.get(i)) {
                for (int j = 0; j < lastInput.cols(); j++) {
                    inputGrad.set(0, j, inputGrad.get(0, j) +
                            outputGrad.get(0, i) * surrogateDeriv.get(0, i) * lastInput.data(0, j));
                }
            }
        }

        return inputGrad;
    }

    protected void backpropagateGradient(SimpleMatrix outputGrad) {
        int neuronCount = net.neurons.size();
        double[] neuronGradients = new double[neuronCount];

        // Set output gradients
        for (int i = 0; i < net.outputCount(); i++) {
            int oo = net.getOutputNeuronIndex(i);
            neuronGradients[oo] = outputGrad.get(i);
        }

        // Backpropagate through the network
        List<SpikingNetwork.SpikingNeuron> neurons = net.getNeurons();
        for (int i = neurons.size() - 1; i >= 0; i--) {
            SpikingNetwork.SpikingNeuron neuron = neurons.get(i);
            double neuronGrad = neuronGradients[i];

            // Apply surrogate derivative
            double surrogateDeriv = computeSurrogateDerivative(neuron);
            neuronGrad *= surrogateDeriv;

            if (neuronGrad != 0) {
                // Update weights and propagate gradient
                for (SpikingNetwork.SpikingSynapse synapse : neuron.synapsesIn) {
                    int sourceIndex = synapse.source;

                    // Update weight
                    double dw = learningRate * neuronGrad * (lastSpikes.get(sourceIndex) ? 1 : 0);
                    synapse.weight = Util.clamp(synapse.weight + dw, 0, +1);

                    // Propagate gradient
                    neuronGradients[sourceIndex] += neuronGrad * synapse.weight;
                }
            }
        }
    }

    protected SimpleMatrix computeInputGradient() {
        SimpleMatrix inputGrad = new SimpleMatrix(1, net.inputCount());
        for (int i = 0; i < net.inputCount(); i++) {
            int inputNeuronIndex = net.getInputNeuronIndex(i);
            double grad = 0;
            for (SpikingNetwork.SpikingSynapse synapse : net.neurons.get(inputNeuronIndex).synapses) {
                grad += synapse.weight * computeSurrogateDerivative(net.neurons.get(synapse.target));
            }
            inputGrad.set(i, grad);
        }
        return inputGrad;
    }

    protected double computeSurrogateDerivative(SpikingNetwork.SpikingNeuron neuron) {
        // Using a simple surrogate gradient function: max(0, 1 - |x|)
        // where x is the neuron's membrane potential
        double x = neuron.membrane_potential / neuron.threshold;
        return Math.max(0, 1 - Math.abs(x));
    }

    protected SimpleMatrix computeSurrogateDerivative() {
        var deriv = new SimpleMatrix(1, net.outputCount());
        for (int i = 0; i < net.outputCount(); i++) {
            // Using a simple surrogate gradient function: max(0, 1 - |x|)
            double x = lastInput.data(0, i % lastInput.cols());
            deriv.set(0, i, Math.max(0, 1 - Math.abs(x)));
        }
        return deriv;
    }
}

