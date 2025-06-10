package jcog.tensor.spike;

import jcog.data.list.Lst;

import java.util.BitSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/** freeform spiking neural network */
class SpikingNetwork {

    public final List<SpikingNeuron> neurons = new Lst<>();
    private final BitSet inputs = new BitSet(), outputs = new BitSet();

    public SpikingNetwork() {
    }

    public static SpikingNetwork random(int inputs, int outputs, int hidden) {

        var snn = new SpikingNetwork();
        for (var i = 0; i < inputs; i++)
            snn.setInputs(snn.addNeuron(1.0, 0.0, 1.0));

        for (var i = 0; i < hidden; i++)
            snn.addNeuron(1.0, 0.0, 1.0);

        for (var i = 0; i < outputs; i++)
            snn.setOutputs(snn.addNeuron(1.0, 0.0, 1.0));

        int s = snn.neurons.size();

        //random connections
        Random rand = ThreadLocalRandom.current();
        for (var i = 0; i < s; i++) {
            for (var j = 0; j < s; j++) {
                if (i == j) continue;
                if (!snn.outputs.get(i) && !snn.inputs.get(j)) {
                    var w = Math.abs(rand.nextGaussian() * 0.5);
                    snn.addSynapse(i, j, w);
                }
            }
        }
        return snn;
    }

    public int addNeuron(double threshold, double resting_potential, double refractory_period) {
        var id = neurons.size();
        neurons.add(new SpikingNeuron(id, threshold, resting_potential, refractory_period));
        return id;
    }

    public int inputCount() {
        return inputs.cardinality();
    }

    public int outputCount() {
        return outputs.cardinality();
    }
    
    public SpikingSynapse addSynapse(int from, int to, double weight) {
        var s = neurons.size();
        if (from < s && to < s) {
            var syn = neurons.get(from).connect(to, weight);
            neurons.get(to).synapsesIn.add(syn);
            return syn;
        } else
            throw new IllegalArgumentException("One or both neuron IDs are out of range");
    }

    public int getInputNeuronIndex(int inputIndex) {
        // Return the index of the neuron corresponding to this input
        return inputIndex;
    }

    public int getOutputNeuronIndex(int outputIndex) {
        // Return the index of the neuron corresponding to this output
        return neurons.size() - outputs.cardinality() + outputIndex;
    }

    public void setInputs(int... inputIds) {
        for (var id : inputIds)
            inputs.set(id);
    }

    public void setOutputs(int... outputIds) {
        for (var id : outputIds)
            outputs.set(id);
    }

    public BitSet forward(double[] x, double t) {
        // Feed inputs to input neurons
        for (var i = inputs.nextSetBit(0); i >= 0; i = inputs.nextSetBit(i + 1)) {
            var n = neurons.get(i);
            n.input(x[i]);
        }

        // Propagate through the network
        for (var n : neurons) {
            if (outputs.get(n.id)) continue;

            if (n.update(t)) {
                for (var s : n.synapses)
                    neurons.get(s.target).acceptSpike(s.weight);
            }
        }


        // Collect output spikes
        var outSpikes = new BitSet(outputs.cardinality());
        for (int o = outputs.nextSetBit(0), j = 0; o >= 0; o = outputs.nextSetBit(o + 1), j++) {
            if (neurons.get(o).update(t))
                outSpikes.set(j);
        }

        //System.out.println(outSpikes);
        return outSpikes;
    }

    public List<SpikingNeuron> getNeurons() {
        return neurons;
    }

    interface SpikingLearningAlgorithm {
        void train(SpikingNetwork network, double[] inputs, BitSet target_outputs);

        void updateWeights(SpikingNetwork network);
    }

    static class SpikingNeuron {
        public final int id;
        public double membrane_potential;
        public double threshold;
        private double resting_potential;
        private double refractory_period;
        private double last_spike_time;

        /** outgoing synapses */
        public final List<SpikingSynapse> synapses = new Lst<>();

        public final List<SpikingSynapse> synapsesIn = new Lst<>();

        public SpikingNeuron(int id, double threshold, double resting_potential, double refractory_period) {
            this.id = id;
            this.threshold = threshold;
            this.resting_potential = resting_potential;
            this.refractory_period = refractory_period;
            this.membrane_potential = resting_potential;
            this.last_spike_time = Double.NEGATIVE_INFINITY;
        }

        public SpikingSynapse connect(int targetNeuronId, double weight) {
            var s = new SpikingSynapse(id, targetNeuronId).weight(weight);
            synapses.add(s);
            return s;
        }

        public void input(double input) {
            membrane_potential += input;
        }

        /** @return true if spiked */
        public boolean update(double current_time) {
            if (current_time - last_spike_time > refractory_period) {
                if (membrane_potential >= threshold) {
                    spike(current_time);
                    return true;
                }
            }

            return false;
        }

        private void spike(double current_time) {
            last_spike_time = current_time;
            membrane_potential = resting_potential;
        }

        public void acceptSpike(double weight) {
            membrane_potential += weight;
        }

    }

    static class SpikingSynapse {
        public final int source, target;

        public double weight;

        /** eligibility trace (optional) */
        public double eligibility;

        public SpikingSynapse(int source, int target) {
            this.source = source;
            this.target = target;
        }

        public SpikingSynapse weight(double weight) {
            this.weight = weight;
            return this;
        }
        public void eligibility(double eligibilityTrace) { this.eligibility = eligibilityTrace; }
    }
}
