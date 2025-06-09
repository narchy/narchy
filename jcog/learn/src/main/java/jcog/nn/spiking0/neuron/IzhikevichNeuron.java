package jcog.nn.spiking0.neuron;

import java.util.concurrent.ThreadLocalRandom;

/**
 * from: https://github.com/dieuwkehupkes/Spiking-Network/tree/java/src/izhikevich/spikingnetwork
 */
public class IzhikevichNeuron extends SpikingNeuron {

    private double a, b, c, d;
    public double u;
    public double v;        // u and v when they were last computed
    private double I;      // external input to neuron
    public boolean fired = false;

    //	public double t=0;            // cur round
//	final public double timeStep = 0.1;
    private double spikeDuration = 1;
    public int nSpikes = 0;        // number of spikes since t=0;


    //public int neighbours[];     // indices pointing to the neighbours of the neuron
    //public double weights[];        // weights to the neighbours

//    public int numNeighbours = 0;
//	int numPotential = 30;        // CHANGE THIS LATER TO BE PART OF ARCHITECTURE!
//	int potentialNeighbours[] = new int[numPotential];        // indices pointing to potential neighbours

    public IzhikevichNeuron() {
        reset();
    }

    public IzhikevichNeuron(double a, double b, double c, double d) {
        this();
        setParams(a, b, c, d);
    }

    public final IzhikevichNeuron setParams(double a, double b, double c, double d) {
        this.a = a;
        this.b = b;
        this.c = c;
        this.d = d;
        return this;
    }

    public final IzhikevichNeuron excitatory() {
        return setParams(0.02, 0.2, -65 + 15 * rng(), 8 - 6 * rng());
    }

    public final IzhikevichNeuron inhibitory() {
        return setParams(0.02 + 0.08 * rng(), 0.25 - 0.05 * rng(), -65, 2);
    }

    public final IzhikevichNeuron regular() {
        return setParams(0.02, 0.2, -65, 8);
    }

    public final IzhikevichNeuron chattering() {
        return setParams(0.02, 0.2, -50, 2);
    }

    public final IzhikevichNeuron fastSpiking() {
        return setParams(0.1, 0.2, -65, 2);
    }

    public final IzhikevichNeuron intrinsicallyBursting() {
        return setParams(0.02, 0.2, -55, 4);
    }

    private static double rng() {
        return ThreadLocalRandom.current().nextDouble();
    }


//	public float[][] plot_v(double I, int nr_of_steps) {
//		// Show behaviour of neuron as a result of a constant input current
//
//		float[][] time_potential = new float[nr_of_steps][2];     // declare array to store output
//		this.nSpikes = 0;		// reset nr of spikes
//		t = 0;				// reset time
//
//		double input = I + this.I;
//
//		// compute behaviour
//		for (int i=0; i<nr_of_steps; i++) {
//			double cur_time = i*timeStep;
//			time_potential[i][0] = (float) cur_time;
//			time_potential[i][1] = (float) v;
//			computeNext(input);
//		}
//
//		System.out.print(this.nSpikes);
//
//		return time_potential;
//	}
//	public float[][] plot_u(double I, int nr_of_steps) {
//		// Show course of v as result of constant input I
//
//		float[][] time_u = new float[nr_of_steps][2];
//
//		for (int i=0; i<nr_of_steps; i++) {
//			double cur_time = i*0.1;
//			time_u[i][0] = (float) cur_time;
//			time_u[i][1] = (float) u;
//			computeNext(I);
//		}
//		return time_u;
//
//	}


    /**
     * Update values v and u of the network, and
     * update the weights if network is in
     * training mode.
     */
    @Override public void update(double ambientActivation, RealtimeBrain n) {
        boolean train = true;//n.train;
        double synapticActivation = activation();
        double inputActivation = this.I + ambientActivation + synapticActivation;

        double t = n.t;

        /* if potential crosses threshold, reset */
        if (this.v > 30) {
            this.v = this.c;
            this.u = this.u + this.d;
            this.fired = true;
            this.nSpikes++;
            this.timeFired = t;
            this.firing = true;
        } else {
            double timeStep = n.timeStep;
            this.v = this.v + 0.5 * timeStep * (0.04 * Math.pow(v, 2) + 5 * v + 140 - u + inputActivation); //% step 0.5 ms
            this.v = this.v + 0.5 * timeStep * (0.04 * Math.pow(v, 2) + 5 * v + 140 - u + inputActivation); //% for numerical
            u = u + timeStep * a * (b * this.v - u); //% stability
            if (this.spikeDuration <= t - this.timeFired) {
                this.firing = false;
            }
            this.fired = false;
        }

        setI(0.0);

    }



//    /**
//     * Update the connections of the network
//     * to his neighbours, based on their
//     * recent firing activity and the learning
//     * function
//     */
//    private void updateWeights(AbstractBrain network) {
//        double[] weights = getWeights();
//        double minWeight = network.getMinWeight();
//        double maxWeight = network.getMaxWeight();
//
//        for (int i = 0; i < this.numNeighbours; i++) {
//            double update = network.updateExistingConnection(this, network.neurons.get(i));
//            double nextWeight = weights[i] + update;
//            weights[i] = Math.max(minWeight, Math.min(maxWeight, nextWeight));
//        }
//
//        prune();     // remove connections below minimum weight of the network
//
////		// loop over close neurons to see if a connection should be established
////		int i = 0;
////		while (numNeighbours < network.getMaxNumNeighbours() && i < potentialNeighbours.length) {
////			// network.updateNonExistingConnection(this, network.getNeurons()[potentialNeighbours[i]]);
////			i++;
////		}
//
//    }

//    private void prune() {
//        /**
//         * Remove connections whose weight got
//         * lower than the minimum weight of the network
//         * the neuron is part of
//         * THIS IS NOT REALLY CORRECT, FIND A DIFFERENT WAY TO REMOVE
//         * CONNECTIONS
//         */
//		/*
//	    int i = 0;      // index of neighbour
//	    while (i < numNeighbours) {
//	      if (weights[i] < network.architecture.minWeight()) {      // remove weights that are too low
//	        weights[i] = weights[numNeighbours-1];      // remove connection and
//	        neighbours[i] = neighbours[numNeighbours-1];   // replace with last connection
//	        numNeighbours--;                            // decrease nr of neighbours
//	        potentialNeighbours[numPotential - numNeighbours] = i;  // add neuron to potential connections
//	        numPotential++;
//	      } else {
//	        i++;
//	      }
//	    }
//		 */
//    }

//    public double spikeAverage() {
//        /**
//         * Return the average number of spikes since the
//         * last reset
//         */
//        if (t == 0) return 0.0;
//        double spikeAverage = this.nSpikes / t;
//        return spikeAverage;
//    }

//    public double averageSpikePeriod() {
//        /**
//         * Return the average spike period
//         */
//        if (this.nSpikes == 0) return 0;
//        double averageSpikePeriod = t / this.nSpikes;
//        return averageSpikePeriod;
//    }

    public void validateParameters(double aMin, double aMax, double bMin, double bMax, double cMin, double cMax, double dMin, double dMax) {
        // validate parameters
        if (a < aMin || a > aMax)
            throw new IllegalArgumentException("Parameter a invalid for type of neuron " + this.getClass());
        if (b < bMin || b > bMax)
            throw new IllegalArgumentException("Parameter b invalid for type of neuron " + this.getClass());
        if (c < cMin || c > cMax)
            throw new IllegalArgumentException("Parameter c invalid for type of neuron " + this.getClass());
        if (d < dMin || d > dMax)
            throw new IllegalArgumentException("Parameter d invalid for type of neuron " + this.getClass());
    }

    public void reset() {
        // reset neuron and time
        this.v = this.c;
        this.u = this.b * this.c;
        this.nSpikes = 0;
        this.timeFired = Double.NEGATIVE_INFINITY;
    }

    public void resetI() {
        // Reset the input variable I to its standard value
        this.I = 0;
    }

    public void printConnections() {
        for (var s : synapses)
            System.out.println(s + "\t" + s.weight);
    }

    public double a() {
        return this.a;
    }

    public double b() {
        return this.b;
    }

    public double c() {
        return this.c;
    }

    public double d() {
        return this.d;
    }

//    // setters
//
//    public void setConnections(int[] connectTo, double[] weights) throws IllegalArgumentException {
//        // set weights to the list of neurons that is
//        // inputted
//
//        // check if input is valid
//        IllegalArgumentException unequalLengthException = new IllegalArgumentException("Number of neurons and number of weights should be equal");
//        if (connectTo.length != weights.length) {
//            throw unequalLengthException;
//        }
//
//        this.setNeighbours(connectTo);
//        this.setWeights(weights);
//        this.numNeighbours = neighbours.length;
//    }

    public void setI(double activation) {
        // increase I with activation
        this.I = activation;
    }

    public double I() {
        // return I
        return I;
    }

    public void set_v(double v) {
        // Set v to a certain value
        this.v = v;
    }


    @Override
    public double getOutput() {
        return firing ? +1 : 0;
    }
}
