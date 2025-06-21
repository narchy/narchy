package jcog.nndepr.spiking0.learn;

import jcog.nndepr.spiking0.neuron.SpikingNeuron;

/**
 * <p>A PlasticTermination implementing a PlasticityRule that accepts spiking input.</p>
 *
 * <p>Spiking input must be dealt with in order to run learning rules in
 * a spiking SimulationMode. Spiking input is also the only way to simulate spike-timing-dependent
 * plasticity.</p>
 *
 * from: Nengo STDPTarget
 * see: https://www.izhikevich.org/publications/spnet.pdf
 * see: https://fzenke.net/auryn/doku.php?id=tutorials:writing_your_own_plasticity_model
 */
public class STDPSynapseLearning extends SpikeSynapseLearning {

    double alpha = 1;

    @Override
    protected double update(SpikingNeuron a, SpikingNeuron b) {
        double scale = 1/(1 + Math.abs(a.timeFired - b.timeFired));
        if (b.firing && a.timeFired < b.timeFired) {
            return +alpha * scale;
        } else if (a.firing && a.timeFired > b.timeFired) {
            return -alpha * scale;
        }
        return 0;
    }

//    // Remember 2 spikes in the past, for triplet based learning rules
//    private static final int HISTORY_LENGTH = 2;
//
//    @Deprecated private final RealtimeBrain brain;
//
//    private float myLastTime = 0.0f;
//
//    private float[][] myPreSpikeHistory;
//    private float[][] myPostSpikeHistory;
//    private boolean[] myPreSpiking;
//    private boolean[] myPostSpiking;
//
//    private float[] myPostTrace1;
//    private float[] myPostTrace2;
//    private float[] myPreTrace1;
//    private float[] myPreTrace2;
//
//    private final float myA2Minus = 6.6e-3f;
//    private final float myA3Minus = 3.1e-3f;
//    private final float myTauMinus = 33.7f;
//    private final float myTauX = 101.0f;
//    private final float myA2Plus = 8.8e-11f;
//    private final float myA3Plus = 5.3e-2f;
//    private final float myTauPlus = 16.8f;
//    private final float myTauY = 125.0f;
//
//    protected float myLearningRate = 5e-7f;
//
//    /**
//     * @param node The parent Node
//     * @param name Name of this Termination
//     * @param synapses Node-level Terminations that make up this Termination. Must be
//     *        all LinearExponentialTerminations
//     * @throws StructuralException If dimensions of different terminations are not all the same
//     */
//    public STDP(RealtimeBrain b)  {
//        this.brain = b;
//        var neurons = b.neurons;
////        super(node, name, nodeTerminations);
////        setOriginName(Neuron.AXON);
//        //int preLength = synapses[0].getDimensions(), postLength = synapses.length;
//        int preLength = neurons.size(), postLength = neurons.size(); //TODO ??
//
//        myPostTrace1 = new float[postLength];
//        myPostTrace2 = new float[postLength];
//        myPreTrace1 = new float[preLength];
//        myPreTrace2 = new float[preLength];
//    }
//
//    @Override
//    protected double update(SpikingNeuron a, SpikingNeuron b) {
//        throw new TODO();
//    }
//
//    private void updateInput(float time)  {
////        var input = this.get();
//
////        if (!(input instanceof SpikeOutput)) {
////            throw new StructuralException("Termination must be Spiking in STDPTermination");
////        }
//
////        if (myPreSpikeHistory[0].length != ((SpikeOutput)input).getDimension()) {
////            throw new IllegalArgumentException("Expected activity of dimension "
////                    + myPreSpikeHistory[0].length + ", got dimension " + ((SpikeOutput)input).getDimension());
////        }
//
//        //boolean[] spikes = ((SpikeOutput)input).getValues();
//        int n = myPreSpiking.length;
//        for (int i = 0; i < n; i++) {
//            AbstractNeuron N = brain.neurons.get(i);
//            if (N instanceof IzhikevichNeuron I && I.firing) {
//                for (int j = HISTORY_LENGTH-1; j > 0; j--) {
//                    myPreSpikeHistory[j][i] = myPreSpikeHistory[j-1][i];
//                }
//                myPreSpikeHistory[0][i] = time;
//                myPreSpiking[i] = true;
//            } else {
//                myPreSpiking[i] = false;
//            }
//        }
//
//    }
//
//    public void updateTransform(float time, int start, int end) {
//        if (myLastTime < time) {
//            myLastTime = time;
//            this.updateInput(time);
//        }
//
//        // before dOmega
//        for (int post_i = 0; post_i < myPostTrace1.length; post_i++) {
//            if (myPostSpiking[post_i]) {
//                myPostTrace1[post_i] += 1.0f;
//            }
//            myPostTrace1[post_i] -= myPostTrace1[post_i] / myTauMinus;
//            if (myPostTrace1[post_i] < 0.0f) {myPostTrace1[post_i] = 0.0f;}
//        }
//
//        for (int pre_i = 0; pre_i < myPreTrace1.length; pre_i++) {
//            if (myPreSpiking[pre_i]) {
//                myPreTrace1[pre_i] += 1.0f;
//            }
//            myPreTrace1[pre_i] -= myPreTrace1[pre_i] / myTauPlus;
//            if (myPreTrace1[pre_i] < 0.0f) {myPreTrace1[pre_i] = 0.0f;}
//        }
//
//        //dOmega
//        float[][] transform = this.getTransform();
//
//        for (int post_i = start; post_i < end; post_i++) {
//            for (int pre_i = 0; pre_i < transform[post_i].length; pre_i++) {
//                if (myPreSpiking[pre_i]) {
//                    transform[post_i][pre_i] += preDeltaOmega(time - myPostSpikeHistory[0][post_i],
//                            time - myPreSpikeHistory[1][pre_i], transform[post_i][pre_i], post_i, pre_i);
//                }
//                if (myPostSpiking[post_i]) {
//                    transform[post_i][pre_i] += postDeltaOmega(time - myPostSpikeHistory[0][post_i],
//                            time - myPreSpikeHistory[1][pre_i], transform[post_i][pre_i], post_i, pre_i);
//                }
//            }
//        }
//
//        // after dOmega
//        for (int pre_i = 0; pre_i < myPreTrace2.length; pre_i++) {
//            if (myPreSpiking[pre_i]) {
//                myPreTrace2[pre_i] += 1.0f;
//            }
//            myPreTrace2[pre_i] -= myPreTrace2[pre_i] / myTauX;
//            if (myPreTrace2[pre_i] < 0.0f) {myPreTrace2[pre_i] = 0.0f;}
//        }
//
//        for (int post_i = 0; post_i < myPostTrace2.length; post_i++) {
//            if (myPostSpiking[post_i]) {
//                myPostTrace2[post_i] += 1.0f;
//            }
//            myPostTrace2[post_i] -= myPostTrace2[post_i] / myTauY;
//            if (myPostTrace2[post_i] < 0.0f) {myPostTrace2[post_i] = 0.0f;}
//        }
//
//
//    }
//
//    /**
//     * @return The transformation matrix, which is made up of the
//     *   weight vectors for each of the PlasticNodeTerminations within.
//     *   This can be thought of as the connection weight matrix in most cases.
//     */
//    public float[][] getTransform() {
//        throw new TODO();
////        Target[] terms = this.getNodeTerminations();
////        float[][] transform = new float[terms.length][];
////        for (int postIx = 0; postIx < terms.length; postIx++) {
////            PlasticNodeTarget pnt = (PlasticNodeTarget) terms[postIx];
////            transform[postIx] = pnt.getWeights();
////        }
////
////        return transform;
//    }
//
//    private float preDeltaOmega(float timeSinceDifferent, float timeSinceSame,
//                                float currentWeight, int postIndex, int preIndex) {
//        float result = myPostTrace1[postIndex] * (myA2Minus + myPreTrace2[preIndex] * myA3Minus);
//
//        return myLearningRate * result;
//    }
//
//    private float postDeltaOmega(float timeSinceDifferent, float timeSinceSame,
//                                 float currentWeight, int postIndex, int preIndex) {
//        float result = myPreTrace1[preIndex] * (myA2Plus + myPostTrace2[postIndex] * myA3Plus);
//
//        return -1 * myLearningRate * result;
//
//    }

}
