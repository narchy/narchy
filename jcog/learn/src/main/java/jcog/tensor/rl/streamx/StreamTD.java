//package jcog.tensor.rl.streamx;
//
//import jcog.tensor.Models;
//import jcog.tensor.Tensor;
//import static jcog.tensor.rl.pg.StreamComponents.*;
//
///**
// * Stream TD(λ) implementation focusing on value learning
// */
//public class StreamTD extends AbstractPG {
//
//    private final Models.Layers value;
//    private final ObservationNormalizer obsNorm;
//    private final EligibilityTraces traces;
//    private final ObGD optimizer;
//    private final double gamma = 0.99;
//
//    public StreamTD(int inputs, int outputs, int hiddenDim) {
//        super(inputs, outputs);
//
//        this.obsNorm = new ObservationNormalizer(inputs);
//
//        // Build value network with LayerNorm
//        this.value = new Models.Layers(
//            Tensor::reluLeaky,
//            null,
//            false,
//            new LayerNormLinear(inputs, hiddenDim),
//            new LayerNormLinear(hiddenDim, hiddenDim),
//            new LayerNormLinear(hiddenDim, 1)
//        );
//
//        // Initialize traces and optimizer
//        this.traces = new EligibilityTraces(
//            Tensor.parameters(value).toList(),
//            gamma,
//            0.8
//        );
//
//        this.optimizer = new ObGD(1.0, 2.0);
//
//        trainMode(false);
//    }
//
//    @Override
//    protected double[] _action(Tensor state) {
//        // For TD learning, we output a value estimate rather than an action
//        Tensor normalized = obsNorm.update(state);
//        Tensor value = this.value.apply(normalized);
//        return new double[] { value.scalar() };
//    }
//
//    @Override
//    public double[] act(double[] input, double reward) {
//        Tensor state = Tensor.row(input);
//
//        // Current value estimate
//        Tensor value = getValue(state);
//
//        // Next state value for TD error
//        Tensor nextValue = value; // Current value as baseline
//
//        // Compute TD error
//        double tdError = reward + gamma * nextValue.scalar() - value.scalar();
//
//        // Update traces and parameters
//        train(value, () -> {
//            value.backward();
//            traces.accumulate();
//            optimizer.step(tdError, traces);
//        });
//
//        return _action(state);
//    }
//
//    private Tensor getValue(Tensor state) {
//        return value.apply(obsNorm.update(state));
//    }
//
//    private void trainMode(boolean training) {
//        train(value, training);
//    }
//
//    @Override
//    public void onEpisodeEnd() {
//        traces.reset();
//    }
//}