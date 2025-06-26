package jcog.tensor.rl.pg2;

// Removed unused imports for Tensor and Experience2 as they are not directly used in this abstract class's code.
// Subclasses will import them as needed.

import jcog.Fuzzy;
import jcog.Util;
import jcog.agent.Agent;
import jcog.math.normalize.FloatNormalizer;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg2.memory.AgentMemory;
import jcog.tensor.rl.pg2.stats.DefaultMetricCollector;
import jcog.tensor.rl.pg2.stats.MetricCollector;
import org.eclipse.collections.api.block.function.primitive.FloatToFloatFunction;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * An abstract base class for implementing {@link PolicyGradientAgent}.
 * This class provides default implementations for managing the training mode status
 * and the update count. Concrete agent implementations should extend this class
 * and implement the core algorithmic logic.
 *
 * <p>Subclasses are responsible for initializing their specific networks, optimizers,
 * memory buffers, and configurations. They must also override {@link #setTrainingMode(boolean)}
 * to propagate the training mode to their internal neural network components.</p>
 */
public abstract class AbstractPolicyGradientAgent extends Agent implements PolicyGradientAgent {

    //public final FloatRange actionRevise = FloatRange.unit(1.0f);
    private final FloatToFloatFunction rewardNorm = new FloatNormalizer(2, 1000);
    @Deprecated private final double[] lastAction;
    public boolean rewardNormalize, inputPolarize, rewardPolarize;

    /**
     * Flag indicating whether the agent is in training mode. Defaults to true.
     * Subclasses should use and respect this flag in their {@code selectAction}
     * and {@code recordExperience} methods.
     */
    protected boolean trainingMode = true;

    /**
     * Counter for the number of learning updates performed.
     * Incremented by calling {@link #incrementUpdateCount()}.
     */
    protected long updateCount = 0;

    final RandomGenerator rng = new XoRoShiRo128PlusRandom();

    public final AgentMemory memory;
    protected MetricCollector metricCollector;

    public AbstractPolicyGradientAgent(int i, int o, AgentMemory memory, @Nullable MetricCollector metricCollector) {
        super(i, o);
        this.memory = Objects.requireNonNull(memory, "AgentMemory cannot be null");
        this.lastAction = new double[o];
        if (metricCollector == null) {
            //System.err.println("Warning: No MetricCollector provided to " + getClass().getSimpleName() + ". Using DefaultMetricCollector.");
            this.metricCollector = new DefaultMetricCollector(); // Fallback to default if null
        } else {
            this.metricCollector = metricCollector;
        }
    }

    public void setMetricCollector(MetricCollector metricCollector) {
        this.metricCollector = metricCollector;
    }

    /**
     * Gets the metric collector associated with this agent.
     * @return The {@link MetricCollector} instance.
     */
    public MetricCollector getMetricCollector() {
        return this.metricCollector;
    }

    @Override
    public final void recordExperience(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null");
        if (!this.trainingMode)
            return; // Do not record or update if not in training mode

        this.memory.add(experience);
        update(0); // totalSteps not critical for this VPG update logic
    }
    /**
     * {@inheritDoc}
     * <p>This implementation returns the current value of the protected {@code trainingMode} field.</p>
     */
    @Override
    public boolean isTrainingMode() {
        return this.trainingMode;
    }

    /**
     * {@inheritDoc}
     * <p>This implementation sets the protected {@code trainingMode} field.
     * Subclasses <strong>must</strong> override this method to also propagate the training mode
     * to their internal neural networks (e.g., policy and value function networks) by calling
     * their respective {@code train(boolean)} methods.</p>
     * <pre>{@code
     * @Override
     * public void setTrainingMode(boolean training) {
     *     super.setTrainingMode(training); // Sets the flag in BasePolicyGradientAgent
     *     this.policy.train(training);
     *     if (this.valueFunction != null) {
     *         this.valueFunction.train(training);
     *     }
     *     if (!training) { // Optional: clear memory when switching to evaluation
     *         this.memory.clear();
     *     }
     * }
     * }</pre>
     */
    @Override
    public void setTrainingMode(boolean training) {
        this.trainingMode = training;
        // Subclasses MUST override to propagate to their networks.
    }

    /**
     * {@inheritDoc}
     * <p>This implementation returns the current value of the protected {@code updateCount} field.</p>
     */
    @Override
    public long getUpdateCount() {
        return this.updateCount;
    }

    /**
     * Increments the learning update counter.
     * Concrete agent implementations should call this method after successfully
     * performing a learning update (e.g., at the end of their {@code update} method).
     */
    protected void incrementUpdateCount() {
        this.updateCount++;
    }

    /**
     * Records a metric using the agent's {@link MetricCollector}.
     * This is a convenience method for subclasses.
     *
     * @param metricName The name of the metric.
     * @param value The value of the metric.
     */
    protected void recordMetric(String metricName, double value) {
        // The step for metrics recorded during an update is typically the current updateCount.
        // If called outside an update, the step context might need to be different.
        // For now, assume it's called when updateCount is relevant.
        this.metricCollector.record(metricName, value, this.updateCount);
    }

    // Abstract methods from PolicyGradientAgent (selectAction, recordExperience, update,
    // getPolicy, getValueFunction, getConfig, clearMemory) must be implemented by concrete subclasses.

    @Override
    public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
        Util.replaceNaNwithRandom(input, rng);
        double r = Double.isNaN(reward) ? rng.nextFloat() : reward;
        if (inputPolarize) Fuzzy.polarize(input);
        //if (rng.nextBoolean(actionRevise.asFloat())) pg.reviseAction(this.lastAction);
        if (rewardNormalize) r = rewardNorm.valueOf((float) r);
        if (rewardPolarize) r = Fuzzy.polarize(r);
        double[] a = act(inputPrev, actionPrev, r, input, false);
        Fuzzy.unpolarize(a);
        System.arraycopy(a, 0, actionNext, 0, actionNext.length);
        System.arraycopy(a, 0, this.lastAction, 0, a.length);
    }

    private double[] act(double[] inputPrev, double[] actionPrev, double reward, double[] input, boolean done) {
        var S = Tensor.row(input);
        var action = selectAction(S, false);
        recordExperience(new Experience2(inputPrev!=null ? Tensor.row(inputPrev) : null, actionPrev, reward, S, done));
        return action;
    }


}
