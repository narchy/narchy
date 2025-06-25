package jcog.tensor.rl.pg;

import jcog.tensor.Tensor;
import jcog.tensor.model.DNCMemory;

/**
 * An older REINFORCE variant using Differentiable Neural Computer (DNC) memory.
 * This class is part of an older API.
 *
 * @deprecated This class is part of an older API. For similar functionality, consider building custom agents
 *             using `jcog.tensor.rl.pg3` components, potentially by creating a custom policy network
 *             that incorporates DNC memory and using it with {@link jcog.tensor.rl.pg3.ReinforceAgent}
 *             or {@link jcog.tensor.rl.pg3.VPGAgent}. The `pg3` package is the focus for new development.
 */
@Deprecated
public class ReinforceDNC extends Reinforce {

    public final DNCMemory mem;

    /** the next state extended, computed prior to remember() */
    private transient Tensor stateExtended;

    public ReinforceDNC(int inputs, int outputs,
                        int hiddenPolicy, int episodeLen,
                        int memorySize, int memoryWidth, int memoryReads,
                        DNCMemory.EraseMode erase) {
        this(inputs, outputs, hiddenPolicy, episodeLen,
                new DNCMemory(memorySize, memoryWidth, memoryReads,
                        new DNCMemory.ContentBasedAddressing(),
                        new DNCMemory.UsageBasedRetention(),
                        erase));
    }

    public ReinforceDNC(int inputs, int outputs, int hiddenPolicy, int episodeLen, DNCMemory mem) {
        super(inputs, outputs, hiddenPolicy, episodeLen, mem.stateSize(), mem.controlSize());
        this.mem = mem;
    }

    @Override
    protected void reset() {
        super.reset();
        mem.reset();
    }

    @Override
    protected double[] _action(Tensor state) {
        Tensor actionLogits;
        synchronized (mem) {

            /*
              currentState has shape [1, originalInputs]
              We read from the DNC => shape [numReads, memoryWidth].
              Flatten that read to shape [1, (numReads * memoryWidth)] so we can concat with currentState.
             */

            // 1) Read from DNC memory (shape: [numReads, memoryWidth])
            var readRow = mem.read().flattenRow();

            // 2) Concatenate the read to currentState => shape [1, originalInputs + numReads*memoryWidth]
            stateExtended = state.concat(readRow);

            // 3) Compute policy network output => shape [1, (actionSize + controlSize)]
            var policyOutput = policy.apply(stateExtended);

            // 4) Split output into action and control parts.
            var totalSize = policyOutput.cols();
            var controlSize = mem.controlSize();
            var actionSize = totalSize - controlSize;

            actionLogits = policyOutput.slice(0, actionSize);
            var controlSlice = policyOutput.slice(actionSize, policyOutput.cols());

            // Extract control signals (with special handling when reads==1)
            mem.update(mem.extractControlSignals(controlSlice));
        }
        return sampleAction(actionLogits.detach());
    }

    @Override
    protected void remember(double reward, Tensor currentState, double[] action) {
        super.remember(reward, stateExtended, action);
    }

}
