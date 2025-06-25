package jcog.tensor.rl.pg3;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg3.configs.VPGAgentConfig; // Using VPGAgentConfig as a base

import java.util.function.UnaryOperator;

// StreamAC is an Actor-Critic model. VPGAgent is the pg3 Actor-Critic.
// This class would extend VPGAgent and potentially override update logic
// or how experiences are processed if "streaming" implies continuous updates
// rather than episodic ones.
public class StreamACAgent extends VPGAgent {

    // If StreamAC has a fundamentally different network structure or update rule
    // not configurable via VPGAgentConfig, those would be implemented here.
    public StreamACAgent(VPGAgentConfig config, int stateDim, int actionDim) {
        super(config, stateDim, actionDim);

        // TODO: Implement any specific modifications for StreamAC.
        // This might involve:
        // 1. Different network architectures (though VPGAgentConfig is fairly flexible).
        // 2. Overriding the `update` method if the learning process is different
        //    (e.g., truly streaming single-experience updates vs. batch/episode updates).
        // 3. Custom handling of memory or experience processing.

        System.err.println("StreamACAgent: Specific streaming logic and potential overrides need implementation.");
    }

    // Example: If StreamAC updates after every experience:
    // @Override
    // public void recordExperience(Experience2 experience) {
    //     super.recordExperience(experience); // Adds to memory
    //     if (isTrainingMode() && this.memory.size() >= 1) { // Or some small batch size
    //         update(0); // Or however totalSteps is tracked
    //         // May or may not clear memory depending on StreamAC's exact nature (on-policy vs. sliding window off-policy)
    //         if (config.memoryConfig().episodeLength() == null || config.memoryConfig().episodeLength() <=1 ) // pseudo-config check
    //            this.memory.clear();
    //     }
    // }
}
