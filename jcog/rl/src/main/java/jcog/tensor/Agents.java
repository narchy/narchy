package jcog.tensor;

import jcog.agent.Agent;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.model.DNCMemory;
import jcog.tensor.rl.blackbox.BayesZeroPolicy;
import jcog.tensor.rl.blackbox.CMAESZeroPolicy;
import jcog.tensor.rl.blackbox.PopulationPolicy;
import jcog.tensor.rl.dqn.PolicyAgent;
import jcog.tensor.rl.pg.*;
import jcog.tensor.rl.pg2.DDPGStrategy; // Will be replaced by pg3.DDPGAgent
import jcog.tensor.rl.pg2.PGBuilder;    // May still be used or parts adapted for pg3 configs
// PG3 imports
import jcog.tensor.rl.pg3.PPOAgent;
import jcog.tensor.rl.pg3.VPGAgent;
import jcog.tensor.rl.pg3.ReinforceAgent;
import jcog.tensor.rl.pg3.DDPGAgent;
import jcog.tensor.rl.pg3.SACAgent;
import jcog.tensor.rl.pg3.configs.*;


import java.util.Random;

import static java.lang.Math.round;

public class Agents {


//    public static Agent DQmlp(int inputs, int actions) {
//        int deep =
//            //0;
//            1;
//            //2;
//            //3;
//            //4;
//            //6;
//            //8;
//
//        boolean ae = false;
////        if (inputs > 96)
////            ae = true;
//
//        return DQmlp(inputs, ae, actions, deep,
//                4, 32,
//                //4, 128,
//                //8, 64,
//                //8, 256,
//                //4, 128,
//                // 0.25f, 8,
//                //0.25f, 512,
//                true
//        );
//    }

//    public static Agent DQNsmall(int inputs, int actions) {
//        return DQmlp(inputs, false, actions, 0,
//                //4, 128,
//                //1, 16,
//                0.25f, 2,
//                // 0.25f, 8,
//                //0.25f, 512,
//                true
//        );
//    }

//    public static Agent DQevolving(int i, int o) {
//        return new PolicyAgent(i, o, (I, O) -> new QPolicySimul(I, O, (ii, oo)-> {
//
//            int maxHidden = (ii+oo)*2;
//            int maxNodes = ii+oo+maxHidden;
//            int maxEdgesPerNode = 10;
//
//            EvolvingGraphNN n = new EvolvingGraphNN() {
//                {
//                    addInputs(ii);
//                    DiffableFunction outActivation =
//                        TanhActivation.the;
//                        //null;
//                        //SigmoidActivation.the;
//                    addOutputs(oo, outActivation);
////                    activationDefault = ReluActivation.the;
////                    addLayersConnectFull(
////                        b/2, b/4
////                        //    2,2,2,2
////                    );
////                    activationDefault = SigmoidActivation.the;
////                    addLayersConnectFull(oo, oo);
//                    activationDefault = ReluActivation.the;
//                    addLayersConnectFull(oo, oo);
//                    activationDefault = SigmoidActivation.the;
//                }
//                @Override
//                public void backward(double[] delta, double learningRate) {
//                    System.out.println(nodes().size() + " nodes, " + edges.size() + " edges "
//                            //+ n4(Util.sumAbs(delta))
//                    );
//                    super.backward(delta, learningRate);
//                    {
//
//                        //weight decay
//                        edges.forEach(e -> e.weight *= 0.999999);
//
//                        //float growNodes = hiddens().size() < (inputCount() + outputCount()) ? 2 : 1;
//
//                        float mutRate = 0.001f;// * edges().size();
//
//                        if (nodes().size() > maxNodes)
//                            removeWeakestNode();
//
//                        if (random.nextFloat() < mutRate) {
////                var in = N.randomNode(true, true, false);
////                var out = N.randomNode(false, true, true);
////                if (in!=out) {
////                    var h = N.newHiddenNode();
////                    N.addEdge(in, h);
////                    N.addEdge(h, out);
////                }
//
//
//                            var in1 = randomNode(true, true, false);
//                            var in2 = randomNode(true, true, false);
//                            if (in1 != in2) {
//                                var out = randomNode(false, true, true);
//                                if (in1 != out) {
//                                    var h = newHiddenNode();
//                                    addEdge(in1, h);
//                                    addEdge(in2, h);
//                                    addEdge(h, out);
//                                }
//                            }
//
//
//
//                        } else if (random.nextFloat() < mutRate) {
//                            removeWeakestEdge();
////                        } else if (random.nextFloat() < mutRate) {
////                            removeWeakestNode();
//                        } else if (random.nextFloat() < mutRate) {
//                            if (edges.size() + 1 < maxEdgesPerNode * nodes().size()) {
//                                var a = randomNode(true, true, false);
//                                var b = randomNode(false, true, false);
//                                if (a != b && !a.hasEdgeTo(b))
//                                    addEdge(a, b);
//                            }
//                        }
//                    }
//                }
//            };
//            return n;
//        })).replay(new BagReplay(512, 8));
//    }
//
//    public static Agent DQrecurrent(int i, int o) {
//        return DQrecurrent(i, o,
//                //8, 4
//                2, 4, 32
//                //0.1f, 4
//                //0.25f, 6
//                //0.25f, 8
//        );
//    }

    public static PolicyAgent CMAESZero(int inputs, int actions) {
        return new PolicyAgent(inputs, actions,
            new CMAESZeroPolicy(actions, actions, 16, 64));
    }

    public static PolicyAgent BayesZero(int inputs, int actions) {
        return BayesZero(inputs, actions, 16, 64);
    }

    public static PolicyAgent BayesZero(int inputs, int actions, int periodMin, int periodMax) {
        int memorySize = 2;
        return new PolicyAgent(inputs, actions,
                new BayesZeroPolicy(actions,
                    actions * memorySize,
                        periodMin, periodMax));
    }

    public static PolicyAgent CMAES(int inputs, int actions) {
        return new PolicyAgent(inputs, actions, new PopulationPolicy(
            new PopulationPolicy.CMAESPopulation())
        );
    }

//    public static PolicyAgent Spiking(int inputs, int actions) {
//        return new PolicyAgent(inputs, actions, new SpikingPolicy());
//    }

//    public static PolicyAgent DQmlp(int inputs, boolean inputAE, int actions, int deep, float brainsScale, int replays, boolean simul) {
//        float dropOut =
//                0;
//        //0.1f;
//        //0.2f;
//        //0.5f;
//        //0.9f;
//        //0.25f;
//        //0.75f;
//
//        int brains = (int) Math.ceil(brainsScale *
//                        //Fuzzy.mean((float)inputs, actions)
//                        Fuzzy.meanGeo((float) inputs, actions)
//                //inputs
//                //actions
//        );
//
//        IntIntToObjectFunction<Predictor> brain = (ii, oo) ->
//                mlpBrain(ii, oo, brains, deep, inputAE, dropOut);
//
//        PolicyAgent a = new PolicyAgent(inputs, actions,
//                (i, o) ->
//                        simul ?
//                                (o > 4 ?
//                                        new BranchedPolicy(i, o, 2, brain)
//                                        : new QPolicySimul(i, o, brain))
//                                :
//                                new QPolicy(
//                                        //brain.value(i,o)
//                                        () -> brain.value(i, o), () -> brain.value(i, o)
//                                )
////                new QPolicyBranched(i, o,
////                          (ii, oo) -> mlpBrain(ii, oo, brains, precise, inputAE)
////                )
////                new A2C(i,o, o)
//        );
//
//        if (replays > 0)
//            a.replay(
//                    //new SimpleReplay(16 * 1024, 1 / 3f, replays)
//                    new BagReplay(Math.max(512, replays * 2), replays)
//            );
//
//        return a;
//    }

//    private static MLP mlpBrain(int i, int o, int brains, int depth, boolean inputAE, float dropOut) {
//
//        List<MLP.LayerBuilder> layers = new Lst<>(4);
//
////        if (inputAE) {
//////                layers.add(new MLP.AutoEncoderLayerBuilder(
//////                                //Fuzzy.mean(i,o*4)
//////                                i
//////                                //Math.round(Util.lerp(0.33f, i, o))
//////                        )
//////                );
////            layers.add(new MLP.AutoEncoderLayerBuilder(
////                            //(int) Math.ceil(Util.sqrt(i))
////                            //i*2
////                            //i/3
////                            Math.min(i / 2, 64)
////                            //Fuzzy.mean(i,o*4)
////                            //i
////                            //Math.round(Util.lerp(0.33f, i, o))
////                    )
////            );
////        }
//
//        var hiddenActivation =
//                //ReluActivation.the
//                //SigmoidActivation.the
//                //TanhActivation.the
//                LeakyReluActivation.the
//                //SeluActivation.the
//                //SigLinearActivation.the
//                //new LeakyReluActivation(0.1f)
//                //EluActivation.the
//                //new SigLinearActivation(4, -1, +1)
//
//                ;
//
//        layers.add(new MLP.LinearLayerBuilder(brains, hiddenActivation));
//
//        if (depth > 0) {
//            float curve =
//                    //0.1f; //weaker
//                    1; //linear
//            //2; //brainier
//
//            for (int p = 0; p < depth; p++) {
//                float a = (float) Math.pow((p + 1f) / (depth + 1), curve);
//                int hidden = Util.lerpInt(a, brains, o);
//                //System.out.println(p + " " + a + " " + hidden);
//                layers.add(new MLP.LinearLayerBuilder(hidden, hiddenActivation));
//            }
//
//        }
//
//        /* OPTIONAL: action post-process */
//        //layers.add(new MLP.Dense(o, hiddenActivation));
//
//        //action output
//        layers.add(new MLP.LinearLayerBuilder(o, PolicyAgent.dqnOutputActivation));
//
//        //layers.add(new NormalizeLayer(o));
//
//        MLP p = new MLP(i, layers).optimizer(
////                new SGDOptimizer(0)
////                        .gradClamp(1)
//
//                //new SGDOptimizer(0)
//                //new SGDOptimizer(0).minibatches(8)
////                    .minibatches(8)
//
//                //new SGDOptimizer(0.9f)
//                //new SGDOptimizer(0.95f)
//                //new SGDOptimizer(0.99f)
//                //new SGDOptimizer(0.9f)
//            new AdamOptimizer()
//                    .gradClamp(1)
//                //new AdamOptimizer().minibatches(8)
//                //new AdamOptimizer().momentum(0.01, 0.9).epsilon(1e-2) //low momentum
//                //new AdamOptimizer().momentum(0.99, 0.99)
//                //new RMSPropOptimizer()
////            new LionOptimizer()
////                .gradClamp(1)
//                //.gradClamp(0.1)
//                //.minibatches(15)
//        );
//
//        if (dropOut > 0) {
//            for (int l = 0; l < p.layers.length; l++)
//                if (p.layers[l] instanceof LinearLayer D)
//                    D.dropout = dropOut;
//        }
//
//        p.clear(null /* init weights to zero */);
//        return p;
//    }

//    public static Agent DQrecurrent(int inputs, int actions, float brainsScale, int computeIters, int replays) {
//        int brains = (int) Math.ceil(Fuzzy.mean(inputs, actions) * brainsScale);
//        return new PolicyAgent(inputs, actions,
//                (i, o) -> new QPolicySimul(i, o,
//                    (ii, oo) -> recurrentBrain(ii, oo, brains, computeIters)
//                )
//        ).replay(
//                //new SimpleReplay(16 * 1024, 1/3f, replayIters)
//                new BagReplay(1024, replays)
//        );
//    }
//
//    public static BackpropRecurrentNetwork recurrentBrain(int inputs, int actions, int hidden, int iterations) {
//        BackpropRecurrentNetwork b = new BackpropRecurrentNetwork(
//                inputs, actions, hidden, iterations);
//        b.momentum =
//                0;
//        //0.9f;
//        b.activationFn(
//                //SigmoidActivation.the,
//                //TanhActivation.the,
//                //ReluActivation.the,
//                //LeakyReluActivation.the,
//                SigLinearActivation.the,
//                PolicyAgent.dqnOutputActivation
//
//                //new SigLinearActivation()
//                //new SigLinearActivation(0, +10, 0, +1)
//                //LinearActivation.the
//                //new LeakyReluActivation(0.1f),
//                //SinActivation.the,
//                //new SigLinearActivation()
//
//        );
//        return b;
//    }
//
//    public static Agent DQN_NTM(int inputs, int actions) {
//        return new PolicyAgent(inputs, actions,
//                (ii, oo) -> {
//                    LivePredictor.NTMPredictor p = new LivePredictor.NTMPredictor(ii, oo, 2, 2);
//                    p.clear(ThreadLocalRandom.current());
//                    return new QPolicy(p);
//                }
//        ).replay(new SimpleReplay(8 * 1024, 1 / 3f, 3));
//    }

    public static Agent Random(int inputs, int actions) {
        return new RandomUniformAgent(inputs, actions);
    }
    public static Agent RandomGaussian(int inputs, int actions) {
        return new RandomGaussianAgent(inputs, actions);
    }

    public static Agent REINFORCE(int i, int o) {
        float s = 3; // Scale factor for hidden units
        int h = round(i * s); // Hidden layer size
        int episodeLen = 16; // Default episode length for REINFORCE buffer, can be shorter

        NetworkConfig policyNetworkConfig = new NetworkConfig(
            1e-3f,          // learningRate
            new int[]{h, h}, // hiddenLayerSizes
            Tensor.RELU,    // hiddenActivation
            null,           // outputActivation (GaussianPolicyNet handles its own)
            false,          // useLayerNorm
            null,           // weightInitializer
            true
        );

        OptimizerConfig policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);
        MemoryConfig memoryConfig = new MemoryConfig(episodeLen, null, null);

        HyperparamConfig hyperparamConfig = new HyperparamConfig(
            0.99f,    // gamma
            null,     // lambda (not used)
            1,        // epochs (REINFORCE updates once per episode)
            false,    // normalizeAdvantages (not applicable as no baseline)
            true,     // normalizeReturns (common for REINFORCE)
            0.01f,    // entropyBonus
            1
        );

        ReinforceAgentConfig agentConfig = new ReinforceAgentConfig(
            policyNetworkConfig,
            policyOptimizerConfig,
            memoryConfig,
            hyperparamConfig,
            null // actionConfig (uses defaults)
        );
        return new ReinforceAgent(agentConfig, i, o);
    }

    public static Agent ReinforceDNC(int i, int o) {
        float s = 3; // Scale factor for hidden units from original
        int h = round(i * s);
        int episodeLen = 12; // From original

        // DNC specific parameters from original
        int dncMemorySize = 16;
        int dncMemoryWords = 12;
        int dncReadHeads = 1;
        DNCMemory.EraseMode eraseMode = DNCMemory.EraseMode.SCALAR;

        // Base ReinforceAgentConfig (some parts might be overridden or adapted by ReinforceDNCAgent constructor or DNC policy)
        NetworkConfig policyNetworkConfig = new NetworkConfig(
            1e-3f,
            new int[]{h, h}, // These hidden sizes might be interpreted differently by a DNC setup
            Tensor.RELU,
            null,
            false,
            null,
            true
        );
        OptimizerConfig policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);
        MemoryConfig memoryConfig = new MemoryConfig(episodeLen, null, null);
        HyperparamConfig hyperparamConfig = new HyperparamConfig(
            0.99f, null, 1, false, true, 0.01f, 1
        );
        ReinforceAgentConfig agentConfig = new ReinforceAgentConfig(
            policyNetworkConfig, policyOptimizerConfig, memoryConfig, hyperparamConfig, null
        );

        // This relies on ReinforceDNCAgent correctly setting up its DNC-based policy.
        // The actual DNC model construction is within ReinforceDNCAgent.
        // Note: The interaction between NetworkConfig and a custom DNC model in ReinforceDNCAgent needs careful implementation.
        // The provided ReinforceDNCAgent is a skeleton.
        return new ReinforceDNCAgent(agentConfig, i, o, dncMemorySize, dncMemoryWords, dncReadHeads, eraseMode);
    }

    public static Agent VPG(int i, int o) {
        float s = 2; // Scale factor for hidden units
        int h = round(i * s); // Hidden layer size
        int episodeLen = 32; // Default episode length for VPG buffer

        NetworkConfig sharedNetworkConfig = new NetworkConfig(
            1e-3f,          // learningRate
            new int[]{h, h}, // hiddenLayerSizes
            Tensor.RELU,    // hiddenActivation
            null,           // outputActivation
            false,          // useLayerNorm
            null,           // weightInitializer
            true
        );

        OptimizerConfig policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);
        OptimizerConfig valueOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);

        MemoryConfig memoryConfig = new MemoryConfig(episodeLen, null, null);

        HyperparamConfig hyperparamConfig = new HyperparamConfig(
            0.99f,    // gamma
            null,     // lambda (not used in basic VPG)
            1,        // epochs (VPG updates once per episode typically)
            true,     // normalizeAdvantages
            true,     // normalizeReturns (VPG uses returns directly as targets for value net if GAE not used)
            0.01f,    // entropyBonus
            1
        );

        VPGAgentConfig agentConfig = new VPGAgentConfig(
            sharedNetworkConfig, // policyNetworkConfig
            sharedNetworkConfig, // valueNetworkConfig
            policyOptimizerConfig,
            valueOptimizerConfig,
            memoryConfig,
            hyperparamConfig,
            null // actionConfig (uses defaults)
        );

        return new VPGAgent(agentConfig, i, o);
    }

    public static Agent PPO(int i, int o) {
        var s = 4; // Scale factor for hidden units based on input size
        var episodes = 24; // Default episode length for PPO buffer
        var h = round(i * s); // Hidden layer size

        // Create PPOAgentConfig
        // NetworkConfig for policy and value networks
        NetworkConfig sharedNetworkConfig = new NetworkConfig(
            3e-4f,          // learningRate (can be overridden by optimizer specific LR)
            new int[]{h, h}, // hiddenLayerSizes
            Tensor.RELU,    // hiddenActivation
            null,           // outputActivation (GaussianPolicyNet handles its own, ValueNet typically linear)
            false,          // useLayerNorm
            null,           // weightInitializer
            true            // useBiasInLastLayer for value net, policy net manages its own
        );

        OptimizerConfig policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null,null);
        OptimizerConfig valueOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null); // Often slightly higher LR for value

        // MemoryConfig: PPO is on-policy, uses episodeLength for its buffer
        MemoryConfig memoryConfig = new MemoryConfig(
            episodes, // episodeLength
            null,     // replayBufferConfig (not used by PPOAgent directly)
            null      // prioritizedReplayConfig (not used)
        );

        // HyperparamConfig for PPO
        HyperparamConfig hyperparamConfig = new HyperparamConfig(
            0.99f,    // gamma
            0.95f,    // lambda (for GAE)
            10,       // epochs for PPO update
            true,     // normalizeAdvantages
            false,    // normalizeReturns (less common for PPO, GAE handles much of it)
            0.01f,    // entropyBonus
            1         // numActorCriticSharedLayers - not directly used by PPOAgent like this, network config more direct
        );
        // PPO specific hyperparams
        PPOAgentConfig.PPOHyperparams ppoHyperparams = new PPOAgentConfig.PPOHyperparams(
            0.2f, // ppoClip
            null, // valueLossCoeff (default 0.5)
            null  // maxGradNorm (default null)
        );

        PPOAgentConfig agentConfig = new PPOAgentConfig(
            sharedNetworkConfig, // policyNetworkConfig
            sharedNetworkConfig, // valueNetworkConfig (can be different if needed)
            policyOptimizerConfig,
            valueOptimizerConfig,
            memoryConfig,
            hyperparamConfig,
            ppoHyperparams,
            null // actionConfig (uses defaults: sigmaMin=0.01, sigmaMax=1.0)
        );

        return new PPOAgent(agentConfig, i, o);
    }

    public static Agent StreamAC(int i, int o) {
        var s = 4; // Scale factor from original
        int h = round(i * s);

        // StreamAC is actor-critic, so use VPGAgentConfig as a base.
        // Modifications might be needed in StreamACAgent class for true streaming updates.
        NetworkConfig sharedNetworkConfig = new NetworkConfig(
            1e-3f, new int[]{h, h}, Tensor.RELU, null, false, null, true
        );
        OptimizerConfig policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);
        OptimizerConfig valueOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);

        // If StreamAC updates per step, episodeLength might be 1 or memory handled differently.
        // For now, using a small episode length.
        MemoryConfig memoryConfig = new MemoryConfig(
            1, // episodeLength (or null if StreamACAgent manages memory for single step updates)
            null, null
        );

        HyperparamConfig hyperparamConfig = new HyperparamConfig(
            0.99f,    // gamma
            0.95f,    // lambda (GAE might still be useful if updates are batched over small windows)
            1,        // epochs
            true,     // normalizeAdvantages
            true,     // normalizeReturns
            0.01f,    // entropyBonus
            1
        );

        VPGAgentConfig agentConfig = new VPGAgentConfig(
            sharedNetworkConfig, sharedNetworkConfig,
            policyOptimizerConfig, valueOptimizerConfig,
            memoryConfig, hyperparamConfig, null
        );

        // Relies on StreamACAgent to implement specific streaming behavior.
        return new StreamACAgent(agentConfig, i, o);
    }

    public static Agent SAC(int i, int o) {
        float s = 2; // Scale factor for hidden units, SAC often uses smaller networks like 256x256
        int h = round(i * s);
        if (h < 64) h = 64; // Ensure a minimum size
        if (h > 256) h = 256; // Cap at a reasonable size for typical SAC

        NetworkConfig policyNetworkConfig = new NetworkConfig(
            3e-4f, new int[]{h, h}, Tensor.RELU, null, false, null, true // Output activation handled by GaussianPolicyNet
        );
        NetworkConfig qNetworkConfig = new NetworkConfig(
            3e-4f, new int[]{h, h}, Tensor.RELU, null, false, null, true // Q-value output is linear
        );

        OptimizerConfig policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null,null);
        OptimizerConfig qOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null,null);
        OptimizerConfig alphaOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null,null);


        MemoryConfig.ReplayBufferConfig replayConfig = new MemoryConfig.ReplayBufferConfig(
            1_000_000, // capacity
            256        // batchSize, common for SAC
        );
        MemoryConfig memoryConfig = new MemoryConfig(null, replayConfig, null);

        HyperparamConfig commonHyperparams = new HyperparamConfig(
            0.99f, null, 1, false, false, 0.0f, 1 // gamma
        );

        SACAgentConfig.SACHyperparams sacSpecificHyperparams = new SACAgentConfig.SACHyperparams(
            0.005f, // tau for target Q updates
            0.2f,   // initialAlpha
            true,   // automaticAlphaTuning
            (float) -o, // targetEntropy = -action_dim
            1,      // targetUpdateInterval
            false   // useFixedAlpha
        );

        SACAgentConfig agentConfig = new SACAgentConfig(
            policyNetworkConfig,
            qNetworkConfig,
            policyOptimizerConfig,
            qOptimizerConfig,
            alphaOptimizerConfig,
            memoryConfig,
            commonHyperparams,
            sacSpecificHyperparams
        );

        return new SACAgent(agentConfig, i, o);
    }

    public static Agent DDPG(int i, int o) {
        float s = 4;
        int h = round(i*s);

        //return new DDPG(i, o, h, h).agent(); // Old pg version

        // PGBuilder and DDPGStrategy from pg2 are being replaced by pg3.DDPGAgent
        // var hyperparams = new PGBuilder.HyperparamConfig();
        // var actionConfig = new PGBuilder.ActionConfig();
        // var memoryConfig = new PGBuilder.MemoryConfig(32, // episodeLength for on-policy
        //    new PGBuilder.MemoryConfig.ReplayBufferConfig(1024, 4) // replayBuffer for off-policy
        // );
        // var policyNetConfig = new NetworkConfig(3e-4f, h, h);
        // var valueNetConfig = new NetworkConfig(1e-3f, h, h);
        // return new PolicyGradientModel(i, o, DDPGStrategy.ddpgStrategy(i, o, actionConfig, policyNetConfig, valueNetConfig, memoryConfig, hyperparams)).agent();

        // New pg3 DDPGAgent
        NetworkConfig actorNetworkConfig = new NetworkConfig(
            1e-4f, new int[]{h, h}, Tensor.RELU, Tensor.TANH, false, null, true
        );
        NetworkConfig criticNetworkConfig = new NetworkConfig(
            1e-3f, new int[]{h, h}, Tensor.RELU, null, false, null, true // Critic output is linear
        );

        OptimizerConfig actorOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-4f, null,null);
        OptimizerConfig criticOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);

        MemoryConfig.ReplayBufferConfig replayConfig = new MemoryConfig.ReplayBufferConfig(
            1_000_000, // capacity
            64         // batchSize
        );
        MemoryConfig memoryConfig = new MemoryConfig(null, replayConfig, null);

        HyperparamConfig commonHyperparams = new HyperparamConfig(
            0.99f, null, 1, false, false, 0.0f, 1 // gamma, other PG params less relevant for DDPG directly
        );
        DDPGAgentConfig.DDPGHyperparams ddpgSpecificHyperparams = new DDPGAgentConfig.DDPGHyperparams(
            0.005f, // tau
            0.1f,   // actionNoiseStddev
            0.2f,   // targetPolicyNoise (more for TD3, but can be set)
            0.5f,   // targetNoiseClip (more for TD3)
            1       // policyUpdateFreq
        );

        DDPGAgentConfig agentConfig = new DDPGAgentConfig(
            actorNetworkConfig,
            criticNetworkConfig,
            actorOptimizerConfig,
            criticOptimizerConfig,
            memoryConfig,
            commonHyperparams,
            ddpgSpecificHyperparams
        );

        return new DDPGAgent(agentConfig, i, o);
    }

    public static Agent ReinforceODE(int i, int o) { // NeuralODE variant
        float s = 1; // Scale factor from original
        // The original used round(i * o * s) for hidden size, which can be very large.
        // Let's use a more conventional scaling based on input `i` or a fixed reasonable size.
        int odeHiddenSize = round(i * s * 2); // Adjusted scaling
        if (odeHiddenSize == 0 && i > 0) odeHiddenSize = i; // Ensure non-zero if i is non-zero
        if (odeHiddenSize < 32 && (i+o) > 16) odeHiddenSize = Math.min(128, Math.max(32, (i+o)*2)); // A more robust default hidden size
        else if (odeHiddenSize == 0) odeHiddenSize = 32;


        int odeSolverSteps = 8; // From original
        int episodeLen = 8;     // From original ReinforceODE.ReinforceNeuralODE

        NetworkConfig policyNetworkConfig = new NetworkConfig(
            1e-3f, new int[]{odeHiddenSize}, Tensor.RELU, null, false, null, true
        );
        OptimizerConfig policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null,null);
        MemoryConfig memoryConfig = new MemoryConfig(episodeLen, null, null);
        HyperparamConfig hyperparamConfig = new HyperparamConfig(
            0.99f, null, 1, false, true, 0.01f, 1
        );
        ReinforceAgentConfig agentConfig = new ReinforceAgentConfig(
            policyNetworkConfig, policyOptimizerConfig, memoryConfig, hyperparamConfig, null
        );

        // Relies on ReinforceODEAgent to set up its ODELayer-based policy.
        // The ReinforceODEAgent skeleton needs full implementation.
        return new ReinforceODEAgent(agentConfig, i, o, odeHiddenSize, odeSolverSteps);
    }

    public static Agent ReinforceLiquid(int i, int o) {
        float s = 1;
        return new ReinforceODE.ReinforceLiquid(i, o, round(i * o * s), 2, 10).agent();
    }


    abstract public static class RandomAgent extends Agent {

        final Random rng;

        protected RandomAgent(int inputs, int actions) {
            super(inputs, actions);
            rng = new XoRoShiRo128PlusRandom();
        }

        @Override
        public String toString() {
            return getClass().getSimpleName();
        }

        @Override
        public void apply(double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
            for (int a = 0; a < actionNext.length; a++)
                actionNext[a] = random();
        }

        abstract protected float random();
    }

    private static class RandomGaussianAgent extends RandomAgent {
        RandomGaussianAgent(int inputs, int actions) {
            super(inputs, actions);
        }

        @Override protected float random() {
            return (float) rng.nextGaussian();
        }
    }

    private static class RandomUniformAgent extends RandomAgent {
        RandomUniformAgent(int inputs, int actions) {
            super(inputs, actions);
        }

        @Override protected float random() {
            return rng.nextFloat();
        }
    }
}

