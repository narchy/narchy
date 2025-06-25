package jcog.tensor;

import jcog.agent.Agent;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.model.DNCMemory;
import jcog.tensor.rl.blackbox.BayesZeroPolicy;
import jcog.tensor.rl.blackbox.CMAESZeroPolicy;
import jcog.tensor.rl.blackbox.PopulationPolicy;
import jcog.tensor.rl.dqn.PolicyAgent;
import jcog.tensor.rl.pg.*;
import jcog.tensor.rl.pg2.DDPGStrategy;
import jcog.tensor.rl.pg2.PGBuilder;
import jcog.tensor.rl.pg3.configs.NetworkConfig;

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
        float s = 3;
        return new Reinforce(i, o, round(i * s), 6).agent();
    }

    public static Agent ReinforceDNC(int i, int o) {
        float s = 3;
        return new ReinforceDNC(i, o, round(i * s), 12, 16, 12, 1, DNCMemory.EraseMode.SCALAR).agent();
    }

    public static Agent VPG(int i, int o) {
        float s = 2;
        return new VPG(i, o, round(i * s), round(i * s), 32).agent();
    }

    public static Agent PPO(int i, int o) {
        var s = 4;
        var episodes = 24;
        var h = round(i * s);
        return new PPO(i, o, h, h, episodes).agent();

        //return new PPOAgent(new PPOAgentConfig(), i, o);
    }

    public static Agent StreamAC(int i, int o) {
        var s = 4;
        return new StreamAC(i, o, round(i * s), round(i * s)).agent();
    }

    public static Agent SAC(int i, int o) {
        float s = 1;
        return new SAC(i, o, round(i * o * s), round(i * s), 6).agent();
    }

    public static Agent DDPG(int i, int o) {
        float s = 4;
        int h = round(i*s);

        //return new DDPG(i, o, h, h).agent();


        var hyperparams = new PGBuilder.HyperparamConfig();
        var actionConfig = new PGBuilder.ActionConfig();
        var memoryConfig = new PGBuilder.MemoryConfig(32, // episodeLength for on-policy
            new PGBuilder.MemoryConfig.ReplayBufferConfig(1024, 4) // replayBuffer for off-policy
        );
        var policyNetConfig = new NetworkConfig(3e-4f, h, h);
        var valueNetConfig = new NetworkConfig(1e-3f, h, h);
        return new PolicyGradientModel(i, o, DDPGStrategy.ddpgStrategy(i, o, actionConfig, policyNetConfig, valueNetConfig, memoryConfig, hyperparams)).agent();
    }

    public static Agent ReinforceODE(int i, int o) {
        float s = 1;
        return new ReinforceODE.ReinforceNeuralODE(i, o, round(i * o * s), 8, 8).agent();
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

