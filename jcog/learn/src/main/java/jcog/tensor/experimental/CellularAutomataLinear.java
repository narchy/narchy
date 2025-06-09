package jcog.tensor.experimental;

import jcog.Util;
import jcog.activation.ReluActivation;
import jcog.activation.SigmoidActivation;
import jcog.math.FloatSupplier;
import jcog.tensor.Tensor;
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 TODO
     Input/Output layout options
     Soft Addressing for Operands: You could extend the idea of differentiable selection to operands by using an attention mechanism, as described in the previous response.
     More Complex Instructions: You could introduce more complex parameterized instructions (e.g., with learnable weights or even small neural networks within each instruction).
     Curriculum Learning: Start with a high temperature and/or simple tasks and gradually decrease the temperature and increase task complexity.
     Advanced Optimizers: Use optimizers like Adam or RMSprop to potentially improve convergence.
     LSTM/GRU-like Mechanisms: Consider adding recurrent connections or memory mechanisms to help with long-range dependencies.
     Exploration of Hyperparameters: There are now more hyperparameters to tune (learning rate, temperature, annealing rate, initial ampInit). Careful experimentation and validation are essential.
*/
public class CellularAutomataLinear implements UnaryOperator<Tensor> {
    public static final String SELF = "SELF", REG = "REG";
    private static final String[] OPCODES = {"ADD", "MUL", "SIG", "RELU", "LERP"};
    private static final int NUM_OPCODES = OPCODES.length;
    private static final String[] OPERANDS = {"N", "S", "E", "W", SELF, REG, "CONST"};

    private final int rows, cols;
    private final int timeSteps;
    private final int numInstructions, numRegisters;

    // Instructions: Each cell has a list of instructions.
    // Now, instructions are represented by their parameters.
    private final InstructionParameters[][][] instructionParams;

    // Current cell values (parameters)
    public final double[][] cells;

    // Cell gradients
    public final double[][] cellGrads;

    // Register values and gradients
    public final double[][][] registers;
    public final double[][][] registerGrads;

    // For forward propagation through time, store all states
    private final double[][][] allStates;


    final int ins, outs;

    public final FloatSupplier learningRate;

    /** Temperature for Gumbel-Softmax */
    public double temperature =
        0.9f;
        //0.75f;
        //0.5f;
        //0.25f;

    public final FloatSupplier temperatureAnnealingRate; // e.g., () -> 0.99

    private boolean gradClamp = false, gradClampIn = false, gradNorm = false;

    /**
     * clamp range for instruction results
     */
    private static final double cellClamp =
            //0.5f;
            //2;
            Util.PHI;

    /** reapplies the input to the cells each iteration of the update
     *  TODO implement this by disallowing cell writes to input cells completely
     */
    private boolean inputLock = true;

    private boolean biasEnabled = true;

    public final @Nullable Tensor bias;

    public CellularAutomataLinear(int ins, int outs, int hiddens, int timeSteps, int numInstructions, FloatSupplier learningRate) {
        this(ins, outs, hiddens, timeSteps, numInstructions, 1, learningRate, ()->1);
    }

    /**
     * @param ins                   number of input cells
     * @param outs                  number of output cells
     * @param hiddens               number of hidden rows (besides input and output rows)
     * @param timeSteps             number of time steps to run the CA
     * @param numInstructions       number of instructions per cell
     * @param numRegisters          number of register slots per cell
     * @param learningRate          learning rate supplier for gradient updates
     * @param temperatureAnnealingRate  annealing rate for Gumbel-Softmax temperature
     */
    public CellularAutomataLinear(int ins, int outs, int hiddens, int timeSteps, int numInstructions, int numRegisters, FloatSupplier learningRate, FloatSupplier temperatureAnnealingRate) {
        this.ins = ins;
        this.outs = outs;
        this.learningRate = learningRate;
        this.temperatureAnnealingRate = temperatureAnnealingRate;

        // Total rows: first row is input, last row is output, hiddens in between
        this.rows = hiddens + 2;
        this.cols = Math.max(ins, outs);

        this.timeSteps = timeSteps;
        this.numInstructions = numInstructions;
        this.numRegisters = numRegisters;

        double ampInit = 0.1; // Reduced to improve stability

        // Initialize cell parameters randomly
        cells = new double[rows][cols];
        cellGrads = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cells[i][j] = (ThreadLocalRandom.current().nextDouble() * 2 - 1) * ampInit;
            }
        }

        instructionParams = new InstructionParameters[rows][cols][numInstructions];
        registers = new double[rows][cols][numRegisters];
        registerGrads = new double[rows][cols][numRegisters];

        var rng = ThreadLocalRandom.current();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                for (int k = 0; k < this.numInstructions; k++) {
                    // Initialize instruction parameters
                    instructionParams[i][j][k] = new InstructionParameters(
                            randomOperand(i, j, rng),
                            randomOperand(i, j, rng),
                            rng.nextBoolean() ? SELF : REG
                    );
                }
            }
        }

        // We'll store states for BPTT
        allStates = new double[timeSteps + 1][rows][cols];

        this.bias = biasEnabled ? new Tensor(1, outs, true).parameter() : null;
    }

    private @NotNull String randomOperand(int i, int j, RandomGenerator rng) {
        var o = OPERANDS[rng.nextInt(OPERANDS.length)];

        // HACK avoid left<->right interactions on first and last rows
        if (i == 0 && (o.equals("W") || o.equals("E") || o.equals("N"))) o = "S";
        if (i == rows - 1 && (o.equals("W") || o.equals("E") || o.equals("S"))) o = "N";

        // Avoid out-of-bounds neighbor references by mirroring
//        if (o.equals("W") && j == 0) o = "E";
//        if (o.equals("E") && j == cols - 1) o = "W";
//        if (o.equals("N") && i == 0) o = "S";
//        if (o.equals("S") && i == rows - 1) o = "N";
        return o;
    }

    private double argValue(int cellRow, int cellCol, String operand, int regIndex, double[][] cellState) {
        return switch (operand) {
            case "N" -> cellRow > 0 ? cellState[cellRow - 1][cellCol] : cellState[0][cellCol];
            case "S" -> cellRow < rows - 1 ? cellState[cellRow + 1][cellCol] : cellState[rows - 1][cellCol];
            case "E" -> cellCol < cols - 1 ? cellState[cellRow][cellCol + 1] : cellState[cellRow][cols - 1];
            case "W" -> cellCol > 0 ? cellState[cellRow][cellCol - 1] : cellState[cellRow][0];
            case SELF -> cellState[cellRow][cellCol];
            case REG -> registers[cellRow][cellCol][regIndex];
            case "CONST" -> 0.5;
            default -> 0;
        };
    }

    // Computes the weighted result of an instruction based on Gumbel-Softmax probabilities
    private double computeInst(int r, int c, InstructionParameters inst, double[][] cellIn) {
        double[] results = new double[NUM_OPCODES];
        for (int opIndex = 0; opIndex < NUM_OPCODES; opIndex++) {
            String op = OPCODES[opIndex];
            var x1 = argValue(r, c, inst.a, 0, cellIn);
            var x2 = argValue(r, c, inst.b, 0, cellIn);

            results[opIndex] = switch (op) {
                case "ADD" -> x1 + x2;
                case "MUL" -> x1 * x2;
                case "SIG" -> Util.sigmoid(x1);
                case "RELU" -> Math.max(0, x1);
                case "LERP" -> Util.lerp(0.5, x1, x2);
                default -> 0;
            };
            results[opIndex] = Double.isFinite(results[opIndex]) ? results[opIndex] : 0;
        }

        // Compute weighted sum based on instruction probabilities
        double weightedResult = 0;
        for (int i = 0; i < NUM_OPCODES; i++) {
            weightedResult += results[i] * inst.probs[i];
        }
        return Util.clampSafePolar(weightedResult, cellClamp);
    }

    private void exeInst(int r, int c, InstructionParameters inst, double[][] cellIn, double[][] cellOut) {
        // Sample instruction using Gumbel-Softmax
        inst.updateProbs(temperature);

        double val = computeInst(r, c, inst, cellIn);
        if (inst.out.equals(SELF)) {
            cellOut[r][c] = val;
        } else { // REG
            registers[r][c][0] = val;
        }
    }

    /**
     * Run the CA forward for 'timeSteps' steps starting from 'x'. Returns final state.
     */
    private double[][] forwardCA(Tensor input) {
        var x = cells;
        Tensor.zero(registers);
        Tensor.zero(registerGrads);

        int ic = input.cols();
        if (ic != ins)
            throw new IllegalArgumentException("Input size mismatch.");


        setInput(input, x);

        // Copy initial state
        double[][] cur = new double[rows][cols];
        copy(x, cur);
        copy(cur, allStates[0]);

        double[][] tmp = new double[rows][cols];

        for (int t = 0; t < timeSteps; t++) {
            copy(cur, tmp);

            if (inputLock)
                setInput(input, cur);

            for (int rr = 0; rr < rows; rr++) {
                for (int cc = 0; cc < cols; cc++) {
                    // Execute instructions on a copy of state 'cur' to produce 'tmp'
                    for (var inst : instructionParams[rr][cc]) {
                        exeInst(rr, cc, inst, cur, tmp);
                    }
                }
            }
            copy(tmp, cur);
            copy(cur, allStates[t + 1]); // store for BPTT
        }

        return cur;
    }

    private void copy(double[][] from, double[][] to) {
        for (int i = 0; i < rows; i++)
            System.arraycopy(from[i], 0, to[i], 0, cols);
    }

    /**
     * Backpropagation Through Time
     */
    private synchronized void bptt(SimpleMatrix gradOutput) {
        Tensor.zero(cellGrads);
        Tensor.zero(registerGrads);

        // Gradients w.r.t the final output
        for (int i = 0; i < outs; i++) {
            var gi = gradOutput.get(i);
            if (gradClampIn)
                gi = Util.clampSafePolar(gi, 1);
            cellGrads[rows - 1][cols - outs + i] = gi;
        }

        // Backprop through time:
        for (int t = timeSteps; t > 0; t--) {
            double[][] curCells = allStates[t];
            double[][] prevCells = allStates[t - 1];

            for (int r = rows - 1; r >= 0; r--) {
                for (int c = cols - 1; c >= 0; c--) {
                    double outGradSelf = cellGrads[r][c];
                    double outGradReg = registerGrads[r][c][0];

                    for (int instIndex = instructionParams[r][c].length - 1; instIndex >= 0; instIndex--) {
                        var inst = instructionParams[r][c][instIndex];
                        double gradTarget = inst.out.equals(SELF) ? outGradSelf : outGradReg;
                        if (gradTarget == 0) continue;

                        // --- Differentiable Instruction Selection Backpropagation ---

                        // 1. Calculate the gradient of the output with respect to the weighted sum.
                        //    This is just the gradTarget, as d(output)/d(weightedResult) = 1.

                        // 2. Calculate the gradient of the weighted sum with respect to each instruction's result.
                        //    This is simply the probability of that instruction: d(weightedResult)/d(result_i) = prob_i
                        double[] dWeightedResult_dResult = Arrays.copyOf(inst.probs, NUM_OPCODES);

                        // 3. Calculate the gradient of each instruction's result with respect to its operands.
                        double[][] dResult_dOperand = new double[NUM_OPCODES][2]; // 2 operands: a and b
                        for (int opIndex = 0; opIndex < NUM_OPCODES; opIndex++) {
                            double aVal = argValue(r, c, inst.a, 0, prevCells);
                            double bVal = argValue(r, c, inst.b, 0, prevCells);

                            switch (OPCODES[opIndex]) {
                                case "ADD":
                                    dResult_dOperand[opIndex][0] = 1;
                                    dResult_dOperand[opIndex][1] = 1;
                                    break;
                                case "MUL":
                                    dResult_dOperand[opIndex][0] = bVal;
                                    dResult_dOperand[opIndex][1] = aVal;
                                    break;
                                case "SIG":
                                    dResult_dOperand[opIndex][0] = SigmoidActivation.the.derivative(aVal);
                                    dResult_dOperand[opIndex][1] = 0;
                                    break;
                                case "RELU":
                                    dResult_dOperand[opIndex][0] = ReluActivation.the.derivative(aVal);
                                    dResult_dOperand[opIndex][1] = 0;
                                    break;
                                case "LERP":
                                    dResult_dOperand[opIndex][0] = 0.5;
                                    dResult_dOperand[opIndex][1] = 0.5;
                                    break;
                            }
                        }

                        // 4. Calculate the gradient of the weighted sum with respect to the instruction logits.
                        // We use the Gumbel-Softmax gradient estimator here. The gradient of the
                        // probabilities with respect to the logits is approximated during the
                        // forward pass using the Gumbel-Softmax trick.
                        // For details, refer to the Gumbel-Softmax paper or বোঝার জন্য
                        // https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html
                        // In this implementation, we assume that the temperature is fixed during
                        // each forward pass. Therefore, the gradient calculation is simplified.

                        // Now, we can calculate the gradient of the loss with respect to the logits.
                        double[] dLoss_dLogits = new double[NUM_OPCODES];
                        for (int opIndex = 0; opIndex < NUM_OPCODES; opIndex++) {
                            double expectedValueChange = 0;
                            for (int otherOpIndex = 0; otherOpIndex < NUM_OPCODES; otherOpIndex++) {
                                // Estimate the expected change in value if we had picked otherOpIndex instead of opIndex
                                double valueDiff = 0;
                                if (opIndex != otherOpIndex) {
                                    // Approximate the value difference; ideally, this would be the difference
                                    // between executing opIndex vs. otherOpIndex given the current state.
                                    // For simplicity, we use the difference in their immediate results.
                                    var x1 = argValue(r, c, inst.a, 0, prevCells);
                                    var x2 = argValue(r, c, inst.b, 0, prevCells);

                                    double resultA = switch (OPCODES[opIndex]) {
                                        case "ADD" -> x1 + x2;
                                        case "MUL" -> x1 * x2;
                                        case "SIG" -> Util.sigmoid(x1);
                                        case "RELU" -> Math.max(0, x1);
                                        case "LERP" -> Util.lerp(0.5, x1, x2);
                                        default -> 0;
                                    };
                                    resultA = Double.isFinite(resultA) ? Util.clampSafePolar(resultA, cellClamp) : 0;

                                    double resultB = switch (OPCODES[otherOpIndex]) {
                                        case "ADD" -> x1 + x2;
                                        case "MUL" -> x1 * x2;
                                        case "SIG" -> Util.sigmoid(x1);
                                        case "RELU" -> Math.max(0, x1);
                                        case "LERP" -> Util.lerp(0.5, x1, x2);
                                        default -> 0;
                                    };
                                    resultB = Double.isFinite(resultB) ? Util.clampSafePolar(resultB, cellClamp) : 0;

                                    valueDiff = resultB - resultA;
                                }
                                // The expected change in loss is the value difference scaled by the probability
                                // of selecting the other operation.
                                expectedValueChange += inst.probs[otherOpIndex] * valueDiff;
                            }
                            // The gradient with respect to the logit is the difference between the actual
                            // result and the expected result, scaled by the gradient target.
                            dLoss_dLogits[opIndex] = gradTarget * (dWeightedResult_dResult[opIndex] - expectedValueChange);

                            // Update the logits (instruction parameters) using the computed gradient
                            inst.dLogits[opIndex] += dLoss_dLogits[opIndex]; // Accumulate gradients over time
                        }

                        // 5. Propagate gradients to operands:
                        for (int opIndex = 0; opIndex < NUM_OPCODES; opIndex++) {
                            double dLoss_dResult_i = gradTarget * dWeightedResult_dResult[opIndex];
                            propagateGrad(inst.a, r, c, dLoss_dResult_i * dResult_dOperand[opIndex][0], 0);
                            propagateGrad(inst.b, r, c, dLoss_dResult_i * dResult_dOperand[opIndex][1], 0);
                        }
                        // --- End of Differentiable Instruction Selection Backpropagation ---

                        // After this instruction was "applied", the output gradient
                        // has been accounted for, so set it to zero if SELF (since we've handled it)
                        if (inst.out.equals(SELF)) {
                            outGradSelf = 0; // gradient passed back
                            cellGrads[r][c] = 0; // consumed by this instruction
                        } else {
                            outGradReg = 0;
                            registerGrads[r][c][0] = 0; // consumed
                        }
                    }
                }
            }
        }

        if (gradClamp)
            Util.clamp(cellGrads, -1, +1);
        if (gradNorm) {
            var norm = Math.sqrt(Util.sumSqr(cellGrads));
            if (norm>1)
                Tensor.mult(cellGrads, 1 / norm);
        }

        // Update cell parameters and instruction logits
        float lr = learningRate.asFloat();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cells[i][j] = Util.clampSafePolar(cells[i][j] + cellGrads[i][j] * lr, cellClamp);
                for (int k = 0; k < numInstructions; k++) {
                    for (int opIndex = 0; opIndex < NUM_OPCODES; opIndex++) {
                        // Directly update logits using accumulated gradients
                        instructionParams[i][j][k].logits[opIndex] += instructionParams[i][j][k].dLogits[opIndex] * lr;
                        instructionParams[i][j][k].dLogits[opIndex] = 0; // Reset
                    }
                }
            }
        }

        // Anneal temperature
        temperature *= temperatureAnnealingRate.asFloat();
    }

    private void propagateGrad(String operand, int r, int c, double dx, int regIndex) {
        if (dx == 0) return;
        switch (operand) {
            case SELF -> cellGrads[r][c] += dx;
            case "N" -> {
                if (r > 0) cellGrads[r - 1][c] += dx;
                else cellGrads[0][c] += dx;
            }
            case "S" -> {
                if (r < rows - 1) cellGrads[r + 1][c] += dx;
                else cellGrads[rows - 1][c] += dx;
            }
            case "E" -> {
                if (c < cols - 1) cellGrads[r][c + 1] += dx;
                else cellGrads[r][cols - 1] += dx;
            }
            case "W" -> {
                if (c > 0) cellGrads[r][c - 1] += dx;
                else cellGrads[r][0] += dx;
            }
            case REG -> registerGrads[r][c][regIndex] += dx;
            case "CONST" -> {
            } // No gradient to constants
            default -> throw new UnsupportedOperationException("Unknown operand: " + operand);
        }
    }

    @Override
    public Tensor apply(Tensor input) {

        // Forward pass
        var newCells = forwardCA(input);

        // Construct output tensor
        var output = new Tensor(1, outs, true);
        for (int i = 0; i < outs; i++)
            output.data.set(i, newCells[rows - 1][cols - outs + i]);

        if (output.hasGrad()) {
            output.op = new Tensor.TensorOp(input) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    bptt(grad);
                    if (gradOut[0] != null) {
                        int ic = input.cols();
                        for (int i = 0; i < ic; i++)
                            gradOut[0].set(i, cellGrads[0][i]);
                    }
                }
            };
        }

        return bias==null ? output : output.add(bias);
    }

    private static void setInput(Tensor input, double[][] cells) {
        int ic = input.cols();
        // Place input into top row of cells
        for (int i = 0; i < ic; i++)
            cells[0][i] = input.data(i);
    }

    public double[][] getCells() {
        return cells;
    }

    /**
     * Represents the parameters of an instruction, including:
     * - logits: The unnormalized probabilities for each opcode.
     * - probs: The probabilities for each opcode after applying Gumbel-Softmax.
     * - a, b: The operands.
     * - out: The output target (SELF or REG).
     */
    public class InstructionParameters {
        public final double[] logits;
        public final double[] dLogits;
        public final double[] probs;
        public final String a, b, out;

        public InstructionParameters(String a, String b, String out) {
            this.logits = new double[NUM_OPCODES];
            this.dLogits = new double[NUM_OPCODES]; // To accumulate gradients for logits
            this.probs = new double[NUM_OPCODES];
            this.a = a;
            this.b = b;
            this.out = out;

            // Initialize logits randomly (small values around 0)
            for (int i = 0; i < NUM_OPCODES; i++) {
                this.logits[i] = ThreadLocalRandom.current().nextGaussian() * 0.1;
            }
        }

        // Updates the probabilities using the Gumbel-Softmax trick
        public void updateProbs(double temperature) {
            double[] gumbelNoise = new double[NUM_OPCODES];
            for (int i = 0; i < NUM_OPCODES; i++) {
                gumbelNoise[i] = -Math.log(-Math.log(ThreadLocalRandom.current().nextDouble()));
            }

            double[] logitsWithNoise = new double[NUM_OPCODES];
            for (int i = 0; i < NUM_OPCODES; i++) {
                logitsWithNoise[i] = (logits[i] + gumbelNoise[i]) / temperature;
            }

            // Softmax
            double sum = 0;
            for (int i = 0; i < NUM_OPCODES; i++) {
                probs[i] = Math.exp(logitsWithNoise[i]);
                sum += probs[i];
            }
            for (int i = 0; i < NUM_OPCODES; i++) {
                probs[i] /= sum;
            }
        }

        @Override
        public String toString() {
            return String.format("Instruction(logits=%s, a=%s, b=%s, out=%s)", Arrays.toString(logits), a, b, out);
        }
    }

    // Example usage:
    public static void main(String[] args) {
        int ins = 3;
        int outs = 2;
        int hiddens = 4;
        int timeSteps = 5;
        int numInstructions = 3;
        int numRegisters = 2;
        FloatSupplier lr = () -> 0.005f; // reduced learning rate
        FloatSupplier tempAnnealingRate = () -> 0.95f;

        var ca = new CellularAutomataLinear(ins, outs, hiddens, timeSteps, numInstructions, numRegisters, lr, tempAnnealingRate);

        Tensor input = new Tensor(1, ins, true);
        input.data.set(0, 0.1);
        input.data.set(1, 0.2);
        input.data.set(2, 0.3);

        // A simple "training" loop
        for (int epoch = 0; epoch < 200; epoch++) {
            Tensor output = ca.apply(input);
            System.out.printf("Epoch: %d, Output: ", epoch);
            output.data.print();

            // Example target output (you would typically compute a loss here)
            SimpleMatrix target = new SimpleMatrix(1, outs);
            target.set(0, 0.8);
            target.set(1, 0.2);

            // Compute a simple mean squared error loss and its gradient
            SimpleMatrix lossGrad = output.data.minus(target).scale(2.0 / outs);

            // Perform backpropagation
            output.op.backward(lossGrad, new SimpleMatrix[]{new SimpleMatrix(1, ins)});

            // Print some instruction probabilities to observe changes
            if (epoch % 10 == 0) {
                System.out.println("Instruction probabilities (cell[1][0], inst 0): " + Arrays.toString(ca.instructionParams[1][0][0].probs));
                System.out.println("Temperature: " + ca.temperature);
            }
        }

        System.out.println("Cells after training:");
        double[][] updated = ca.getCells();
        for (int i = 0; i < updated.length; i++) {
            for (int j = 0; j < updated[i].length; j++)
                System.out.printf("%.4f ", updated[i][j]);
            System.out.println();
        }
    }
}