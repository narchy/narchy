package jcog.tensor.experimental;

import jcog.random.XoRoShiRo128PlusRandom;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.random.RandomGenerator;

public class AdaptiveCacheAwareMemoryLayer {

    private final RandomGenerator random = new XoRoShiRo128PlusRandom();
    private final float weightClamp;
    private final float gradClamp;
    private final int numHashes;
    private final int chunkSize;
    private final int valueSize;
    private final int tableSize;
    private final Hasher hasher;
    private final Interpolator interpolator;
    private final EvictionPolicy evictionPolicy;
    private final Chunk[] table;
    private final float[][][] interpolationWeights;
    private final float[] defaultValue; // Learnable default value

    public AdaptiveCacheAwareMemoryLayer(int numHashes, int chunkSize, int valueSize, int tableSize,
                                         Hasher hasher, Interpolator interpolator, EvictionPolicy evictionPolicy,
                                         float weightClamp, float gradClamp) {
        this.numHashes = numHashes;
        this.chunkSize = chunkSize;
        this.valueSize = valueSize;
        this.tableSize = tableSize;
        this.hasher = hasher;
        this.interpolator = interpolator;
        this.evictionPolicy = evictionPolicy;
        this.weightClamp = weightClamp;
        this.gradClamp = gradClamp;

        this.table = new Chunk[tableSize];
        for (int i = 0; i < tableSize; i++) {
            this.table[i] = new Chunk();
        }

        this.interpolationWeights = new float[tableSize][numHashes][valueSize];
        for (int i = 0; i < tableSize; i++)
            for (int j = 0; j < numHashes; j++)
                for (int k = 0; k < valueSize; k++)
                    this.interpolationWeights[i][j][k] = (float) ((random.nextDouble() - 0.5) * 0.01); // Initialize to small random values

        this.defaultValue = new float[valueSize];
        for (int i = 0; i < valueSize; i++)
            this.defaultValue[i] = (float) ((random.nextDouble() - 0.5) * 0.01);
    }

    // Data Structure to hold key-value pairs within a chunk
    private class Chunk {
        int[] keys = new int[chunkSize];
        float[][] values = new float[chunkSize][valueSize];
        int count = 0; // Number of entries currently in use

        boolean isFull() {
            return count == chunkSize;
        }
    }

    // Functional Interfaces for Strategy Patterns
    interface Hasher {
        int[] hash(float[] input);
    }

    interface Interpolator {
        float[] interpolate(float[][] values, float[] weights, float[] defaultValue);
    }

    interface EvictionPolicy {
        void access(int chunkIndex);
        int evict();
    }

    public float[] forward(float[] input) {
        int[] hashes = hasher.hash(input);
        float[][] retrievedValues = new float[numHashes][];
        float[] weights = new float[numHashes];

        for (int i = 0; i < numHashes; i++) {
            var hi = hashes[i];

            int chunkIndex = hashToIndex(hi);
            evictionPolicy.access(chunkIndex);

            Chunk chunk = table[chunkIndex];
            int entryIndex = findEntry(chunk, hi);

            if (entryIndex != -1) {
                retrievedValues[i] = chunk.values[entryIndex];
                System.arraycopy(interpolationWeights[chunkIndex][i], 0, weights, i, 1); // Copy single weight
            } else {
                retrievedValues[i] = defaultValue; // Use learnable default value
                weights[i] = 0; // No weight for default value in this case
            }
        }

        return interpolator.interpolate(retrievedValues, weights, defaultValue);
    }

    public void backward(float[] input, float[] gradOut) {
        float[] clampedGradOut = new float[gradOut.length];
        clamp(gradOut, clampedGradOut, -gradClamp, gradClamp);

        int[] hashes = hasher.hash(input);
        for (int i = 0; i < numHashes; i++) {
            int chunkIndex = hashToIndex(hashes[i]);
            Chunk chunk = table[chunkIndex];
            int entryIndex = findEntry(chunk, hashes[i]);

            if (entryIndex != -1) {
                // Update existing entry
                for (int j = 0; j < valueSize; j++) {
                    chunk.values[entryIndex][j] += clampedGradOut[j];
                }
            } else {
                // Allocate a new entry
                if (chunk.isFull()) {
                    chunkIndex = evictionPolicy.evict();
                    chunk = table[chunkIndex];
                    chunk.count = 0; // Reset the count of the evicted chunk
                }
                entryIndex = chunk.count;
                chunk.keys[entryIndex] = hashes[i];

                // Initialize with small random values
                for (int j = 0; j < valueSize; j++) {
                    chunk.values[entryIndex][j] = (float) ((random.nextDouble() - 0.5) * 0.01);
                }

                // Update the newly initialized values
                for (int j = 0; j < valueSize; j++) {
                    chunk.values[entryIndex][j] += clampedGradOut[j];
                }

                chunk.count++;
            }

            // Update interpolation weights
            for (int j = 0; j < valueSize; j++) {
                interpolationWeights[chunkIndex][i][j] =
                        clamp(interpolationWeights[chunkIndex][i][j] + clampedGradOut[j], -weightClamp, weightClamp);
            }
        }

        // Update the default value
        for (int j = 0; j < valueSize; j++) {
            defaultValue[j] += clampedGradOut[j];
        }
    }

    private int hashToIndex(int hash) {
        return Math.abs(hash) % tableSize;
    }

    private int findEntry(Chunk chunk, int hash) {
        for (int i = 0; i < chunk.count; i++) {
            if (chunk.keys[i] == hash) {
                return i;
            }
        }
        return -1;
    }

    // Utility Functions
    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void clamp(float[] values, float[] out, float min, float max) {
        for (int i = 0; i < values.length; i++) {
            out[i] = clamp(values[i], min, max);
        }
    }

    // Example Concrete Implementations

    public static class SimpleHasher implements Hasher {
        @Override
        public int[] hash(float[] input) {
            int hash1 = 17;
            int hash2 = 31;
            for (float v : input) {
                int floatBits = Float.floatToIntBits(v);
                hash1 = hash1 * 31 + floatBits;
                hash2 = hash2 * 17 + floatBits;
            }
            return new int[]{hash1, hash2};
        }
    }

    public static class LinearInterpolator implements Interpolator {
        @Override
        public float[] interpolate(float[][] values, float[] weights, float[] defaultValue) {
            float[] y = new float[values[0].length];
            float weightSum = 0;
            for (float w : weights)
                weightSum += w;

            // Add a small epsilon to avoid division by zero if weightSum is very small.
            weightSum += 1e-10f;

            for (int i = 0; i < values.length; i++) {
                float normalizedWeight = weights[i] / weightSum;
                for (int j = 0; j < values[i].length; j++) {
                    y[j] += values[i][j] * normalizedWeight;
                }
            }

            // Blend with default value based on the sum of weights
            float defaultWeight = Math.max(0, 1 - weightSum);
            for (int j = 0; j < y.length; j++) {
                y[j] += defaultValue[j] * defaultWeight;
            }

            return y;
        }
    }

    public static class LRUEvictionPolicy implements EvictionPolicy {
        private final int[] accessOrder;
        private int count; // Current number of elements in the cache
        private final int tableSize;

        public LRUEvictionPolicy(int tableSize) {
            this.tableSize = tableSize;
            this.accessOrder = new int[tableSize];
            this.count = 0;
        }

        @Override
        public void access(int chunkIndex) {
            // Find the chunk index in the access order
            int index = -1;
            for (int i = 0; i < count; i++) {
                if (accessOrder[i] == chunkIndex) {
                    index = i;
                    break;
                }
            }

            // If found, remove it from its current position
            if (index != -1) {
                for (int i = index; i < count - 1; i++) {
                    accessOrder[i] = accessOrder[i + 1];
                }
                count--;
            }

            // Add the accessed index to the end (most recently used)
            // Ensure count does not exceed tableSize
            if (count < tableSize) {
                accessOrder[count++] = chunkIndex;
            } else {
                // If count is equal to tableSize, replace the last element
                accessOrder[tableSize - 1] = chunkIndex;
            }
        }

        @Override
        public int evict() {
            if (count == 0) {
                return -1; // Or handle this case appropriately, maybe throw an exception
            }

            // Evict the least recently used (first element)
            int lruChunk = accessOrder[0];
            for (int i = 0; i < count - 1; i++) {
                accessOrder[i] = accessOrder[i + 1];
            }
            count--;
            return lruChunk;
        }
    }


//    // Main method for demonstration
//    public static void main(String[] args) {
//        // Example usage:
//        AdaptiveCacheAwareMemoryLayer layer = new AdaptiveCacheAwareMemoryLayer(
//                2,  // numHashes
//                4,  // chunkSize
//                3,  // valueSize
//                5,  // tableSize
//                new SimpleHasher(),
//                new LinearInterpolator(),
//                new LRUEvictionPolicy(5),
//                1.0f,
//                0.1f
//        );
//
//        // Generate dummy data
//        float[][] inputs = {
//                {1.0f, 2.0f, 3.0f},
//                {2.5f, 1.5f, 0.5f},
//                {-1.0f, -2.0f, -3.0f},
//                {0.0f, 1.0f, -1.0f},
//                {0.5f, -0.5f, 1.5f}
//        };
//
//        float[][] targets = {
//                {2.0f, 4.0f, 6.0f},
//                {3.5f, 2.5f, 1.5f},
//                {-2.0f, -4.0f, -6.0f},
//                {1.0f, 2.0f, 0.0f},
//                {1.5f, -1.5f, 2.5f}
//        };
//
//        // Training loop
//        for (int epoch = 0; epoch < 1000; epoch++) {
//            float totalLoss = 0;
//            for (int i = 0; i < inputs.length; i++) {
//                float[] input = inputs[i];
//                float[] target = targets[i];
//
//                float[] output = layer.forward(input);
//                float[] gradOut = new float[output.length];
//                for (int j = 0; j < output.length; j++) {
//                    gradOut[j] = output[j] - target[j]; // Simple error calculation
//                    totalLoss += Math.abs(gradOut[j]);
//                }
//
//                layer.backward(input, gradOut);
//            }
//
//            if ((epoch + 1) % 100 == 0)
//                System.out.println("Epoch " + (epoch + 1) + ", Loss: " + totalLoss / inputs.length);
//        }
//
//        // Test after training
//        System.out.println("\nTest after training:");
//        for (float[] input : inputs) {
//            float[] output = layer.forward(input);
//            System.out.println("Input: " + Arrays.toString(input) + ", Output: " + Arrays.toString(output));
//        }
//    }
    /**
     TODO
        Ensure Animates and displays realtime data
        Animate parameter values as color-coded matrix
        Higher-Dimensional Visualization: For output dimensions greater than 2, consider using dimensionality reduction techniques (like PCA or t-SNE) to project the data onto a 2D plane for visualization.
        More Hyperparameters: Add controls for other hyperparameters like numHashes, chunkSize, and valueSize.
        Performance Metrics: Display quantitative performance metrics like Mean Squared Error (MSE) in the ParameterPanel.
        Error Visualization: Color the error lines based on the magnitude of the error (e.g., red for large errors, green for small errors).
        Customizable Data Generation: Allow users to define their own data generation functions or load data from files.
        Advanced Interpolation: Implement and visualize different interpolation strategies.
        Save/Load Model: Add functionality to save and load trained models.
        More Complex Test Functions: Add more test functions with varying shapes and behaviors to stress-test the system and expose its failure modes.
     */
    static class ModelEvaluator extends JFrame {

        // Model Parameters (Adjustable)
        private int chunkSize = 64;
        private int tableSize = 16;
        private double learningRate = 0.01;


        private int numHashes = 2;

        // Dataset Parameters
        private int datasetSize = 10;
        private int inputSize = 8;
        private int outputSize = 2;
        private int valueSize = outputSize;
        private int complexity = 1; // 1: Simple, 2: Medium, 3: Complex

        // Model
        private AdaptiveCacheAwareMemoryLayer acml;

        // Dataset
        private java.util.List<DataPoint> trainingData;
        private java.util.List<DataPoint> testingData;

        // Visualization
        private DataPanel dataPanel;
        private ParameterPanel parameterPanel;
        private LossPanel lossPanel;

        // Random Number Generator
        private RandomGenerator random = new XoRoShiRo128PlusRandom();

        // Synthetic Data Generation Strategies
        interface DataGenerator {
            DataPoint generate(int inputSize, int outputSize);
        }

        class SimpleDataGenerator implements DataGenerator {
            @Override
            public DataPoint generate(int inputSize, int outputSize) {
                var input = new float[inputSize];
                var output = new float[outputSize];
                for (var i = 0; i < inputSize; i++) {
                    input[i] = random.nextFloat() * 2 - 1; // Range: -1 to 1
                }
                // Simple linear relationship
                output[0] = input[0] + input[1];
                output[1] = input[0] - input[1];
                return new DataPoint(input, output);
            }
        }

        class MediumDataGenerator implements DataGenerator {
            @Override
            public DataPoint generate(int inputSize, int outputSize) {
                var input = new float[inputSize];
                var output = new float[outputSize];
                for (var i = 0; i < inputSize; i++) {
                    input[i] = random.nextFloat() * 2 - 1;
                }
                // Non-linear relationship (polynomial)
                output[0] = input[0] * input[0] + input[1];
                output[1] = input[0] - input[1] * input[1];
                return new DataPoint(input, output);
            }
        }

        class ComplexDataGenerator implements DataGenerator {
            @Override
            public DataPoint generate(int inputSize, int outputSize) {
                var input = new float[inputSize];
                var output = new float[outputSize];
                for (var i = 0; i < inputSize; i++) {
                    input[i] = random.nextFloat() * 2 - 1;
                }
                // Non-linear relationship (sinusoidal)
                output[0] = (float) Math.sin(input[0] * Math.PI) + input[1];
                output[1] = (float) Math.cos(input[1] * Math.PI) - input[0];
                return new DataPoint(input, output);
            }
        }

        // Data Point
        static class DataPoint {
            float[] input;
            float[] target;

            DataPoint(float[] input, float[] target) {
                this.input = input;
                this.target = target;
            }
        }

        public ModelEvaluator() {
            super("Model Evaluator");
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            setSize(1200, 800);

            // Initialize Model and Data
            acml = new AdaptiveCacheAwareMemoryLayer(
                    numHashes, chunkSize, valueSize, tableSize,
                    new AdaptiveCacheAwareMemoryLayer.SimpleHasher(),
                    new AdaptiveCacheAwareMemoryLayer.LinearInterpolator(),
                    new AdaptiveCacheAwareMemoryLayer.LRUEvictionPolicy(tableSize),
                    1,1
                    //weightClamp, gradClamp
            );
            generateData();

            // Create Panels
            dataPanel = new DataPanel();
            parameterPanel = new ParameterPanel();
            lossPanel = new LossPanel();

            // Layout
            setLayout(new BorderLayout());
            var mainSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, dataPanel, parameterPanel);
            mainSplitPane.setDividerLocation(0.7);
            var verticalSplitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, mainSplitPane, lossPanel);
            add(verticalSplitPane, BorderLayout.CENTER);
            verticalSplitPane.setDividerLocation(0.7);

            setVisible(true);

            // Start Training Loop (in a separate thread)
            new Thread(this::trainingLoop).start();
        }

        private void generateData() {
            DataGenerator dataGenerator = switch (complexity) {
                case 1 -> new SimpleDataGenerator();
                case 2 -> new MediumDataGenerator();
                case 3 -> new ComplexDataGenerator();
                default -> throw new IllegalStateException("Unexpected value: " + complexity);
            };

            trainingData = new ArrayList<>();
            testingData = new ArrayList<>();
            for (var i = 0; i < datasetSize; i++) {
                trainingData.add(dataGenerator.generate(inputSize, outputSize));
                testingData.add(dataGenerator.generate(inputSize, outputSize));
            }
        }

        private float calculateLoss() {
            float totalLoss = 0;
            for (var dp : testingData) {
                var predicted = acml.forward(dp.input);
                for (var i = 0; i < outputSize; i++) {
                    totalLoss += Math.abs(predicted[i] - dp.target[i]);
                }
            }
            return totalLoss / testingData.size();
        }

        private void trainingLoop() {
            while (true) {
                // Train on a batch
                for (var dp : trainingData) {
                    var output = acml.forward(dp.input);
                    var outputGradient = new float[outputSize];
                    for (var i = 0; i < outputSize; i++) {
                        outputGradient[i] = (dp.target[i] - output[i]) * (float) learningRate; // Simplified error
                    }
                    //System.out.println(n2(outputGradient));
                    acml.backward(dp.input, outputGradient);
                }

                // Evaluate and Update Visualization
                var loss = calculateLoss();
                SwingUtilities.invokeLater(() -> {
                    dataPanel.repaint();
                    parameterPanel.updateValues();
                    lossPanel.updateLoss(loss);
                    lossPanel.repaint();
                    System.out.println(loss);
                    repaint();
                });

                // Small Delay
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }

        // Data Visualization Panel
        class DataPanel extends JPanel {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                var g2d = (Graphics2D) g;
                g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                var w = getWidth();
                var h = getHeight();

                // Draw Target and Predicted Outputs (only for 2D output for now)
                if (outputSize == 2) {
                    for (var dp : testingData) {
                        var predicted = acml.forward(dp.input);

                        // Scale and center points
                        var xTarget = (int) (dp.target[0] * w / 4 + w / 2);
                        var yTarget = (int) (dp.target[1] * h / 4 + h / 2);
                        var xPredicted = (int) (predicted[0] * w / 4 + w / 2);
                        var yPredicted = (int) (predicted[1] * h / 4 + h / 2);

                        // Draw target (blue)
                        g2d.setColor(Color.BLUE);
                        g2d.fill(new Ellipse2D.Double(xTarget - 3, yTarget - 3, 6, 6));

                        // Draw predicted (red)
                        g2d.setColor(Color.RED);
                        g2d.fill(new Ellipse2D.Double(xPredicted - 3, yPredicted - 3, 6, 6));

                        // Draw error line
                        g2d.setColor(Color.BLACK);
                        g2d.drawLine(xTarget, yTarget, xPredicted, yPredicted);
                    }
                }

//                // Draw parameter matrix
//                var matrixCellSize = 20;
//                var matrixXOffset = 20;
//                var matrixYOffset = 20;
//                for (var i = 0; i < tableSize; i++) {
//                    for (var j = 0; j < chunkSize; j++) {
//                        for (var k = 0; k < 1 /*valueSize*/ /*TODO*/; k++) {
//                            var value = acml.table[i].values[j][k];
//                            // Normalize value for color mapping (assuming values are between -1 and 1)
//                            var normalizedValue = (value + 1) / 2;
//                            g2d.setColor(Color.getHSBColor(normalizedValue, 1, 1));
//                            g2d.fillRect(matrixXOffset + j * matrixCellSize, matrixYOffset + i * matrixCellSize, matrixCellSize, matrixCellSize);
//                        }
//                    }
//                }
            }
        }

        // Parameter Control Panel
        class ParameterPanel extends JPanel {
            private JLabel learningRateLabel;
            private JLabel complexityLabel;
            private JLabel tableSizeLabel;
            private JLabel numHashesLabel;
            private JLabel chunkSizeLabel;

            public ParameterPanel() {
                setLayout(new GridLayout(0, 2));
                setBorder(BorderFactory.createTitledBorder("Parameters"));

                // Learning Rate
                learningRateLabel = new JLabel("Learning Rate: " + learningRate);
                var learningRateSlider = new JSlider(JSlider.HORIZONTAL, 0, 100, (int) (learningRate * 1000));
                learningRateSlider.addChangeListener(e -> {
                    learningRate = learningRateSlider.getValue() / 1000.0;
                    learningRateLabel.setText("Learning Rate: " + learningRate);
                });

                // Complexity
                complexityLabel = new JLabel("Complexity: " + complexity);
                var complexityLevels = new String[]{"Simple", "Medium", "Complex"};
                var complexityComboBox = new JComboBox<String>(complexityLevels);
                complexityComboBox.setSelectedIndex(complexity - 1); // Adjust for 0-based index
                complexityComboBox.addActionListener(e -> {
                    complexity = complexityComboBox.getSelectedIndex() + 1;
                    complexityLabel.setText("Complexity: " + complexity);
                    generateData();
                });

                // Table Size
                tableSizeLabel = new JLabel("Table Size: " + tableSize);
                var tableSizeSlider = new JSlider(JSlider.HORIZONTAL, 1, 8, (int) (Math.log(tableSize) / Math.log(2)));
                tableSizeSlider.addChangeListener(e -> {
                    tableSize = (int) Math.pow(2, tableSizeSlider.getValue());
                    tableSizeLabel.setText("Table Size: " + tableSize);
                    resetModel();
                });

                // Num Hashes
                numHashesLabel = new JLabel("Num Hashes: " + numHashes);
                var numHashesSlider = new JSlider(JSlider.HORIZONTAL, 1, 16, numHashes);
                numHashesSlider.addChangeListener(e -> {
                    numHashes = numHashesSlider.getValue();
                    numHashesLabel.setText("Num Hashes: " + numHashes);
                    resetModel();
                });

                // Chunk Size
                chunkSizeLabel = new JLabel("Chunk Size: " + chunkSize);
                var chunkSizeSlider = new JSlider(JSlider.HORIZONTAL, 1, 6, (int) (Math.log(chunkSize) / Math.log(2)));
                chunkSizeSlider.addChangeListener(e -> {
                    chunkSize = (int) Math.pow(2, chunkSizeSlider.getValue());
                    chunkSizeLabel.setText("Chunk Size: " + chunkSize);
                    resetModel();
                });

                add(learningRateLabel);
                add(learningRateSlider);
                add(complexityLabel);
                add(complexityComboBox);
                add(tableSizeLabel);
                add(tableSizeSlider);
                add(numHashesLabel);
                add(numHashesSlider);
                add(chunkSizeLabel);
                add(chunkSizeSlider);
            }

            private void resetModel() {
                acml = new AdaptiveCacheAwareMemoryLayer(
                        numHashes, chunkSize, valueSize, tableSize,
                        new AdaptiveCacheAwareMemoryLayer.SimpleHasher(),
                        new AdaptiveCacheAwareMemoryLayer.LinearInterpolator(),
                        new AdaptiveCacheAwareMemoryLayer.LRUEvictionPolicy(tableSize),
                        1,1
                );
            }

            public void updateValues() {
                // Update any values displayed in the panel (if needed)
            }
        }

        class LossPanel extends JPanel {
            private java.util.List<Float> lossHistory = new ArrayList<>();
            private static final int MAX_DATA_POINTS = 100;

            public LossPanel() {
                setBorder(BorderFactory.createTitledBorder("Loss"));
            }

            public void updateLoss(float loss) {
                lossHistory.add(loss);
                if (lossHistory.size() > MAX_DATA_POINTS) {
                    lossHistory.remove(0);
                }
            }

            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                var g2d = (Graphics2D) g;
                g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                var w = getWidth();
                var h = getHeight();

                if (lossHistory.isEmpty()) return;

                // Find max loss for scaling
                float maxLoss = lossHistory.stream().max(Float::compare).get();

                // Draw loss curve
                g2d.setColor(Color.BLUE);
                var x1 = 0;
                var y1 = (int) (h - lossHistory.get(0) / maxLoss * h);
                for (var i = 1; i < lossHistory.size(); i++) {
                    var x2 = (int) (i * 1.0 / MAX_DATA_POINTS * w);
                    var y2 = (int) (h - lossHistory.get(i) / maxLoss * h);
                    g2d.drawLine(x1, y1, x2, y2);
                    x1 = x2;
                    y1 = y2;
                }
            }
        }

        public static void main(String[] args) {
            SwingUtilities.invokeLater(ModelEvaluator::new);
        }

    }
}