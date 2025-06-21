package jcog.deep;

import jcog.tensor.experimental.RecurrentAutoencoder;
import jcog.tensor.experimental.RecurrentAutoencoder.LSTMRecurrentAutoencoder;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class RecurrentAutoencoderTest {
    private RecurrentAutoencoder autoencoder;
    private LSTMRecurrentAutoencoder lstmAutoencoder;
    private double[] inputs, targets;

    @BeforeEach
    public void setUp() {
        int inputSize = 4, hiddenSize = 6, sequenceLength = 12;
        double learningRate = 0.01;
        autoencoder = new RecurrentAutoencoder(inputSize, hiddenSize, sequenceLength, learningRate);
        lstmAutoencoder = new LSTMRecurrentAutoencoder(inputSize, hiddenSize, sequenceLength, learningRate);
        inputs = new double[inputSize * sequenceLength];
        targets = new double[inputSize * sequenceLength];
        generateSampleData(inputs);
        generateSampleData(targets);
    }

    private void generateSampleData(double[] data) {
        Random rand = new Random();
        for (int i = 0; i < data.length; i++)
            data[i] = rand.nextDouble();
    }

    @Test
    public void testGet() {
        double[] output = autoencoder.get(inputs);
        assertEquals(inputs.length, output.length);

        double[] lstmOutput = lstmAutoencoder.get(inputs);
        assertEquals(inputs.length, lstmOutput.length);
    }

    @Test
    public void testPut() {
        testPut(autoencoder);
    }

    @Test public void testPutLSTM() {
        testPut(lstmAutoencoder);
    }

    private void testPut(RecurrentAutoencoder e) {
        double initialLoss = e.loss(inputs, targets);
        for (int i = 0; i < 10000; i++) {
            e.put(inputs, targets);
            if(i%100==0)
                System.out.println(e.loss(inputs, targets));
        }
        double finalLoss = e.loss(inputs, targets);
        assertTrue(finalLoss < initialLoss/2);
    }

    @Test
    public void testInvalidInput() {
        double[] invalidInput = new double[15]; // Invalid shape
        assertThrows(IllegalArgumentException.class, () -> autoencoder.get(invalidInput));
        assertThrows(IllegalArgumentException.class, () -> lstmAutoencoder.get(invalidInput));
    }


    @Nested
    public class CharTest {
        private RecurrentAutoencoder autoencoder;
        private LSTMRecurrentAutoencoder lstmAutoencoder;
        private double[] inputs, targets;

        @BeforeEach
        public void setUp() {
            int inputSize = 5, hiddenSize = 3, sequenceLength = 10;
            double learningRate = 0.01;
            autoencoder = new RecurrentAutoencoder(inputSize, hiddenSize, sequenceLength, learningRate);
            lstmAutoencoder = new LSTMRecurrentAutoencoder(inputSize, hiddenSize, sequenceLength, learningRate);
            inputs = new double[inputSize * sequenceLength];
            targets = new double[inputSize * sequenceLength];
            generateSampleData(inputs);
            generateSampleData(targets);
        }

        private void generateSampleData(double[] data) {
            Random rand = new Random();
            for (int i = 0; i < data.length; i++)
                data[i] = rand.nextDouble();
        }

        @Test
        public void testQuery() {
            double[] output = autoencoder.get(inputs);
            assertEquals(inputs.length, output.length);

            double[] lstmOutput = lstmAutoencoder.get(inputs);
            assertEquals(inputs.length, lstmOutput.length);
        }

        @Test
        public void testput() {
            testPut(autoencoder);
        }

        @Test
        public void testPutLSTM() {
            testPut(lstmAutoencoder);
        }

        private void testPut(RecurrentAutoencoder e) {
            double initialLoss = e.loss(inputs, targets);
            for (int i = 0; i < 100; i++)
                e.put(inputs, targets);
            double finalLoss = e.loss(inputs, targets);
            assertTrue(finalLoss < initialLoss);
        }


        @Test
        public void testInvalidInput() {
            double[] invalidInput = new double[15]; // Invalid shape
            assertThrows(IllegalArgumentException.class, () -> autoencoder.get(invalidInput));
            assertThrows(IllegalArgumentException.class, () -> lstmAutoencoder.get(invalidInput));
        }

        @Test
        public void testUUEncodingDecoding() {
            String inputText = "Hello World!";
            double[][] encodedInputs = uuencode(inputText);
            double[][] encodedTargets = uuencode(inputText);

            int inputSize = encodedInputs[0].length;
            int sequenceLength = encodedInputs.length;
            double learningRate = 0.01;

            LSTMRecurrentAutoencoder e = new LSTMRecurrentAutoencoder(inputSize, 8, sequenceLength, learningRate);

            // Flatten the 2D array to a 1D array for training
            double[] flatInputs = flattenArray(encodedInputs);
            double[] flatTargets = flattenArray(encodedTargets);

            double initialLoss = e.loss(flatInputs, flatTargets);
            for (int i = 0; i < 1000; i++) {
                e.put(flatInputs, flatTargets);
                String decodedOutput = uudecode(reshapeArray(e.get(flatInputs), sequenceLength, inputSize));
                if (i%20==0)
                    System.out.println(e.loss(flatInputs, flatTargets) + "\t" + decodedOutput);
            }
            double finalLoss = e.loss(flatInputs, flatTargets);

            assertTrue(finalLoss < initialLoss/2);

            // Test decoding
            String decodedOutput = uudecode(reshapeArray(e.get(flatInputs), sequenceLength, inputSize));
            assertEquals(inputText, decodedOutput);
        }

        private double[] flattenArray(double[][] array) {
            double[] flatArray = new double[array.length * array[0].length];
            for (int i = 0; i < array.length; i++)
                System.arraycopy(array[i], 0, flatArray, i * array[0].length, array[0].length);
            return flatArray;
        }

        private double[][] reshapeArray(double[] flatArray, int rows, int cols) {
            double[][] array = new double[rows][cols];
            for (int i = 0; i < rows; i++)
                System.arraycopy(flatArray, i * cols, array[i], 0, cols);
            return array;
        }


        private final char[] uuencodeTable = {
                '`', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
                '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_'
        };

        private double[][] uuencode(String text) {
            byte[] bytes = text.getBytes();
            double[][] encoded = new double[bytes.length * 8 / 6][6];
            int index = 0;
            for (int i = 0; i < bytes.length; i += 3) {
                int value = 0;
                for (int j = 0; j < 3 && i + j < bytes.length; j++) {
                    value |= (bytes[i + j] & 0xFF) << (16 - 8 * j);
                }
                for (int j = 0; j < 4; j++) {
                    int charIndex = (value >> (18 - 6 * j)) & 0x3F;
                    if (index < encoded.length) {
                        for (int k = 0; k < 6; k++) {
                            encoded[index][k] = (charIndex >> (5 - k)) & 1;
                        }
                        index++;
                    }
                }
            }
            return encoded;
        }

        private String uudecode(double[][] encoded) {
            byte[] bytes = new byte[encoded.length * 6 / 8];
            int index = 0;
            for (int i = 0; i < encoded.length; i += 4) {
                int value = 0;
                for (int j = 0; j < 4 && i + j < encoded.length; j++) {
                    int charValue = 0;
                    for (int k = 0; k < 6; k++) {
                        charValue |= (encoded[i + j][k] > 0.5 ? 1 : 0) << (5 - k);
                    }
                    value |= charValue << (18 - 6 * j);
                }
                for (int j = 0; j < 3 && index < bytes.length; j++) {
                    bytes[index++] = (byte) ((value >> (16 - 8 * j)) & 0xFF);
                }
            }
            return new String(bytes);
        }
    }
}