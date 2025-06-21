//package jcog.nn.tf;
//
//import org.tensorflow.Graph;
//import org.tensorflow.Operand;
//import org.tensorflow.Session;
//import org.tensorflow.framework.optimizers.Adam;
//import org.tensorflow.op.Ops;
//import org.tensorflow.op.core.Placeholder;
//import org.tensorflow.op.core.Variable;
//import org.tensorflow.types.TFloat32;
//
//public class XOR2_TensorFlow {
//    public static void main(String[] args) {
//        System.setProperty("org.bytedeco.javacpp.logger.debug","true");
//        try (Graph graph = new Graph()) {
//            Ops tf = Ops.create(graph);
//
//            // Define input and output placeholders
//            Placeholder<TFloat32> x = tf.placeholder(TFloat32.class);
//            Placeholder<TFloat32> y = tf.placeholder(TFloat32.class);
//
//            // Define the model
//            Variable<TFloat32> w1 = tf.variable(tf.random.truncatedNormal(tf.constant(new int[]{2, 4}), TFloat32.class));
//            Variable<TFloat32> b1 = tf.variable(tf.zeros(tf.constant(new int[]{4}), TFloat32.class));
//            Operand<TFloat32> layer1 = tf.nn.relu(tf.math.add(tf.linalg.matMul(x, w1), b1));
//
//            Variable<TFloat32> w2 = tf.variable(tf.random.truncatedNormal(tf.constant(new int[]{4, 1}), TFloat32.class));
//            Variable<TFloat32> b2 = tf.variable(tf.zeros(tf.constant(new int[]{1}), TFloat32.class));
//            Operand<TFloat32> output = tf.nn.elu(tf.math.add(tf.linalg.matMul(layer1, w2), b2));
//
//            // Define loss function
//            Operand<TFloat32> loss = tf.math.mean(tf.math.square(tf.math.sub(output, y)),  tf.constant(new int[]{0,0}));
//
//
//            // Define optimizer
//            var optimizer = new Adam(graph, 0.1f);
//            var train = optimizer.minimize(loss);
//
//
//            // Prepare training data
//            float[][] inputData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
//            float[][] outputData = {{0}, {1}, {1}, {0}};
//
//            try (Session session = new Session(graph)) {
//                // Initialize variables
//                session.initialize();
//
//                // Training loop
//                for (int epoch = 0; epoch < 5000; epoch++) {
//                    for (int i = 0; i < inputData.length; i++) {
//                        var inputTensor = tf.constant(inputData[i]);
//                        var outputTensor = tf.constant(outputData[i]);
//                          session.runner()
//                                  .feed(x.asOutput(), inputTensor.asTensor())
//                                  .feed(y.asOutput(), outputTensor.asTensor())
//                                  .addTarget(train)
//                                  .run();
//
//                    }
//                }
//
//                // Test the trained model
//                for (float[] input : inputData) {
//                    TFloat32 inputTensor = TFloat32.vectorOf(input);
//                    var result = session.runner()
//                            .feed(x.asOutput(), inputTensor.asRawTensor())
//                            .fetch(output)
//                            .run()
//                            .get(0);
//
//                    var outputValue = result.asRawTensor().data().asFloats();
//                    System.out.printf("Input: [%.0f, %.0f], Output: %.4f%n", input[0], input[1], outputValue);
//
//                }
//            }
//        }
//    }
//}