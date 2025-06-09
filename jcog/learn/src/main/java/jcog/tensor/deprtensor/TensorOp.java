package jcog.tensor.deprtensor;

import jcog.TODO;
import jcog.Util;
import jcog.tensor.Tensor;
import jcog.util.KahanSum;

import static java.lang.Math.exp;
import static jcog.Util.fma;

public enum TensorOp {
    ADD() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            Tens0r a = x[0], b = x[1];
            double[][] A = a.data, B = b.data;
            int m = A.length, n = A[0].length;
            if (m != B.length || n != B[0].length)
                throw new IllegalArgumentException("Shape mismatch");
            return new double[m][n];
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            Tens0r a = x[0], b = x[1];
            double[][] A = a.data, B = b.data;
            int m = y.length, n = y[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    y[i][j] = A[i][j] + B[i][j];
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0], B = x[1];
            double[][] ad = A.data, bd = B.data;
            double[][] ag = A.grad, bg = B.grad;
            int w = ad.length, h = ad[0].length;
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++) {
                    double gij = grad[i][j];
                    ag[i][j] += gij;
                    bg[i][j] += gij;
                }
        }

    },
    SUM {
        @Override
        public double[][] allocate(Tens0r[] x) {
            // Since sum returns a single value, allocate a 1x1 array.
            return new double[1][1];
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            y[0][0] = Util.sum(x[0].data);
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            // Distribute the gradient to each element of the original tensor's gradient.
            Util.addTo(x[0].grad, grad[0][0]);
        }
    },
    ADD_SCALAR() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            Tens0r a = x[0], _b = x[1];

            double[][] A = a.data;
            double b = _b.scalar(); //ensures B is scalar

            int m = A.length, n = A[0].length;
            return new double[m][n];
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            Tens0r a = x[0], b = x[1];
            double[][] A = a.data;
            double B = b.scalar(); //ensures B is scalar
            int m = y.length, n = y[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) {
                    y[i][j] = A[i][j] + B;
                }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0], B = x[1];
            double[][] ad = A.data, ag = A.grad;
            double[][] bd = B.data, bg = B.grad;
            int w = ad.length, h = ad[0].length;
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++) {
                    double gij = grad[i][j];
                    ag[i][j] += gij;
                    bg[0][0] += gij;
                }
        }
    },
    RELU() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            int m = y.length, n = y[0].length;
            var A = x[0].data;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) {
                    double a = A[i][j];
                    y[i][j] = a <= 0 ? 0 : a;
                }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0];
            double[][] ad = A.data, ag = A.grad;

            int w = ad.length, h = ad[0].length;
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++)
                    ag[i][j] += grad[i][j] * (ad[i][j] <= 0 ? 0 : 1);
        }

    },
    LEAKY_RELU() {
        public static final double alpha = 0.01;  // Leakiness coefficient

        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            int m = y.length, n = y[0].length;
            var A = x[0].data;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) {
                    double a = A[i][j];
                    y[i][j] = a <= 0 ? alpha * a : a;  // Use alpha * a if a <= 0
                }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0];
            double[][] ad = A.data, ag = A.grad;

            int w = ad.length, h = ad[0].length;
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++) {
                    ag[i][j] += grad[i][j] * (ad[i][j] <= 0 ? alpha : 1);  // Use alpha if ad[i][j] <= 0
                }
        }
    },
    /**
     * https://arxiv.org/pdf/2310.01365
     */
    ELEPHANT() {
        /*
            // The Elephant function
            public double elephant(double x) {
                return 1.0 / (1.0 + Math.pow(x / a, d));
            }

            // The derivative of the Elephant function
            public double elephantDerivative(double x) {
                double base = x / a;
                double numerator = d * Math.pow(base, d - 1);
                double denominator = a * Math.pow(1.0 + Math.pow(base, d), 2);
                return -numerator / denominator;
            }

            public class SimplifiedElephantFunction {

                // The simplified Elephant function where a = 1 and d = 1
                public static double elephant(double x) {
                    return 1.0 / (1.0 + x);
                }

                // The derivative of the simplified Elephant function
                public static double elephantDerivative(double x) {
                    return -1.0 / Math.pow(1.0 + x, 2);
                }
            }
         */
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            int m = y.length, n = y[0].length;
            var A = x[0].data;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) {
                    double a = A[i][j];
                    y[i][j] = 1.0 / (1.0 + a);
                }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0];
            double[][] d = A.data, g = A.grad;

            int w = d.length, h = d[0].length;
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++)
                    g[i][j] += grad[i][j] * (-1.0 / Util.sqr(1.0 + d[i][j]));
        }

    },
    SIGMOID() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] d = x[0].data;
            int m = y.length, n = y[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    y[i][j] = Util.sigmoid(d[i][j]);
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r X = x[0];
            double[][] d = X.data, g = X.grad;
            int w = d.length, h = d[0].length;
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < h; j++) {
                    double s = Util.sigmoid(d[i][j]);
                    g[i][j] += grad[i][j] * (s * (1 - s));
                }
            }
        }

    },
    TANH() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            var a = x[0];
            double[][] A = a.data;
            int m = y.length, n = y[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    y[i][j] = Math.tanh(A[i][j]);
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0];
            double[][] ad = A.data, ag = A.grad;
            int w = ad.length, h = ad[0].length;
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < h; j++) {
                    double s = Math.tanh(ad[i][j]);
                    ag[i][j] += grad[i][j] * (1 - s * s);
                }
            }
        }

    },

    MSE() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            Tens0r a = x[0], b = x[1];
            double[][] Y = a.data, T = b.data;
            int m = Y.length, n = Y[0].length;
            if (m != T.length || n != T[0].length)
                throw new IllegalArgumentException("Shape mismatch");
            return new double[1][1];
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            Tens0r a = x[0], b = x[1];
            double[][] Y = a.data, T = b.data;
            int I = Y.length, J = Y[0].length;
            double sum = 0;
            for (int i = 0; i < I; i++)
                for (int j = 0; j < J; j++)
                    sum += Util.sqr(Y[i][j] - T[i][j]);
            y[0][0] = sum / (I * J);
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0], B = x[1];
            double[][] ad = A.data, bd = B.data;
            double[][] ag = A.grad, bg = B.grad;
            int w = ad.length, h = ad[0].length;
            double errorGrad = 2 * grad[0][0] / (w * h);
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < h; j++) {
                    double d = (ad[i][j] - bd[i][j]) * errorGrad;
                    ag[i][j] += d;  //PREDICTION
                    //bg[i][j] -= d; //TARGET
                }
            }
        }
    }, M_MULT() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            Tens0r a = x[0], b = x[1];
            double[][] A = a.data, B = b.data;
            int m = A.length, p = A[0].length, n = B[0].length;
            if (p != B.length)
                throw new IllegalArgumentException("Shape mismatch");
            return new double[m][n];
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            Tens0r a = x[0], b = x[1];
            double[][] A = a.data, B = b.data;
            int m = A.length, p = A[0].length, n = B[0].length;
            KahanSum s = new KahanSum();
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < p; k++)
                        s.add(A[i][k] * B[k][j]); //s += A[i][k] * B[k][j];
                    y[i][j] += s.valueClear();
                }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0], B = x[1];
            double[][] ad = A.data, bd = B.data;
            double[][] ag = A.grad, bg = B.grad;
            int m = ad.length, p = ad[0].length, n = bd[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double gij = grad[i][j];
                    for (int k = 0; k < p; k++) {
                        ag[i][k] = fma(gij, bd[k][j], ag[i][k]);
                        //ag[i][k] += gij * bd[k][j];
                        bg[k][j] = fma(gij, ad[i][k], bg[k][j]);
                        //bg[k][j] += ad[i][k] * gij;
                    }
                }
            }
        }
    }, NEG() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            int m = y.length, n = y[0].length;
            double[][] A = x[0].data;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    y[i][j] = -A[i][j];
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0];
            double[][] ag = A.grad;
            int w = ag.length, h = ag[0].length;
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++)
                    ag[i][j] -= grad[i][j];  // Subtract the gradient, as the derivative of negation is -1
        }
    }, EW_MULT() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            Tens0r a = x[0], b = x[1];
            if (a.data.length != b.data.length || a.data[0].length != b.data[0].length)
                throw new IllegalArgumentException("Shape mismatch for element-wise multiplication.");

            return new double[a.data.length][a.data[0].length];
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            Tens0r a = x[0], b = x[1];
            double[][] A = a.data, B = b.data;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                double[] yi = y[i], ai = A[i], bi = B[i];
                for (int j = 0; j < n; j++)
                    yi[j] = ai[j] * bi[j];
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r A = x[0], B = x[1];
            double[][] ad = A.data, bd = B.data;
            double[][] ag = A.grad, bg = B.grad;
            int m = ad.length, n = ad[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double gij = grad[i][j];
                    ag[i][j] += gij * bd[i][j];
                    bg[i][j] += gij * ad[i][j];
                }
            }
        }
    }, MULT_SCALAR() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return new double[x[0].data.length][x[0].data[0].length];
        }

        @Override
        public void forward(Tens0r[] X, double[][] y) {
            Tensor.mult(X[0].data, X[1].scalar(), y);
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            Tens0r a = x[0], b = x[1];
            double[][] ad = a.data, bd = b.data;
            double[][] ag = a.grad, bg = b.grad;
            for (int i = 0; i < ag.length; i++) {
                for (int j = 0; j < ag[0].length; j++) {
                    double gij = grad[i][j];
                    ag[i][j] += gij * bd[0][0];
                    if (bg != null) //???
                        bg[0][0] += gij * ad[i][j];
                }
            }
        }
    },
    MIN() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]); // Assuming both tensors have the same shape
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data, B = x[1].data;
            for (int i = 0; i < A.length; i++) {
                for (int j = 0; j < A[i].length; j++) {
                    y[i][j] = Math.min(A[i][j], B[i][j]);
                }
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] A = x[0].data, B = x[1].data;
            double[][] gA = x[0].grad, gB = x[1].grad;
            for (int i = 0; i < A.length; i++) {
                for (int j = 0; j < A[i].length; j++) {
                    double gij = grad[i][j];
                    if (A[i][j] <= B[i][j]) {
                        gA[i][j] += gij;
                    } else {
                        gB[i][j] += gij;
                    }
                }
            }
        }
    },
    MIN_ELE() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return new double[1][1];
        }

        @Override
        public void forward(Tens0r[] tensors, double[][] y) {
            y[0][0] = minVal(tensors);
        }

        private static double minVal(Tens0r[] tensors) {
            double min = Double.POSITIVE_INFINITY;
            for (Tens0r tensor : tensors) {
                if (tensor.data.length!=1)
                    throw new TODO();
                for (double element : tensor.data[0])
                    min = Math.min(min, element);
            }
            return min;
        }

        @Override
        public void backward(Tens0r[] tensors, double[][] grad) {
//            double minVal = Arrays.stream(tensors)
//                    .flatMapToDouble(t -> Arrays.stream(t.data[0]))
//                    .min().getAsDouble();
            double minVal = minVal(tensors);

            //TODO shuffle order for fairness in case minVal is equal in both
            for (Tens0r tensor : tensors) {
                for (int i = 0; i < tensor.data[0].length; i++) {
                    if (tensor.data[0][i] == minVal) {
                        tensor.grad[0][i] += grad[0][0];
                        break;
                    }
                }
            }
        }
    },
    CONCAT() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            int totalSize = x[0].data[0].length + x[1].data[0].length;
            return new double[x[0].data.length][totalSize];
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data, B = x[1].data;
            for (int i = 0; i < A.length; i++) {
                System.arraycopy(A[i], 0, y[i], 0, A[i].length);
                System.arraycopy(B[i], 0, y[i], A[i].length, B[i].length);
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] gA = x[0].grad, gB = x[1].grad;
            int ALen = x[0].data[0].length;
            for (int i = 0; i < grad.length; i++) {
                System.arraycopy(grad[i], 0, gA[i], 0, ALen);
                System.arraycopy(grad[i], ALen, gB[i], 0, grad[i].length - ALen);
            }
        }
    },

    DIV() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data, B = x[1].data;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    y[i][j] = B[i][j] != 0 ? A[i][j] / B[i][j] : Double.POSITIVE_INFINITY;
                }
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] A = x[0].data, B = x[1].data;
            double[][] ag = x[0].grad, bg = x[1].grad;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double gij = grad[i][j];
                    double bij = B[i][j];
                    ag[i][j] += gij / bij;
                    bg[i][j] -= gij * A[i][j] / (bij * bij);
                }
            }
        }
    },
    POW() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data;
            double p = x[1].scalar(); // Assuming the power is passed as a scalar tensor
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    y[i][j] = Math.pow(A[i][j], p);
                }
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] A = x[0].data;
            double[][] ag = x[0].grad;
            double p = x[1].scalar();
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    ag[i][j] = fma(p * Math.pow(A[i][j], p - 1), grad[i][j], ag[i][j]); //ag[i][j] += p * Math.pow(A[i][j], p - 1) * grad[i][j];
        }
    },
    EXP() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    y[i][j] = exp(A[i][j]);
                }
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] A = x[0].data;
            double[][] ag = x[0].grad;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    ag[i][j] = fma(exp(A[i][j]), grad[i][j], ag[i][j]); //ag[i][j] += Math.exp(A[i][j]) * grad[i][j];
        }
    },

    LOG() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    y[i][j] = Math.log(A[i][j]);
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] A = x[0].data;
            double[][] ag = x[0].grad;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    ag[i][j] = fma(1 / A[i][j], grad[i][j], ag[i][j]);//ag[i][j] += 1 / (A[i][j]) * grad[i][j];
        }
    },
    LOG1P() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    y[i][j] = Util.log1p(A[i][j]);
                }
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] A = x[0].data;
            double[][] ag = x[0].grad;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    ag[i][j] = fma(1 / (1 + A[i][j]), grad[i][j], ag[i][j]);//ag[i][j] += 1 / (1 + A[i][j]) * grad[i][j];
        }
    },

    SOFTPLUS() {
        @Override
        public double[][] allocate(Tens0r[] x) {
            return Tens0r.newArraySizeOf(x[0]);
        }

        @Override
        public void forward(Tens0r[] x, double[][] y) {
            double[][] A = x[0].data;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    y[i][j] = Util.log1p(exp(A[i][j]));
                }
            }
        }

        @Override
        public void backward(Tens0r[] x, double[][] grad) {
            double[][] A = x[0].data;
            double[][] ag = x[0].grad;
            int m = A.length, n = A[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    ag[i][j] += 1 / (1 + exp(-A[i][j])) * grad[i][j];
                }
            }
        }
    },

    @Deprecated DENSE() {

    };


    public double[][] allocate(Tens0r[] x) {
        throw new TODO();
    }

    public void forward(Tens0r[] x, double[][] y) {
        /* nop */
    }

    public void backward(Tens0r[] x, double[][] grad) {
        /* nop */
    }
}
