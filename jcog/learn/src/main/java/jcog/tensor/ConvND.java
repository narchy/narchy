package jcog.tensor;

import java.util.function.UnaryOperator;

/** TODO untested */
public class ConvND implements UnaryOperator<Tensor> {
    private final Tensor[] filters;
    private final int[] kernelSizes, strides, paddings, inputDims;
    private final int dimensions;
    private final PoolingType poolingType;

    public enum PoolingType { NONE, MAX, AVG }

    public ConvND(int inChannels, int outChannels, int[] kernelSizes, int[] strides, int[] paddings, int[] inputDims, PoolingType poolingType) {
        this.kernelSizes = kernelSizes;
        this.strides = strides;
        this.paddings = paddings;
        this.inputDims = inputDims;
        this.dimensions = inputDims.length;
        this.poolingType = poolingType;

        filters = new Tensor[outChannels];
        for (var i = 0; i < outChannels; i++)
            filters[i] = Tensor.randGaussian(inChannels, product(kernelSizes), Math.sqrt(2.0 / product(kernelSizes))).grad(true).parameter();
        
    }

    // Method to calculate the total number of outputs
    public int outputs() {
        int[] outputShape = outputShape(inputDims);
        int totalSpatialSize = 1;
        for (int size : outputShape) {
            totalSpatialSize *= size;
        }
        return totalSpatialSize * filters.length;  // Multiply by number of output channels (filters)
    }

    private int[] outputShape(int[] inputShape) {
        int[] outputShape = new int[dimensions];
        for (int i = 0; i < dimensions; i++) {
            outputShape[i] = (inputShape[i] - kernelSizes[i] + 2 * paddings[i]) / strides[i] + 1;
        }
        return outputShape;
    }

    @Override
    public Tensor apply(Tensor x) {
        Tensor yy = null;

        for (var filter : filters) {
            var y = convolve(x, filter);
            yy = (yy == null) ? y : yy.add(y);
        }

        return pool(yy, poolingType);
    }

    private int[] calculateOutputDims() {
        var outputDims = new int[dimensions];
        for (var i = 0; i < dimensions; i++)
            outputDims[i] = (inputDims[i] - kernelSizes[i] + 2 * paddings[i]) / strides[i] + 1;

        return outputDims;
    }

    private Tensor convolve(Tensor x, Tensor filter) {
        var reshapedInput = x;//.reshape(inputDims);
        Tensor y = null;

        for (var i = 0; i < dimensions; i++) {
            var z = applyFilterAlongDimension(reshapedInput, filter, i);
            y = y == null ? z : y.add(z);
        }

        return y;
    }

    private Tensor applyFilterAlongDimension(Tensor x, Tensor filter, int dim) {
        var stride = strides[dim];
        var padding = paddings[dim];
        var kernelSize = kernelSizes[dim];
        var inputSize = inputDims[dim];

        Tensor y = null;//Tensor.zeros(calculateOutputDims()[dim], 1);

        for (var i = 0; i < inputSize - kernelSize + 1; i += stride) {
            var region = x.slice(i, i + kernelSize);
            var z = region.mul(filter);
            y = y ==null ? z : y.add(z);
        }

        return y;
    }

    private Tensor pool(Tensor x, PoolingType p) {
        if (p == PoolingType.NONE) return x;

        var poolSize = product(calculateOutputDims());
        Tensor[] y = new Tensor[poolSize];
        for (var i = 0; i < poolSize; i++) {
            var region = x.slice(i, i + poolSize);
            y[i] = p == PoolingType.MAX ? region.maxValue() : region.mean();
        }
        return Tensor.concat(y);
    }

    private static int product(int[] xx) {
        var y = 1;
        for (var x : xx)
            y *= x;
        return y;
    }
}
