package jcog.tensor.deprtensor;

import jcog.data.list.Lst;
import jcog.math.FloatSupplier;

import java.util.List;

abstract public class OptimizeWithMemory extends Optimize {

    private final int valuesPerParameter;
    private final List<double[][]> data = new Lst<>();

    protected OptimizeWithMemory(FloatSupplier learningRate, int valuesPerParameter) {
        super(learningRate);
        this.valuesPerParameter = valuesPerParameter;
    }

    protected double[][] data(int index, int rows, int cols) {
        List<double[][]> d = data;
        if (index >= d.size())
            return init(rows, cols);

        double[][] a = d.get(index);
        return arrayResized(rows, cols, a) ? realloc(index, rows, cols, d) : a;
    }

    private double[][] init(int rows, int cols) {
        double[][] array = new double[rows][cols];
        data.add(array);
        return array;
    }

    private static boolean arrayResized(int rows, int cols, double[][] array) {
        return array.length != rows || array[0].length != cols;
    }

    private static double[][] realloc(int index, int rows, int cols, List<double[][]> list) {
        double[][] array;
        //THIS SHOULD NOT HAPPEN IF RE-USING THE RIGHT OPTIMIZER ON THE SAME NETWORK
        array = new double[rows][cols];
        list.set(index, array);
        return array;
    }
}
