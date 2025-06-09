package jcog.tensor.deprtensor;

import jcog.Util;

import static jcog.Util.assertFinite;

public class ClipTensor extends TensorFn {
    private final double min, max;  // Store the start and end indices for the slice

    ClipTensor(Tens0r parent, double min, double max) {
        super(parent);  // Calls the default constructor to initialize fields
        assertFinite(min);
        assertFinite(max);
        this.min = min;
        this.max = max;

        this.data = newArraySizeOf(parent);
        this.grad = newArraySizeOf(parent);

        forward();  // Initialize data based on the parent tensor
    }

    @Override
    public void forward() {
        double[][] x1 = this.x[0].data;
        double[][] y = data;
        Util.clamp(x1, y, min, max);
    }


    @Override
    public void backward() {
        if (x == null)
            return;
        Tens0r parent = this.x[0];
        double[][] pg = parent.grad;
        double[][] tg = this.grad;
        double[][] d = data;
        int I = d.length, J = d[0].length;
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++) {
                double dij = d[i][j];
                // Zero gradients where the clipping is active
                //if (dij >= min && dij <= max) //inclusive
                if (dij > min && dij < max) //exclusive
                    pg[i][j] += tg[i][j];
            }

        x[0].backward();
    }

}
