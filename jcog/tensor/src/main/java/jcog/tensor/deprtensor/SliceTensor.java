package jcog.tensor.deprtensor;

import jcog.TODO;

public class SliceTensor extends TensorFn {
    private final int start, end;  // Store the start and end indices for the slice

    SliceTensor(Tens0r parent, int start, int end) {
        super(parent);  // Calls the default constructor to initialize fields

        int w = parent.w();
        if (parent.h() != 1 || start < 0 || end > w || end <= start)
            throw new UnsupportedOperationException();

        this.start = start;
        this.end = end;
        this.data = new double[1][end - start];  // Allocate space for the sliced data
        this.grad = new double[1][end - start];  // Allocate space for the gradient

        forward(parent);  // Initialize data based on the parent tensor
    }

    @Override
    public void forward() {
        throw new TODO();
    }

    protected void forward(Tens0r parent) {
        if (parent.h()!=1)
            throw new TODO();
        System.arraycopy(parent.data[0], start, this.data[0], 0, end - start);
    }

    @Override
    public void backward() {
        if (x == null)
            return;

        Tens0r parent = this.x[0];
        double[][] pg = parent.grad;
        double[][] tg = this.grad;
        for (int i = start; i < end; i++) {
            pg[0][i] += tg[0][i - start];
        }

        x[0].backward();
    }

}
