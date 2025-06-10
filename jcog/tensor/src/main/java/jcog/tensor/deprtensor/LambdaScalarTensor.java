package jcog.tensor.deprtensor;

import java.util.function.DoubleSupplier;

final class LambdaScalarTensor extends TensorFn {
    private final DoubleSupplier value;

    LambdaScalarTensor(DoubleSupplier value) {
        super();  // Set the operation type if needed
        this.value = value;
        this.data = new double[1][1];  // Allocate memory for a scalar
        forward(); //initialize
    }

    @Override
    public void forward() {
        this.data[0][0] = value.getAsDouble(); // Update the scalar value on each forward pass
    }

    // No backward method needed as there should be no gradient computation
}
