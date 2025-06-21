package jcog.tensor.model;

import jcog.tensor.Tensor;

import java.util.function.IntSupplier;
import java.util.function.UnaryOperator;

public class ODELayer implements ODEIntegration.ODE, UnaryOperator<Tensor> {

    public final Tensor h0, h1, w, bias;
    public final ODEIntegration.ODEIntegrator integrator;
    public final IntSupplier steps;

    public ODELayer(int inputs, int outputs, int hidden, boolean rk4OrEuler, IntSupplier steps) {
        this(inputs, outputs, hidden, rk4OrEuler ? new ODEIntegration.RK4Integrator() : new ODEIntegration.EulerIntegrator(), steps);
    }

    /**
     * @param rk4OrEuler 5–20 steps are usually sufficient for RK4 to match the accuracy of 30–50 Euler steps.
     */
    public ODELayer(int inputs, int outputs, int hidden, ODEIntegration.ODEIntegrator integrator, IntSupplier steps) {
        h0 = Tensor.randXavier(inputs, hidden).parameter().grad(true);
        h1 = Tensor.randXavier(hidden, outputs).parameter().grad(true);
        w = Tensor.randXavier(hidden, hidden).parameter().grad(true);
        bias = Tensor.zeros(1, hidden).parameter().grad(true); // Time bias
        this.steps = steps;
        this.integrator = integrator;
    }

    public Tensor apply(Tensor x) {
        var s = x.matmul(h0);
        var i = integrator.integrate(this, s, 0, 1, steps.getAsInt());
        return i.matmul(h1);
    }

    /**
     * Time-dependent ODE: dh/dt = tanh(Wh + b * t)
     */
    public Tensor forward(Tensor x, double t) {
        return x.matmul(w).addMul(bias, t).tanh();
    }
}
