package jcog.tensor;

public enum ODEIntegration { ;

    public interface ODE {
        Tensor forward(Tensor x, double t);
    }

    public interface SDE extends ODE {
        Tensor diffusion(Tensor x, double t);
    }

    public interface ODEIntegrator {
        Tensor integrate(ODE ode, Tensor x, double tStart, double tEnd, int steps);
    }

    public static class EulerIntegrator implements ODEIntegrator {

        @Override
        public Tensor integrate(ODE ode, Tensor x, double tStart, double tEnd, int steps) {
            var dt = (tEnd - tStart) / steps;
            for (var i = 0; i < steps; i++)
                x = x.addMul(ode.forward(x, tStart + i * dt), dt);
            return x;
        }
    }

    public static class RK4Integrator implements ODEIntegrator {

        @Override
        public Tensor integrate(ODE ode, Tensor x, double tStart, double tEnd, int steps) {
            var dt = (tEnd - tStart) / steps;
            var dtHalf = dt / 2;

            for (var i = 0; i < steps; i++) {
                var t = tStart + i * dt;
                var k1 = ode.forward(x, t);
                var k2 = ode.forward(x.addMul(k1, dtHalf), t + dtHalf);
                var k3 = ode.forward(x.addMul(k2, dtHalf), t + dtHalf);
                var k4 = ode.forward(x.addMul(k3, dt), t + dt);
                x = x.addMul(k1.addMul(k2, 2).addMul(k3, 2).add(k4), dt / 6);
            }

            return x;
        }
    }

}
