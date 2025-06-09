package jcog.tensor.deprtensor;

import jcog.data.list.Lst;
import jcog.util.KahanSum;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public abstract class Parameterized {
    final Lst<Tens0r> params = new Lst<>();

    public abstract Tens0r forward(Tens0r input);

    public abstract void zeroGrad();

    public List<Tens0r> params() {
        return params;
    }

    public void gradScale(double v) {
        if (v == 1) return;
        for (var l : params())
            l.scaleGrad(v);
    }

    /**
     * sums the abs of the grad, ex: to measure incremental progress
     */
    public double gradAbsSum() {
        KahanSum s = new KahanSum();
        for (var p : params())
            s.add(p.gradSumAbs());
        return s.value();
    }

    public double gradMaxAbs() {
        double m = 0;
        for (var p : params()) {
            double x = p.gradMaxAbs();
            m = Math.max(x, m);
        }
        return m;
    }

    /**
     * LERPs parameters
     * if rng!=null: hard else soft
     */
    public void setData(TensorFn.Layers source, double tau, @Nullable Random rng) {
        if (rng != null) {
            if (rng.nextFloat() > tau)
                return;
            tau = 1;
        }

        _setData(source, tau);
    }

    private void _setData(TensorFn.Layers source, double tau) {
        List<Tens0r> s = source.params(), t = this.params();
        IntStream.range(0, t.size()).forEach(i ->
                t.get(i).setDataSoft(s.get(i), tau));
    }

    public double paramCount() {
        int sum = 0;
        for (var p : params())
            sum += p.volume();
        return sum;
    }
}
