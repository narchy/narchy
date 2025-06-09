package jcog.lab;

public class ProxySensor<X,Y> extends Sensor<X,Y> {
    private final Sensor<X, Y> s;

    public ProxySensor(Sensor<X,Y> s) {
        this(s.id, s);
    }

    public ProxySensor(String id, Sensor<X,Y> s) {
        super(id);
        this.s = s;
    }

    @Override public Y apply(X x) {
        return s.apply(x);
    }

    @Override
    public void register(Experiment.DataTarget data) {
        data.defineNumeric(id);
    }
}
