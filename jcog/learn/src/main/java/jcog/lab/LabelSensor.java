package jcog.lab;


public class LabelSensor<X> extends Sensor<X,String> {

    private String cur = "";

    public LabelSensor(String id) {
        super(id);
    }

    public void set(String cur) {
        this.cur = cur;
    }

    public String get() {
        return cur;
    }

    @Override
    public String apply(Object o) {
        return get();
    }

    /** convenience method */
    @Deprecated public LabelSensor<X> record(String value, Experiment<?> e) {
        set(value);
//        e.record();
        return this;
    }

    @Override
    public void register(Experiment.DataTarget data) {
        data.defineLabel(id);
    }
}