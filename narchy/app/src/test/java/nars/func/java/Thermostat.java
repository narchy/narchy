package nars.func.java;

import jcog.Util;

import java.util.function.Consumer;

class Thermostat {
    private int current;
    private int target;

    /**
     * limits
     */
    static final int cold = 0;
    static final int hot = 3;

    public int is() {
        return current;
    }

    public int should() {
        return target;
    }

    private void add(int delta) {
        current = Util.clamp(current + delta, cold, hot);
    }

    public void up() {
        System.err.println("t++");
        add(+1);
    }

    public void down() {
        System.err.println("t--");
        add(-1);
    }

    public String report() {
        String msg;
        if (is() < should())
            msg = "too cold";
        else if (is() > should())
            msg = "too hot";
        else
            msg = "temperature ok";
        System.err.println(msg);
        return msg;
    }

    void should(int x) {
        System.err.println("temperature should " + x);
        this.target = x;
    }

    void is(int x) {
        System.err.println("temperature is " + x);
        this.current = x;
    }

    static Consumer<Thermostat> change(boolean isHot, boolean shouldHot) {
        return x -> {
            x.is(isHot ? hot : cold);
            x.should(shouldHot ? hot : cold);
        };
    }
}
