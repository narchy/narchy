package jcog.lab;

import jcog.signal.FloatRange;
import jcog.util.Range;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LabTest {

    @Test
    void Simplest() {
        var a = Lab.auto(Model.class);

        var r = a.optimizeEach(Model::score).run(64);

        r.print();

        System.out.println("\ndisc=1");
        r.tree(2, 4).print();
        System.out.println("\ndisc=2");
        r.tree(3, 4).print();
        System.out.println("\ndisc=3");
        var t = r.tree(4, 4);
        t.print();
        t.printExplanations();

        var best = r.best();
        assertTrue(((Number) best.getObject(0)).doubleValue() >= 4.9f);
        assertTrue(a.vars.size() >= 4);
        assertEquals(5, r.data.columnCount());
    }

    @Test
    void ExperimentRunSimple() {

        var t = new Lab<>(Dummy::new)
            .sense(Sensor.unixtime)
            .sense(Sensor.nanotime())
            .sense("hash", System::identityHashCode)
            .experiment((x)->{ /* NOP */ });

        t.run();
        t.run();

        var data = ((Experiment.DataTableTarget)t.data).data;
        assertEquals(3, data.columnCount());
        assertEquals(2, data.rowCount());

        t.runSerial(10);
        assertEquals(2+10, data.rowCount());

        t.runParallel(100);
        assertEquals(2+10+100, data.rowCount());
    }

    @Test
    void ExperimentRunComplex() {

        LabelSensor<Dummy> ctx;

        var lab = new Lab<>(Dummy::new)
                .sense(Sensor.unixtime)
                .sense(Sensor.nanotime())
                .sense(ctx = Sensor.label("ctx"))
                .sense("a", (Dummy m) -> m.a);

        var t = lab.experiment((m, trial) -> {
            ctx.record("start", trial).set("running");


            for (var i = 0; i < 10; i++) {
                m.a = (float) Math.sin(i);
                trial.record(m);
            }

            m.a = 0.5f;
            m.a = 0.2f;

            ctx.record("end", trial);
        });

        t.run();
        //System.out.println(t.data.print());
    }

    static class Dummy {
        float a = 0;
    }

    public static class Model {
        public final SubModel sub = new SubModel();
        public final FloatRange floatRange = new FloatRange(0.5f, 0, 10);

        @Range(min=0, max=5/*, step=1f*/)
        public float tweakFloat = 0;

        @Range(min=-4, max=+4, step=2f)
        public int tweakInt = 0;

        @SuppressWarnings("unused")
        public static final int untweakInt = 0;

        public double score() {
            return (
                    tweakInt +
                            Math.sin(-1 + tweakFloat) * tweakFloat +
                            (1f/(1f+sub.tweakFloatSub)))
                    + floatRange.floatValue();
        }

        @SuppressWarnings("unused")
        public float score2() {
            return (float)(Math.cos(floatRange.floatValue()/4f));
        }

    }

    public static class SubModel {
        @Range(min=0, max=3, step=0.05f)
        public float tweakFloatSub;
    }



//    @Test
//    void optilive1() {
//        Lab<LabTest.Model> a = new Lab<>(LabTest.Model::new).auto();
//
//        Optilive<LabTest.Model,?> o = a.optilive(LabTest.Model::score);
//
//        o.start();
//        Util.sleepMS(1000);
//        o.pause();
//        o.resume();
//        Util.sleepMS(1000);
//        o.stop();
//    }

//
//    @Test
//    public void MultiObjective() {
//        Variables<Model> a = new Variables<>(Model::new).discover();
//
//
//
//        Lab.Result r = a.optimize((m->m), ((FloatFunction<Model>)(m->m.score())), m->m.score2()).run(25);
//        assertEquals(7, r.data.attrCount());
//
//        r.print();
//        r.tree(3, 6).print();
//
//        r.data.print();
//    }
}