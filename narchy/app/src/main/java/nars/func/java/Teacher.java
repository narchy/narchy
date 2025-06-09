package nars.func.java;

import nars.$;
import nars.NAR;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Consumer;
import java.util.function.Predicate;

public class Teacher<X> {

    static final Logger logger = LoggerFactory.getLogger(Teacher.class);
    protected final X x;
    protected final NAR n;

    public Teacher(NAR n, X c) {
        this(new Opjects(n.main()), c);
    }

    public Teacher(NAR n, Class<? extends X> c) {
        this(new Opjects(n.main()), c);
    }

    public Teacher(Opjects objs, X instance) {
        this.n = objs.nar;
        this.x = objs.the("a_" + instance.getClass().getSimpleName(), instance);
    }

    public Teacher(Opjects objs, Class<? extends X> clazz) {
        this.n = objs.nar;
        this.x = objs.a($.$$("(a," + clazz.getSimpleName() +")"), clazz);
    }

    public Trick<X> teach(String taskName,
                      Consumer<X> pre,
                      Consumer<X> task,
                      Predicate<X> post /* validation*/) {

        Trick<X> t = new Trick<>(taskName, pre, task, post);
        t.train(x, n);

        boolean valid = t.valid(x);
        if (!valid)
            throw new RuntimeException("invalid after training. please dont confuse NARS");

        return t;
    }
}