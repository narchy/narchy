package jcog;

/** an exception which can only truly be caught by a developer */
@Research
@Is({"Imagination", "Planning"})
public class TODO extends UnsupportedOperationException {

    public TODO() { super(); }

    public TODO(String what) {
        super(what);
    }

    public TODO(Object what) {
        this(what.toString());
    }

}