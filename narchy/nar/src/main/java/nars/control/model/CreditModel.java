//package nars.control.model;
//
//import jcog.signal.tensor.AtomicFloatVector;
//import nars.control.MetaGoal;
//
//public abstract class CreditModel extends ControlModel {
//    protected static final int dim = MetaGoal.values().length;
//    private transient volatile int numWhy = 0;
//    protected volatile AtomicFloatVector credits = new AtomicFloatVector(0);
//
//    private int index(short why, int what) {
//        return why * dim + what;
//    }
//
//    public void learn(short why, int what, float pri) {
//        if (numWhy < why)
//            realloc((short) (why+2));
//
//        this.credits.addAt(index(why, what), pri);
//    }
//    public float credit(short why, int what) {
//        return credits.getAt(index(why, what));
//    }
//
//    private void realloc(short why) {
//        synchronized (this) {
//            if (numWhy < why) {
//                credits = new AtomicFloatVector((why+1) * dim);
//                numWhy = why;
//            }
//        }
//    }
//
//}