//package nars.control.model;
//
//import nars.NAR;
//import nars.Task;
//import nars.control.Why;
//import nars.term.Term;
//
//abstract public class ControlModel {
//
////    final Feedback<ControlModel,Short, MetaGoal> feedback = new Feedback<>() {
////        @Override
////        public void accept(ControlModel ctx, Short cause, MetaGoal result) {
////            ctx.learn((short)result.ordinal(), cause, 1);
////        }
////
////        @Override
////        public ControlModel newContext() {
////            return ControlModel.this;
////        }
////    };
//
//    public ControlModel() {
////        if (installFeedback())
////            Feedback.set(feedback);
//    }
//
//    abstract public void update(NAR n);
//
//
//    //    public static float privaluate(Lst<Cause> values, ShortIterator effect) {
////
////    }
//
//    public void learn(int what, Task t, float strength, NAR n) {
//        Term why = t.why();
//        if (why != null)
//            learn(what, why, strength, n);
//    }
//
//    public void learn(int what, Term why, float strength, NAR n) {
//        Why.eval(why, strength, n.control.why.array(), (w, p, CC)->learn(CC[w].id, what, p));
//    }
//
//    abstract public void learn(short why, int what, float strength);
//
//}