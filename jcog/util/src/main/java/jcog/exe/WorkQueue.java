//package jcog.exe;
//
//import org.jctools.queues.MpmcArrayQueue;
//
//import java.util.function.Consumer;
//
//public class WorkQueue<X> {
//    private final MpmcArrayQueue<X> in;
//
//    public WorkQueue(int capacity) {
//        in = new MpmcArrayQueue<>(capacity);
//    }
//
//    public boolean offer(X x) {
//        return in.offer(x);
//        //in.add(x);
//        //return true;
//    }
//
//    public int size() { return in.size(); }
//
//    public X poll() { return in.poll(); }
//
//    public final void acceptAll(Consumer/*<X>*/ x) {
//        in.drain(x::accept);
//    }
//
//    public int accept(Consumer/*<X>*/ c, float completeness) {
//        int batchSize = -1;
//        X next;
//        int done = 0;
//
//        //CRITICAL
//        while ((next = poll()) != null) {
//
//            c.accept(next);
//            done++;
//
//            if (batchSize == -1) {
//                //initialization once for subsequent attempts
//                int available; //estimate
//                if ((available = size()) <= 0)
//                    break;
//
//                /* all of it if low throttle. this allows most threads to remains asleep while one awake thread takes care of it all */
//                batchSize =
//                        (int) (completeness * available);
//                        //Util.clamp(available, 1, (int) Math.ceil(completeness * available));
//
//            } else if (--batchSize <= 0)
//                break; //enough
//        }
//
//        return done;
//    }
//}
