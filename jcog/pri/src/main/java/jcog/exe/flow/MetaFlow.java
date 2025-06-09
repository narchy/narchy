package jcog.exe.flow;

import com.google.common.reflect.ClassPath;
import jcog.Str;
import jcog.TODO;
import jcog.data.byt.DynBytes;
import jcog.data.list.Lst;
import jcog.data.map.ConcurrentFastIteratingHashMap;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tree.radix.MyRadixTree;
import org.HdrHistogram.ConcurrentHistogram;
import org.eclipse.collections.impl.map.mutable.primitive.ByteObjectHashMap;

import java.io.PrintStream;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.Target;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Random;
import java.util.function.BooleanSupplier;

import static java.lang.annotation.RetentionPolicy.RUNTIME;

/**
 * EXPERIMENTAL TODO
 * instrumented CFG with cost/benefit analysis measured at decision points
 * stored in call trie
 * https://download.java.net/java/early_access/jdk11/docs/api/java.base/java/lang/StackWalker.html
 * https://www.sitepoint.com/deep-dive-into-java-9s-stack-walking-api/#exceptionvsstackwalker
 * */
public class MetaFlow {

    final Random rng = new XoRoShiRo128PlusRandom(1);

    public void good(float value, Object... args) {
        value(cursor().reset(2).append(args).arrayCopy(), GOOD.id, value);
    }
    public void bad(float value, Object... args) {
        value(cursor().reset(2).append(args).arrayCopy(), BAD.id, value);
    }

//    public void value(String quality, float value, Object[] args) {
//        value(cursor().reset().arrayCopy(), quality(quality).id, value);
//    }

    public MetaFlow forkWhile(BooleanSupplier kontinue, Runnable... options) {
        while (kontinue.getAsBoolean()) {
            runOne(options);
        }
        return this;
    }


    public MetaFlow forkIterations(int iterations, Runnable... options) {
        assert(iterations > 0);

        int[] countDown = {iterations};
        return forkWhile(() -> countDown[0]-- > 0, options);
    }

    public MetaFlow forkUntil(long deadlineNS, Runnable... options) {
        return forkWhile(()->System.nanoTime() < deadlineNS, options);
    }

    public MetaFlow runOne(Runnable[] options) {
        //TODO roulette weighting
        run(options[rng.nextInt(options.length)]);
        return this;
    }

    private void run(Runnable r) {
        try {
            r.run();
        } catch (RuntimeException t) {
            //TODO generatecorrect exception stack frame representation
            bad(1, t.getStackTrace()[0].toString(), "throw", t.getClass().getName());
        }
    }



    static final int digitResolution = 3;
    final float ditherScale = (float) Math.pow(10, digitResolution);

    private void value(byte[] cursor, byte quality, float value) {
        Node n = plan.putIfAbsent(cursor, Node::new);
        n.getIfAbsentPutWithKey(quality,
            (q)->new ConcurrentHistogram(digitResolution) {
                @Override
                public String toString() {
                    return Str.histogramString(this, false);
                }
            })
                .recordValue(Math.round(ditherScale * value));
    }

    @Retention(RUNTIME)
    @Target(ElementType.METHOD)
    public @interface Value {
        String id() default "";
    }
    @Retention(RUNTIME)
    @Target(ElementType.METHOD)
    public @interface Run {
        Class[] any();
    }

    private final StackWalker.StackFrame base;



    static class Quality {
        public final String name;
        public final byte id;

        /** adjustable factor: positive=gain, negative=loss;  0 = no effect in value decision*/
        volatile float value = 0f;

        Quality(String name, byte id) {
            this.name = name;
            this.id = id;
        }

        Quality value(float newValue) {
            this.value = newValue;
            return this;
        }
    }

    /** labels for cost/value qualities */
    final ConcurrentFastIteratingHashMap<String,Quality> qualia = new ConcurrentFastIteratingHashMap<>(new Quality[0]);

    class Node extends ByteObjectHashMap<ConcurrentHistogram> {

        @Override
        public String toString() {
            if (isEmpty())
                return super.toString();

            StringBuilder sb = new StringBuilder(512);
            sb.append('{');
            forEachKeyValue((k,v)-> sb.append(qualia.getIndex(k).name).append('=').append(v).append(", "));
            sb.setLength(sb.length()-2);
            sb.append('}');
            return sb.toString();
        }


    }

//    public static final StackWalker walker = StackWalker.getInstance(StackWalker.Option.RETAIN_CLASS_REFERENCE);
    public static final StackWalker walkerSummary = StackWalker.getInstance();

    static final String thisClassName = MetaFlow.class.getName();

    private static final class StackCursor {

        final StackWalker.StackFrame base;
        final Lst<StackWalker.StackFrame> buffer = new Lst(256);
        private final DynBytes bytes;

        public StackCursor(StackWalker.StackFrame base) {
            bytes = new DynBytes(2048);
            this.base = base;
        }

        public final StackCursor reset() {
            return reset(+1); //for this method
        }

        /** position the cursor for the current stack state */
        public StackCursor reset(int preSkip) {


            //TODO align with previous buffer rather than clearing it, at least some of it is common
            //store the character position of each frame so that on ascent, it can just reposition to that previous location
            buffer.clear();
            walkerSummary.walk((s) -> {
                s.skip(preSkip).takeWhile(f -> {
//                    if (f == at) {
//                        //no change
//                        return false;
//                    } else { //TODO
                        return !equalsClassAndMethod(f, base);
//                    }
                }).filter(MetaFlow::frameFilter).forEach(buffer::add);
                return null;
            });

            //StackWalker.StackFrame at = buffer.getLast();

//            Optional<Class<?>> callerClass = walker.walk(s ->
//                    s.map(StackWalker.StackFrame::getDeclaringClass)
//                            .filter(interestingClasses::contains)
//                            .findFirst());
            bytes.clear();
            StackWalker.StackFrame prev = null;
            for (int i = buffer.size()-1; i>=0; i--) {
            //for (int i = buffer.size()-1; i>=0; i--) {
                StackWalker.StackFrame next = buffer.get(i);
                write(prev, next);
                if (i != 0) {
                    char SEPARATOR = '{';
                    bytes.writeByte((byte)SEPARATOR);
                }
                prev = next;
            }
            return this;
        }



        protected void write(StackWalker.StackFrame prev, StackWalker.StackFrame next) {


            String pClass = null, nClass = next.getClassName();
            if (prev==null || !(pClass = prev.getClassName()).equals(nClass
            )) {
                //TODO elide re-writing package and class if unchanged from previous
                if (pClass!=null && nClass.startsWith(pClass)) {
                    //inner class
                    nClass = nClass.substring(pClass.length());
                }
                bytes.writeUTF(nClass);
                bytes.writeByte((byte)'{');
            }
            bytes.writeUTF(next.getMethodName());
            bytes.write(compactDescriptor(next.getDescriptor()));

        }


        public void print() {
            print(System.out);
        }
        public void print(PrintStream out) {
            out.println(bytes.toStringFromBytes());
            buffer./*reverseForEach*/forEach(out::println);
            out.println();
        }

         /** appends the values as annotation segment */
        public DynBytes append(Object... args) {
            if (args.length == 0)
                return this.bytes;


            //TODO add to buffer, with codepoints etc
            bytes.writeByte('{');
            for (Object x : args) {
                bytes.writeUTF(x.toString()); //HACK
                bytes.writeByte(',');
            }
            bytes.rewind(1);//.writeByte('{');

            return this.bytes;
        }
    }

    static boolean equalsClassAndMethod(StackWalker.StackFrame x, StackWalker.StackFrame y) {
        return x==y ||
                (x.getClassName().equals(y.getClassName()) &&
                        x.getMethodName().equals(y.getMethodName()) &&
                                x.getDescriptor().equals(y.getDescriptor()));
    }

//    static final Memoize<String,byte[]> compactMethodDescriptors = null; /* TODO *///new QuickMemoize<>(1024);

    static byte[] compactDescriptor(String _descriptor) {
        throw new TODO();
//        return compactMethodDescriptors.apply(_descriptor, (descriptor)->{
//            //TODO use byteseek automaton
//            String descriptor1 = descriptor;
//            descriptor1 = descriptor1.replace("java/lang/","");
//            if (descriptor1.endsWith("V")) //void return value
//                descriptor1 = descriptor1.substring(0, descriptor1.length()-1);
//            return descriptor1.getBytes();
//        });
    }

    /** shared by all cursors */
    protected static boolean frameFilter(StackWalker.StackFrame f) {
        return !f.getClassName().equals(thisClassName);
    }

    final MyRadixTree<Node> plan = new MyRadixTree<>();

    public Quality quality(String q) {
        return qualia.computeIfAbsent(q, qq -> {
            synchronized (qualia) {
                int id = qualia.size();
                if (id > 126)
                    throw new TODO("limit to 127 qualities");
                return new Quality(qq, (byte) id);
            }
        });
    }

    final ThreadLocal<StackCursor> cursors;

    public MetaFlow() {
        base = walkerSummary.walk(s->s.dropWhile(x -> {
            String cn = x.getClassName();
            return cn.equals(MetaFlow.class.getName()) || cn.startsWith(ThreadLocal.class.getName());
        })
                .findFirst().get());
        cursors = ThreadLocal.withInitial(()-> new StackCursor(base));
    }


    static final ThreadLocal<MetaFlow> meta = ThreadLocal.withInitial(MetaFlow::new);

    final Quality BAD = quality("bad").value(-1);
    final Quality GOOD = quality("good").value(+1);


    public static MetaFlow exe() {
        return meta.get();
    }

    public static StackCursor stack() {
        return exe().cursor();
    }

    /** gets thread-local cursor */
    public StackCursor cursor() {
        return cursors.get();
    }



//    public <X> X exe()

    public static void run(Class cl, String methodName) throws Exception {
        run(cl.getName(), methodName);
    }

    public static void run(String className, String methodName) throws Exception {

//        /*
//            Assume your application has a "home" directory
//            with /classes and /lib beneath it, where you can put
//            loose files and jars.
//
//            Thus,
//
//            /usr/local/src/APP
//            /usr/local/src/APP/classes
//            /usr/local/src/APP/lib
//         */
//
//        String HOME = "/usr/local/src/YOURAPP";
//        String CLASSES = HOME + "/classes";
//        String LIB = HOME + "/lib";
//
//        // add the classes dir and each jar in lib to a List of URLs.
//        List urls = new ArrayList();
//        urls.addAt(new File(CLASSES).toURL());
//        for (File f : new File(LIB).listFiles()) {
//            urls.addAt(f.toURL());
//        }
//
        URL[] urls = ClassPath.from(ClassLoader.getSystemClassLoader().getParent()).getAllClasses().stream().map(ClassPath.ResourceInfo::url).toArray(URL[]::new);
        // feed your URLs to a URLClassLoader!
        ClassLoader classloader = new URLClassLoader(urls, ClassLoader.getSystemClassLoader().getParent());

        // relative to that classloader, find the main class
        // you want to bootstrap, which is the first cmd line arg
        Class mainClass = classloader.loadClass(className);
        Method main = mainClass.getMethod(methodName
                /*new Class[]{args.getClass()}*/);

        Thread.currentThread().setContextClassLoader(classloader);

        main.invoke(null/*, new Object[] { nextArgs }*/);
    }

}
