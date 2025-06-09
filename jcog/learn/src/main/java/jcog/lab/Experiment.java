package jcog.lab;

import jcog.Str;
import jcog.data.list.Lst;
import jcog.table.DataTable;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * the collected data associated with a subject's (X) execution of an experiment or episode
 *
 * the integration of a subject, a repeatable procedure, and measurement schema
 * <p>
 * contains:
 * -all or some of the Lab's sensors
 * -executable procedure for applying the starting conditions to the subject via
 * some or all of the variables
 * -executable schedule for recording sensor measurements, with at least
 * the start and ending state enabled by default. TODO
 */
public class Experiment<X>  {

    /**
     * data specific to this experiment; can be merged with multi-experiment
     * data collections later
     *
     * TODO abstract to a 'DataCollector' interface that can stream to other targets
     */
    public final DataTarget data;

    public interface DataTarget {
        void accept(Object[] row);

        void defineNumeric(String id);

        void defineLabel(String id);
    }

    public static class DataTableTarget implements DataTarget {

        final DataTable data;

        public DataTableTarget() {
            this(new DataTable());
        }

        public DataTableTarget(DataTable data) {
            this.data = data;
        }

        @Override
        public void accept(Object[] row) {
            this.data.add(row);
        }

        @Override
        public void defineNumeric(String id) {
            this.data.defineNumeric(id);
        }

        @Override
        public void defineLabel(String id) {
            this.data.defineText(id);
        }
    }

    public static class ConsoleTarget extends DataTableTarget {

        public ConsoleTarget() {
            super();
        }

        @Override
        public void accept(Object[] row) {
            //System.out.println(Arrays.toString(row));
            var sb = new StringBuilder(1024);
            for (var i = 0; i < row.length; i++) {
                sb.append(data.column(i).name()).append("=").append(row[i]).append(' ');
            }
            System.out.println(sb);
        }

    }
    public static class CSV extends DataTableTarget {

        static final int flushPeriod = 64;

        private final PrintStream out;
        private final char delimeter;
        int rows = 0;

        /**
         * @param delimeter '\t' (tab) or ',' (comma)
         */
        public CSV(OutputStream o, char delimeter) {
            super();
            this.out = new PrintStream(o);
            this.delimeter = delimeter;
        }

        @Override
        public void accept(Object[] row) {
            if (rows++==0)
                printHeader();
            printRow(row);

            if (rows%flushPeriod==0) out.flush();
        }

        private void printRow(Object[] row) {
            var cols = row.length;
            for (var i = 0; i < cols; i++) {
                printItem(row[i]);
                if (i < cols-1)
                    out.append(delimeter);
            }
            out.append('\n');
        }

        private void printItem(Object o) {
            out.append(switch (o) {
                case Integer i -> Integer.toString(i);
                case Long l -> Long.toString(l);
                case Float f -> Float.toString(f);
                case Double d -> Double.toString(d);
                case String s -> Str.quote(s);
                case null, default -> throw new UnsupportedOperationException();
            });
        }

        private void printHeader() {
            var cols = data.columnCount();
            for (var i = 0; i < cols; i++) {
                var name = data.column(i).name();
                out.append(
                    name
                    //Str.quote(name)
                    //name.replaceAll("\"","'")
                );
                if (i < cols-1)
                    out.append('\t');
            }
            out.append('\n');
        }

    }

    @Deprecated private final BiConsumer<X, Experiment<X>> procedure;

    private final List<Sensor<X, ?>> sensors = new Lst<>();

    private final Supplier<X> subjectBuilder;

    public Experiment(X subject, DataTarget data) {
        this(()->subject, data, (x, e)-> { }, List.of());
    }

    public Experiment(Supplier<X> subjectBuilder, DataTarget data, BiConsumer<X, Experiment<X>> proc, Iterable<Sensor<X,?>> sensors) {
        this.subjectBuilder = subjectBuilder;
        this.procedure = proc;
        this.data = data;
        for (var s : sensors)
            sense(s);
        for (var s : this.sensors)
            s.register(data);

    }

    public synchronized final Experiment<X> sense(Sensor<X,?> s) {
        sensors.add(s);
        return this;
    }

    public void run() {
//        long startTime = System.currentTimeMillis();

        try {
            procedure.accept(subjectBuilder.get(), this);
        } catch (RuntimeException t) {
            //sense(t.getMessage());
            t.printStackTrace(); //TODO
        }

//        long endTime = System.currentTimeMillis();
//        if (data != null) ((ARFF) data).setComment(subject + ": " + procedure +
//                "\t@" + startTime + ".." + endTime + " (" + new Date(startTime) + " .. " + new Date(endTime) + ')');
    }

    public void record(X subject) {
        synchronized (data) {
            data.accept(Lab.row(subject, sensors));
        }
    }

    public void runSerial(int iters) {
        for (var i = 0; i < iters; i++)
            run();
    }

    public void runParallel(int iters/*, int threads*/) {
        IntStream.range(0, iters).parallel().forEach(z -> run());
    }

}