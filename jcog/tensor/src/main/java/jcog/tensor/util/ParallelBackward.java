package jcog.tensor.util;

import jcog.data.list.Lst;
import jcog.tensor.Tensor;
import jcog.tensor.Tensor.GradQueue;
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Deque;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import static jcog.tensor.Tensor.array;

public class ParallelBackward {

    private static final Collector<AccumulateTask, ?, Map<Integer, Lst<AccumulateTask>>> groupingCollector =
            Collectors.groupingBy(t -> t.tensor.op.color, Collectors.toCollection(Lst::new));
    private final static int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

    public static Iterable<Tensor> backward(Tensor root) {
        var grads = new GradQueue();
        var newTasks = new ConcurrentLinkedDeque<AccumulateTask>();

        gradAccum(root, SimpleMatrix.ones(root.rows(), root.cols()), grads, newTasks);

        color(root);

        var tasks = new Lst<AccumulateTask>();
        while (!newTasks.isEmpty()) {
            tasks.addAll(newTasks);
            newTasks.clear();

            if (tasks.size() == 1) {
                tasks.getFirst().backward(grads, newTasks);
            } else {
                var groups = groupByColor(tasks);
                if (groups.length == 1)
                    backward(groups[0], grads, newTasks);
                else
                    Arrays.stream(groups).parallel().unordered()
                            .forEach(t -> backward(t, grads, newTasks));
            }
            tasks.clear();
        }

        grads.optimize(null);

        return grads;
    }

    private static void color(Tensor root) {
        root.recurse(r -> {
            if (r.op != null) r.op.color = -1;
        }); //reset
        AccumulateTask.color(root, null);
    }

    private static void backward(Lst<AccumulateTask> tt, GradQueue grads, Deque<AccumulateTask> newTasks) {
        for (var t : tt)
            t.backward(grads, newTasks);
    }

    private static Lst<AccumulateTask>[] groupByColor(Lst<AccumulateTask> tasks) {
        return tasks.stream().collect(groupingCollector).values().toArray(Lst[]::new);
    }

    private static void gradAccum(Tensor x, SimpleMatrix y,
                                  GradQueue grads, Deque<AccumulateTask> newTasks) {
        if (!x.sameShape(y))
            throw new UnsupportedOperationException();

        grads.addGrad(x, array(y));

        if (x.op != null)
            newTasks.add(new AccumulateTask(x, y));
    }


    private record AccumulateTask(Tensor tensor, SimpleMatrix gradient) {

        private static void color(Tensor tensor, @Nullable Tensor child) {
            var o = tensor.op;
            if (o == null)
                return;

            // Color this tensor first
            o.color = child != null ? Math.max(o.color, child.op.color + 1) : 0;

            // Then recurse on parents
            for (var parent : o.parents)
                color(parent, tensor);
        }

        private void backward(GradQueue grads, Deque<AccumulateTask> newTasks) {
            var op = tensor.op;
            var gradOut = op.allocateParentGrads();

            op.backward(gradient, gradOut);

            var parents = op.parents;
            for (int i = 0; i < parents.length; i++) {
                var gi = gradOut[i];
                if (gi != null) {
                    gradAccum(parents[i], gi, grads, newTasks);
                    gradOut[i] = null; //GC help?
                }
            }
        }
    }
}