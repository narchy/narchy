package nars.focus.util;

import jcog.data.list.Lst;
import jcog.pri.PLink;
import jcog.pri.bag.Sampler;
import jcog.random.RandomBits;
import nars.Deriver;
import nars.Focus;
import nars.NALTask;
import nars.control.DefaultBudget;
import nars.table.dynamic.TruthCurveBeliefTable;
import nars.utils.Profiler;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.util.function.Consumer;
import java.util.function.Function;

public class TaskBagAttentionSampler {
    private static final Logger logger = jcog.Log.log(TaskBagAttentionSampler.class);
    public final Lst<NALTask> tasks = new Lst<>();

    private final Consumer<NALTask> addTask = tasks::addFast;
    final Function<PLink<NALTask>, Sampler.SampleReaction> sample = this::sample;
    private final Deriver deriver;

    public TaskBagAttentionSampler(Deriver d) {
        this.deriver = d;
    }

    public Lst<NALTask> seed(Focus f, int iter, RandomBits rng) {
        long startTime = Profiler.startTime();
        logger.info("TaskBagAttentionSampler.seed called for focus {}, requested tasks: {}", f.id, iter);
        tasks.clear();
        tasks.ensureCapacity(iter);

        // Initialize samplesRemain and itersRemain before calling sample
        this.samplesRemain = iter; // Assuming initially we want to sample up to 'iter' tasks.
                                  // This might need adjustment based on how 'sample' is intended to limit.
                                  // If 'sample' itself is the primary loop controller for 'iter' times,
                                  // then this might be set differently or inside f.attn.sample.
                                  // For now, this interpretation makes sense for logging sampled count.
        this.itersRemain = iter; // Similar assumption for total iterations.

        f.attn.sample(this, iter, ((DefaultBudget) f.budget).puncSeed, rng);

        int sampledCount = tasks.size();
        logger.info("TaskBagAttentionSampler.seed for focus {} actually sampled: {} tasks", f.id, sampledCount);
        if (sampledCount > 0) {
            // Logging the list of tasks could be very verbose.
            // Consider a more summarized logging or conditional on TRACE level if full list is needed.
            logger.debug("Sampled tasks for focus {}: {}", f.id, tasks);
        }

        if (tasks.size() > 1) {
            tasks.shuffleThis(rng);
            //seeds.sortThisByFloat(z -> -z.priElseZero());
        }
        Profiler.recordTime("TaskBagAttentionSampler.seed", startTime);
        return tasks;
    }

    @Nullable
    private NALTask refine(NALTask x) {
        //return x;
        return x instanceof TruthCurveBeliefTable.CurveTask c ? c.task(deriver.focus.when(), deriver.nar) : x;
    }

    int samplesRemain, itersRemain;

    private Sampler.SampleReaction sample(PLink<NALTask> taskLink) {
        var task = taskLink.id;
        var deleted = task.isDeleted();

        var stop = --itersRemain <= 0;

        if (!deleted/* || acceptDeleted*/)
            stop = sample(task, stop);

        var remove = deleted /*|| pop*/;
        return Sampler.SampleReaction.the(remove, stop);
    }

    private boolean sample(NALTask task, boolean stop) {
        var t = refine(task);
        if (t!=null) {
            addTask.accept(t);
            if (--samplesRemain <= 0)
                stop = true;
        }
        return stop;
    }
}
