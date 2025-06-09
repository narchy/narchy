package nars;

import jcog.Log;
import nars.concept.util.ConceptBuilder;
import nars.concept.util.DefaultConceptBuilder;
import nars.derive.Deriver;
import nars.derive.impl.TaskBagDeriver;
import nars.derive.reaction.ReactionModel;
import nars.derive.reaction.Reactions;
import nars.focus.Focus;
import nars.memory.CaffeineMemory;
import nars.memory.Memory;
import nars.memory.SimpleMemory;
import nars.time.Time;
import nars.time.clock.CycleTime;
import nars.time.clock.RealTime;
import org.slf4j.Logger;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * NAR builder
 */
public class NARS {

    public static final Logger log = Log.log(NARS.class);

    protected Supplier<Memory> index;

    protected Time time;

    @Deprecated protected final Function<Term, Focus> focus;

    protected Supplier<Random> rng;

    protected Supplier<ConceptBuilder> conceptBuilder;

    public final NAR get(Consumer<NAR> init) {
        NAR n = get();
        init.accept(n);
        return n;
    }

    public final NAR get() {
        NAR n = new NAR(
            index.get(),
            time,
            rng,
            conceptBuilder.get()
        );

        Term self = n.self();

        n.add(n.main = focus.apply(self)); //HACK

        if (log.isInfoEnabled()) log.info("init {} {}", self, System.identityHashCode(n));

        step.forEach((a,p)->{
            log.debug("+ {} {}", self, a);
            p.accept(n);
        });
        return n;
    }

    /**
     * applied in sequence as final step before returning the NAR
     */
    private final Map<String,Consumer<NAR>> step = new LinkedHashMap<>();

    public NARS memory(Memory concepts) {
        this.index = () -> concepts;
        return this;
    }

    public NARS time(Time time) {
        this.time = time;
        return this;
    }

    public NARS concepts(ConceptBuilder cb) {
        this.conceptBuilder = () -> cb;
        return this;
    }

    /**
     * adds a deriver with the standard rules for the given range (inclusive) of NAL levels
     */
    @Deprecated public NARS withNAL(int minLevel, int maxLevel, String... extra) {
        return then("nal", n -> deriver(Derivers.nal(
                minLevel, maxLevel, extra).core().stm().temporalInduction(), n)
            .everyCycle(n.main()));
    }

    public Deriver deriver(Reactions m, NAR n) {
        return deriver(m.compile(n), n);
    }

    protected Deriver deriver(ReactionModel m, NAR n) {
        return new TaskBagDeriver(m, n);
        //return new QueueDeriver(m, n);
        //return new FairDeriver(m, n);
        //return new SerialDeriver(m, n);
        //return new CachedSerialDeriver(m, n, 8 * 1024);
        //return new EqualizerDeriver(m, n);
        //return new MixDeriver(m, n);
    }

    /**
     * generic defaults
     */
    @Deprecated
    public static class DefaultNAR extends NARS {

        public DefaultNAR(int nal, boolean threadSafe) {
            this(0, nal, threadSafe);
        }

        public DefaultNAR(int nalMin, int nalMax, boolean threadSafe) {
            super();
            assert(nalMin <= nalMax);

            if (threadSafe)
                index = () -> new CaffeineMemory(64 * 1024);

            if (nalMax > 0)
                withNAL(nalMin, nalMax);

            then("misc", n-> n.complexMax.set(16));
        }

    }



    public NARS() {

        index = () ->
            new SimpleMemory(1 * 1024)
        ;

        time = new CycleTime();

        focus = w -> new Focus(w, 64);

        rng = ThreadLocalRandom::current;

        conceptBuilder = DefaultConceptBuilder::new;
    }

    /**
     * temporary, disposable NAR. safe for single-thread access only.
     * full NAL8 with STM Linkage
     */
    public static NAR tmp() {
        return tmp(8);
    }


    /**
     * temporary, disposable NAR. useful for unit tests or embedded components
     * safe for single-thread access only.
     *
     * @param nal adjustable NAL level. level >= 7 include STM (short-target-memory) Linkage plugin
     */
    public static NAR tmp(int nal) {
        return tmp(0, nal);
    }

    public static NAR tmp(int nalStart, int nalEnd) {
        return new DefaultNAR(nalStart, nalEnd, false).get();
    }

    /**
     * single thread but for multithread usage:
     * unbounded soft reference index
     */
    @Deprecated public static NAR threadSafe() {
        return threadSafe(8);
    }

    @Deprecated private static NAR threadSafe(int level) {
        NARS d = new DefaultNAR(level, true)
                .time(new RealTime.CS().durFPS(25.0f));

        d.rng = ThreadLocalRandom::current;

        return d.get();
    }


    /** @param durFPS milliseconds realtime.  <=0: infinite */
    public static NARS realtime(float durFPS) {
        DefaultNAR d = new DefaultNAR(0, true);
        return durFPS > 0 ? d.time(new RealTime.MS().durFPS(durFPS)) : d;
    }

    /**
     * provides only low level functionality.
     * an empty deriver, but allows any kind of target
     */
    public static NAR shell() {
        return tmp(0);
    }


    public NARS then(String what ,Consumer<NAR> action) {
        step.put(what, action);
        return this;
    }

}