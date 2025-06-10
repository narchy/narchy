package nars;


import com.google.common.primitives.Longs;
import com.google.common.util.concurrent.MoreExecutors;
import jcog.Log;
import jcog.Str;
import jcog.Util;
import jcog.data.byt.DynBytes;
import jcog.data.map.ObjIntHashMap;
import jcog.event.ByteTopic;
import jcog.event.ListTopic;
import jcog.event.Off;
import jcog.exe.Cycled;
import jcog.exe.InstrumentedLoop;
import jcog.exe.Loop;
import nars.utils.Profiler;
import jcog.exe.SchedExecutor;
import jcog.func.TriConsumer;
import jcog.signal.FloatRange;
import jcog.signal.meter.Metered;
import jcog.thing.Part;
import nars.Narsese.NarseseException;
import nars.concept.NodeConcept;
import nars.concept.Operator;
import nars.concept.PermanentConcept;
import nars.concept.TaskConcept;
import nars.concept.util.ConceptBuilder;
import nars.control.*;
import nars.eval.Evaluator;
import nars.eval.TaskEvaluation;
import nars.focus.util.*;
import nars.io.IO;
import nars.link.MutableTaskLink;
import nars.memory.Memory;
import nars.table.question.QuestionTable;
import nars.table.util.DynTables;
import nars.task.proxy.SpecialNegTask;
import nars.task.util.TaskException;
import nars.term.Compound;
import nars.term.Functor;
import nars.term.Neg;
import nars.term.Termed;
import nars.term.atom.Atom;
import nars.term.atom.Atomic;
import nars.term.util.TermTransformException;
import nars.time.Every;
import nars.time.ScheduledTask;
import nars.time.Tense;
import nars.time.Time;
import nars.time.event.WhenTimeIs;
import nars.time.part.DurLoop;
import nars.time.part.DurNARConsumer;
import nars.truth.proj.TruthProjection;
import nars.util.NARPart;
import org.HdrHistogram.Histogram;
import jcog.TODO;
import jcog.Util;
import jcog.event.Topic;
import jcog.memoize.MemoGraph;
import jcog.pri.PLink;
import jcog.pri.PriReference;
import jcog.pri.bag.Bag;
import jcog.pri.bag.impl.ArrayBag;
import jcog.pri.op.PriMerge;
import jcog.random.RandomBits;
import nars.action.link.index.BagAdjacentTerms;
import nars.focus.BagForget;
import nars.focus.BasicTimeFocus;
import nars.focus.PriSource;
import nars.focus.TimeFocus;
import nars.task.CommandTask;
import nars.task.util.PuncBag;
import org.eclipse.collections.impl.map.mutable.primitive.ByteIntHashMap;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.function.*;
import java.util.stream.Stream;
import java.util.zip.GZIPOutputStream;

import static nars.$.$;
import static nars.$.$$;
import static nars.Op.*;


import org.HdrHistogram.LongHistogram;

public final class NAR extends NAL<NAR> implements Consumer<Task>, Cycled {
	private final Term self;
	private static final Logger logger = Log.log(NAR.class);
	public final Memory memory;


	public final SchedExecutor exe;

    public final NARLoop loop;

	public final Emotion emotion = new Emotion();

	public final Causes causes;
	public final PriTree pri;

    public final Budget budget;
    public Attn narAttn;
    public final ByteTopic<Task> taskEventTopic = new ByteTopic<>(Op.Punctuation);
    public final Function<Bag, Consumer<Prioritizable>> attnUpdater = new BagForget();


	public final ListTopic<NAR> eventClear = new ListTopic<>(), eventCycle = new ListTopic<>();

	private static final int DEFAULT_EVALUATOR_CACHE_CAPACITY = 64 * 1024;

	public final Evaluator evaluator =
		new nars.eval.CachedEvaluator(this::axioms, DEFAULT_EVALUATOR_CACHE_CAPACITY );


    private volatile long prevCycleTime, currentCycleTime, commitNextCycleTime, novelThresholdTime;
    private final FloatRange commitDurs = new FloatRange(1, 0.5f, 16);
    public TimeFocus timeFocus = new BasicTimeFocus();
    public float durSys = 1;


	public NAR(Memory m, Time t, Supplier<Random> rng, ConceptBuilder conceptBuilder, boolean asyncLoop) {
		super(t, rng);

		this.self = randomSelf();

		this.memory = m;

		conceptBuilder.init(emotion, t);

		this.exe = new SchedExecutor(MoreExecutors.newDirectExecutorService());

		m.start(this, conceptBuilder);

        this.budget = new DefaultBudget();
        this.narAttn = new TaskBagAttn(this);


		this.causes = new Causes();
		this.pri = new PriTree();

		onDur(emotion);
		onDur(pri::commit);

		NARS.Functors.init(this);

		this.loop = new NARLoop(asyncLoop);

        this.currentCycleTime = time();
        this.prevCycleTime = this.currentCycleTime;
        commitTime(this.currentCycleTime);

	}


    private void commitTime(long now) {
        this.prevCycleTime = this.currentCycleTime;
        this.currentCycleTime = now;
        this.durSys = this.time.dur();

        var durCommit = Math.max(1, durSys * commitDurs.floatValue());
        this.commitNextCycleTime = now + Math.round(durCommit);

        float durNovel = durCommit;
        long cycNovel = Math.max(1, Tense.occToDT(Math.round(durNovel * NAL.belief.NOVEL_DURS)));
        novelThresholdTime = now - cycNovel;
    }

    public boolean isNovel(NALTask x) {
        long creation;
        if (NAL.TASK_ACTIVATE_ALWAYS || ((creation = x.creation()) == TIMELESS || creation <= novelThresholdTime)) {
            x.setCreation(time());
            return true;
        } else
            return false;
    }


	@Nullable public Term conceptTerm(Termed X, boolean taskConceptOnly) {
		if (X instanceof Concept)
			return (taskConceptOnly && !(X instanceof TaskConcept)) ? null : X.term();

        var x = X.term();

		if (!NAL.term.CONCEPTUALIZE_OVER_VOLUME && x.complexity() > complexMax())
			return null;

        var z = x.concept();

		return taskConceptOnly && !NALTask.TASKS(z) ? null : z;
	}

	public static void proofPrint(NALTask t) {
		proofAppend(t, System.out);
	}

	public static void proofAppend(NALTask t, Appendable o) {
		try {
			o.append(t.proof()).append('\n');
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public final Functor axioms(Atom term) {
		Termed x = concept(term);
		if (x instanceof NodeConcept.FunctorConcept)
			x = x.term();
		return x instanceof Functor f ? f : null;
	}


	public static class SnapshotStats {
		public final LongSummaryStatistics beliefs = new LongSummaryStatistics();
		public final LongSummaryStatistics goals = new LongSummaryStatistics();
		public final LongSummaryStatistics questions = new LongSummaryStatistics();
		public final LongSummaryStatistics quests = new LongSummaryStatistics();
		public final ObjIntHashMap<Class<?>> clazz = new ObjIntHashMap<>();
		public final ByteIntHashMap rootOp = new ByteIntHashMap();
		public final Histogram volume = new Histogram(1, term.COMPOUND_VOLUME_MAX, 3);
	}

	public synchronized SortedMap<String, Object> stats(boolean concepts, boolean emotions) {
		SortedMap<String, Object> x = new TreeMap<>();

        var now = time();

		if (concepts) {
			SnapshotStats currentStats = new SnapshotStats();

			concepts().forEach(c -> {

                var ct = c.term();
				currentStats.volume.recordValue(ct.complexity());
				currentStats.rootOp.addToValue(ct.opID(), 1);
				currentStats.clazz.increment(c.getClass());

				currentStats.beliefs.accept(c.beliefs().taskCount());
				currentStats.goals.accept(c.goals().taskCount());
				currentStats.questions.accept(c.questions().taskCount());
				currentStats.quests.accept(c.quests().taskCount());
			});


			if (loop.isRunning())
				loop.stats("loop", x);

			x.put("time", now);
			x.put("concept count", memory.size());

			x.put("belief count", (double) currentStats.beliefs.getSum());
			x.put("goal count", (double) currentStats.goals.getSum());

			currentStats.rootOp.forEachKeyValue((opID, count) -> x.put("concept op " + op(opID), count));

			Str.histogramDecodeExact(currentStats.volume, "concept volume", 4, x::put);

			var uniqueConceptClasses = new int[]{0};
			currentStats.clazz.forEachKeyValue((c, count) -> {
				if (count > 1)
					x.put("concept class " + c, count);
				else {
					uniqueConceptClasses[0]++;
				}
			});
			if (uniqueConceptClasses[0] > 0)
				x.put("concept class unique count", uniqueConceptClasses[0]);

		}

		if (emotions) {
			new Metered.MeterReader() {
				@Override public void set(Object value) {
					x.put(metric, value);
				}
			}.run(now, emotion.fields);
		}

		return x;

	}


	public void delete() {
		logger.debug("delete {}", self());
		synchronized (exe) {

			stop();

            if (narAttn != null) narAttn.clear();


			memory.clear();

			super.delete();

			eventCycle.clear();
			eventClear.clear();

			exe.shutdownNow();
		}
	}


	public final <T extends Task> T inputTask(String taskText) throws Narsese.NarseseException {
		T x = Narsese.task(taskText, this);
		input(x);
		return x;
	}

	public List<Task> input(String text) throws NarseseException, TaskException {
		var l = Narsese.tasks(text, this);
		switch (l.size()) {
			case 0 -> l = Collections.EMPTY_LIST;
			case 1 -> input(l.getFirst());
			default -> input(l);
		}
		return l;
	}

	public final @Nullable Concept conceptualize(String conceptTerm) throws NarseseException {
		return conceptualize($(conceptTerm));
	}


	public final void input(Task t) {
        logger.info("Input task: {}, type: {}, content: {}", t, t.getClass().getSimpleName(), t.term());
        Profiler.incrementCounter("NAR.tasksInput");
        long profilerStartTime = Profiler.startTime();
        try {

            if (t instanceof NALTask X) {
                if (isNovel(X)) {
                    narAttn.activate(X); // Assuming narAttn.activate handles its own profiling/logging if needed
                    taskEventTopic.emit(t, t.punc());
                }
            } else if (t instanceof CommandTask C) {
                runCommand(C);
                taskEventTopic.emit(t, t.punc());
            } else {
                logger.warn("Unknown task type received in NAR.input: {}", t.getClass().getName());
                throw new UnsupportedOperationException("Unknown task type: " + t.getClass());
            }
        } finally {
            Profiler.recordTime("NAR.inputTaskDuration", profilerStartTime); // Renamed for clarity from NAR.inputTask
            // Counters below are fine, but the main timer for the method is inputTaskDuration
            if (t instanceof NALTask) {
                Profiler.incrementCounter("NAR.inputTask.NALTask.calls");
            } else if (t instanceof CommandTask) {
                Profiler.incrementCounter("NAR.inputTask.CommandTask.calls");
            }
        }
	}

    private void runCommand(CommandTask t) {
        if (!Functor.isFunc(t.term()))
            throw new UnsupportedOperationException("Command task term is not a functor: " + t.term());
        new CommandEvaluation(t);
    }

    private class CommandEvaluation extends TaskEvaluation<Task> {
        private final CommandTask task;

        CommandEvaluation(CommandTask t) {
            super(new Evaluator(NAR.this::axioms));
            this.task = t;
            eval(t.term());
        }

        @Override public CommandTask task() { return task; }
        @Override public NAR nar() { return NAR.this; }
        @Override public void accept(Task x) { NAR.this.input(x); }
        @Override public boolean test(Term term) { return false; }
    }



	@Override
	public final void accept(Task task) {
		input(task);
	}

	@Override
	public final @Nullable Term term(Part<NAR> p) {
		return ((NARPart) p).id;
	}

	public final boolean add(NARPart p) {
		if (p.nar!=this && p.nar!=null)
			throw new UnsupportedOperationException();

		p.nar = this;

		return add(p.id, p, true);
	}

	public final boolean add(NARPart p, boolean autoStart) {
		return add(p.id, p, autoStart);
	}

	public final void add(Functor f) {
		memory.set(f);
	}

	public final NARPart add(Class<? extends NARPart> p, boolean autoStart) {
		return add(null, p, autoStart);
	}

	public final NARPart add(Class<? extends NARPart> p) {
		return add(p, true);
	}

	public final NARPart add(@Nullable Term key, Class<? extends NARPart> p) {
		return add(key, p, true);
	}

	public final NARPart add(@Nullable Term key, Class<? extends NARPart> p, boolean autoStart) {
		NARPart pp = null;
		if (key != null)
			pp = (NARPart) parts.get(key);

		if (pp == null) {
			pp = build(p).get();
			if (key == null)
				key = pp.id;
			if (parts.get(key) == pp) {
				if (autoStart && !pp.isStarted()) pp.start();
				return pp;
			}
		} else {
			if (p.isAssignableFrom(pp.getClass())) {
				if (autoStart && !pp.isStarted()) pp.start();
				return pp;
			} else {
				remove(key);
			}
		}


		if (pp.nar == null) pp.nar = this;


		if (autoStart) {
            if (!add(key, pp)) {
				NARPart currentInMap = (NARPart) parts.get(key);
				if (currentInMap != null && p.isAssignableFrom(currentInMap.getClass())) {
					if (!currentInMap.isStarted()) currentInMap.start();
					return currentInMap;
				} else if (currentInMap != null) {
                    logger.warn("NARPart type conflict for key {}: expected {}, but found {}. Returning existing.",
                            key, p.getName(), currentInMap.getClass().getName());
                    if (!currentInMap.isStarted()) currentInMap.start();
					return currentInMap;
				}
			}
		} else {
			Part<?> current = parts.putIfAbsent(key, pp);
			if (current != null && current != pp) {
				if (p.isAssignableFrom(current.getClass())) {
					return (NARPart) current;
				} else {
                    logger.warn("NARPart type conflict for key {}: expected {}, but found {}. Returning existing.",
                            key, p.getName(), current.getClass().getName());
					return (NARPart) current;
				}
			}
		}
		return pp;
	}


	public final Functor add(Atom name, BiFunction<Task, NAR, Task> exe) {
        var f = Operator.simple(name, exe);
		add(f);
		return f;
	}



	public final @Nullable Truth truth(Termed concept, boolean beliefOrGoal, long start, long end, float dur) {
        var neg = concept instanceof Term && concept.term() instanceof Neg;
		if (neg)
			concept = ((Term)concept).unneg();

		@Nullable BeliefTable table = table(concept, beliefOrGoal);
		if (table != null) {
            var tt = table.truth(start, end, concept instanceof Term t ? t : concept.term(), null, dur, this);
			if (tt != null)
				return neg ? tt.neg() : tt;
		}
		return null;
	}


	public final @Nullable Truth beliefTruth(Termed concept, long start, long end, float dur) {
		return truth(concept, true, start, end, dur);
	}
	public final @Nullable Truth goalTruth(Termed concept, long start, long end, float dur) {
		return truth(concept, false, start, end, dur);
	}

	public final NAR stop() {

		loop.stop();

		synchronized (exe) {
			synch();
		}

		return this;
	}

	public final void pause() {
		loop.stop();
	}


	public AutoCloseable log() {
        return logTo(System.out, null);
	}

	public AutoCloseable log(Appendable out) {
        return logTo(out, null, null);
	}

    public Off logTo(Appendable out, @Nullable Predicate<String> includeKey, @Nullable Predicate<?> includeValue) {
        return taskEventTopic.on(new TaskLogger(includeValue, out, includeKey));
    }

    static class TaskLogger implements Consumer<Task> {
        private final @Nullable Predicate<?> includeValue;
        private final Appendable out;
        String previous;

        TaskLogger(@Nullable Predicate<?> includeValue, Appendable out, @Nullable Predicate<String> includeKey) {
            this.includeValue = includeValue;
            this.out = out;
            previous = null;
            Topic.all(this, (k, v) -> {
                if (includeValue == null || includeValue.test(v)) {
                    outputEvent(out, previous, k, v);
                    previous = k;
                }
            }, includeKey);
        }

        private static void outputEvent(Appendable out, String previou, String chan, Object v) {
            try {
                if (!chan.equals(previou)) {
                    out.append(chan).append(": ");
                } else {
                    int n = chan.length() + 2;
                    for (int i = 0; i < n; i++)
                        out.append(' ');
                }

                if (v instanceof Object[] va) {
                    v = Arrays.toString(va);
                } else if (v instanceof Task tv) {
                    v = tv.toString(true);
                }
                out.append(v.toString()).append('\n');
            } catch (IOException e) {
                NAR.logger.error("outputEvent", e);
            }
        }
        @Override
        public void accept(Task v) {
            if (includeValue == null || includeValue.test(v)) {
                outputEvent(out, previous, "task", v);
                previous = "task";
            }
        }
    }

	@Override
	public final NARLoop startPeriodMS(int periodMS) {
		loop.startPeriodMS(periodMS);
		return loop;
	}

	@Override
	public void run() {
		synchronized (loop) {
			loop.next();
		}
	}

	public NAR input(String... ss) throws NarseseException {
		for (var s : ss)
			input(s);
		return this;
	}

	public NAR inputNarsese(URL url) throws IOException, NarseseException {
		try (var s = url.openStream()) {
			return inputNarsese(s);
		}
	}

	public NAR inputNarsese(InputStream inputStream) throws IOException, NarseseException {
		input(new String(inputStream.readAllBytes()));
		return this;
	}

	public NAR inputAt(long time, String... tt) {

		runAt(time, nn -> {
			try {
				nn.input(tt);
			} catch (NarseseException e) {
				e.printStackTrace();
			}
		});

		return this;
	}

	public void inputAt(long when, Task... x) {
		runAt(when, nn -> {
            for (Task task : x) {
                nn.input(task);
            }
        });
	}

	public final void runAt(long whenOrAfter, Consumer<NAR> t) {
		if (time() >= whenOrAfter)
			t.accept(this);
		else
			runAt(WhenTimeIs.then(whenOrAfter, t));
	}

	public final ScheduledTask runAt(long whenOrAfter, Runnable t) {
		return runAt(WhenTimeIs.then(whenOrAfter, t));
	}

	public final ScheduledTask runAt(ScheduledTask t) {
		exe.runAt(()->t.accept(this), t.next);
		return t;
	}

	public final void runLater(Runnable t) {
		runAt(WhenTimeIs.then(time(), t));
	}

	public Stream<NALTask> tasks(boolean beliefs, boolean questions,
							  boolean goals, boolean quests) {
		return concepts().flatMap(c ->
			c.tasks(beliefs, questions, goals, quests));
	}

	public void tasks(boolean includeConceptBeliefs, boolean includeConceptQuestions, boolean includeConceptGoals,
					  boolean includeConceptQuests, BiConsumer<Concept, Task> each) {
		concepts().forEach(c ->
			c.tasks(includeConceptBeliefs,
				includeConceptQuestions,
				includeConceptGoals,
				includeConceptQuests).forEach(t -> each.accept(c, t))
		);
	}

	public Stream<NALTask> tasks() {
		return tasks(true, true, true, true);
	}

	public final @Nullable Concept concept(Termed x) {
		return concept(x, false);
	}

	public final @Nullable TaskConcept concept(NALTask x) {
		return (TaskConcept)concept((Termed)x);
	}
	public final @Nullable TaskConcept conceptualize(NALTask x) {
		return (TaskConcept)conceptualize((Termed)x);
	}

	public final @Nullable Concept concept(String term) {
		return concept($$(term), false);
	}

	public final @Nullable Concept conceptualize(Termed termed) {
		return concept(termed, true);
	}

	public final @Nullable Concept concept(Termed _x, boolean createIfMissing) {
        long profilerStartTime = Profiler.startTime();
        Concept result = null;
        try {
            result = _x instanceof Concept xConcept && memory.elideConceptGets() && !xConcept.isDeleted() ?
                    xConcept :
                    _conceptualize(conceptTerm(_x, false), createIfMissing);
            return result;
        } finally {
            if (createIfMissing) {
                Profiler.recordTime("NAR.concept.wrapper.createIfMissing", profilerStartTime);
                Profiler.incrementCounter("NAR.concept.wrapper.createIfMissing.calls");
            } else {
                Profiler.recordTime("NAR.concept.wrapper.presentOnly", profilerStartTime);
                Profiler.incrementCounter("NAR.concept.wrapper.presentOnly.calls");
            }
        }
	}

	private Concept _conceptualize(@Nullable Term x, boolean createIfMissing) {
        long profilerStartTime = Profiler.startTime();
        try {
            return x != null ? memory.get(x, createIfMissing) : null;
        } finally {
            if (createIfMissing) {
                Profiler.recordTime("NAR._conceptualize.createIfMissing", profilerStartTime);
                Profiler.incrementCounter("NAR._conceptualize.createIfMissing.calls");
            } else {
                Profiler.recordTime("NAR._conceptualize.presentOnly", profilerStartTime);
                Profiler.incrementCounter("NAR._conceptualize.presentOnly.calls");
            }
        }
	}

	public final Stream<Concept> concepts() {
		return memory.stream();
	}

	public final Off onCycle(Consumer<NAR> each) {
		return eventCycle.on(each);
	}

	public final Off onCycle(Runnable each) {
		return onCycle(ignored -> each.run());
	}

	public final DurLoop onDur(Runnable on, float durPeriod) {
		return onDur(on).durs(durPeriod);
	}

	public final DurLoop onDur(Runnable on) {
		var r = new DurLoop.DurRunnable(on);
		add(r);
		return r;
	}

	public final DurLoop onDur(Consumer<NAR> on) {
        var r = new DurNARConsumer(on);
		add(r);
		return r;
	}

	public void input(Iterable<? extends Task> tasks) {
        for (Task task : tasks) {
            input(task);
        }
	}

	public final void input(Stream<? extends Task> tasks) {
        tasks.forEach(this::input);
	}

	@Override
	public final boolean equals(Object obj) {
		return this == obj;
	}

	public NAR believe(Term c, Tense tense) {
		believe(c, tense, 1.0f, confDefault(BELIEF));
		return this;
	}

	public final <P extends Concept & PermanentConcept> P add(P c) {

		memory.set(c);

		memory.conceptBuilder.start(c);

		return c;
	}



	public final NAR outputBinary(File f, boolean append, Predicate<Task> each) throws IOException {
		return outputBinary(f, append, (Task t) -> each.test(t) ? t : null);
	}

	public NAR outputBinary(File f, boolean append, UnaryOperator<Task> each) throws IOException {
        var f1 = new FileOutputStream(f, append);
		OutputStream ff = new GZIPOutputStream(f1, IO.gzipWindowBytesDefault);
		outputBinary(ff, each);
		return this;
	}


	public final NAR outputBinary(OutputStream o, Predicate<Task> filter) {
		return outputBinary(o, (Task x) -> filter.test(x) ? x : null);
	}

	public NAR outputBinary(OutputStream o, UnaryOperator<Task> each) {

		Util.time("outputBinary", logger, () -> {

            var oo = new DataOutputStream(o);

            var total = new int[]{0};
            var wrote = new int[]{0};

            var d = new DynBytes(128);

			tasks().map(each).filter(Objects::nonNull).distinct().forEach(x -> {

				total[0]++;

				try {
                    var b = IO.taskToBytes(x, d);
					oo.write(b);
					wrote[0]++;
				} catch (IOException e) {
					if (DEBUG)
						throw new RuntimeException(e);
					else
						logger.warn("output binary", e);
				}

			});

			logger.info("{} output {}/{} tasks ({} bytes uncompressed)", o, wrote[0], total[0], oo.size());


			try {
				oo.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		});

		return this;
	}

	public NAR outputText(OutputStream o, UnaryOperator<Task> each) {


        var ps = new PrintStream(o);


        var sb = new StringBuilder();
        var t = tasks().map(each).filter(Objects::nonNull);

        var sortByEvi = true;
		if (sortByEvi)
			t = t.sorted(Comparator.comparingDouble((Task z) -> -(z.BELIEF_OR_GOAL() ? ((NALTask)z).evi() : -(1))));

		t.forEach(x -> {

			sb.setLength(0);
			ps.println(x.appendTo(sb, true));
		});

		try {
			o.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return this;
	}

	public NAR output(File o, boolean binary) throws FileNotFoundException {
		return output(new BufferedOutputStream(new FileOutputStream(o), IO.outputBufferBytesDefault), binary);
	}

	public NAR output(File o, UnaryOperator<Task> f) throws FileNotFoundException {
		return outputBinary(new BufferedOutputStream(new FileOutputStream(o), IO.outputBufferBytesDefault), f);
	}

	public NAR output(OutputStream o, boolean binary) {
		return binary ?
				outputBinary(o,
						(Task x) -> x.isDeleted() ? null : x) :
				outputText(o, x -> x.isDeleted() ? null : x);
	}

	public NAR inputBinary(InputStream i) throws IOException {

        IO.tasksStream(i).forEach(this::input);


		return this;
	}

	public final Term self() {
		return self;
	}

	public @Nullable NALTask belief(Termed x, long start, long end) {
		return answer(x, BELIEF, start, end);
	}

	public final NALTask belief(String x, long when) throws NarseseException {
		return belief($(x), when);
	}

	public final NALTask belief(Termed x, long when) {
		return belief(x, when, when);
	}

	public final NALTask belief(Term x) {
		return belief(x, ETERNAL);
	}

	public final NALTask belief(String x) throws NarseseException {
		return belief($(x), ETERNAL);
	}

	public final @Nullable NALTask answer(Term x, byte punc, long when) {
		return answer(x, punc, when, when);
	}

	public final @Nullable NALTask answer(Termed X, byte punc, long start, long end) {
        return answer(X, punc, start, end, dur());
	}

	public final @Nullable NALTask answer(Termed X, byte punc, long start, long end, float dur) {
		assert punc == BELIEF || punc == GOAL;

        var x = X.term();
        var negate = x instanceof Neg;
		if (negate)
			x = x.unneg();

		TaskTable table = table(x, punc==BELIEF);
		if (table == null || table.isEmpty()) return null;

        var y = answer(table, start, end, x, null, dur, answer.ANSWER_CAPACITY);
		return negate && y!=null ? SpecialNegTask.neg(y) : y;
	}

	@Nullable public final NALTask answer(TaskTable table, long start, long end, @Nullable Term template, @Nullable Predicate<NALTask> filter, float dur, int capacity) {
        var a = match(table, start, end, template, filter, dur, capacity);
		return a!=null ? a.task(
			    false
		) : null;
	}

	public final Answer match(TaskTable t, long start, long end, @Nullable Term template, @Nullable Predicate<NALTask> filter, float dur, int capacity) {
        return t.isEmpty() ? null : new Answer(template, !(t instanceof QuestionTable),
                start, end, dur,
                capacity, this).filter(filter)
                .match(t);
	}

	public void stats(boolean concepts, boolean emotions, Logger logger) {
        var s = new StringBuilder(4 * 1024);
		stats(concepts, emotions, s);
		if (!s.isEmpty())
			logger.info("{} stats\n{}", self, s);
	}

	public SortedMap<String, Object> stats(boolean concepts, boolean emotions, Appendable out) {

        var stat = stats(concepts, emotions);

		try {
			for (var e : stat.entrySet()) {
				out
					.append(e.getKey().replace(" ", "/"))
					.append(" \t ")
					.append(e.getValue().toString())
					.append('\n');
			}
			out.append('\n');
		} catch (IOException e) {
			e.printStackTrace();
		}

		return stat;
	}

	public final void retract(long stampComponent, Predicate<Task> each) {
		tasks().filter(x -> Longs.contains(x.stamp(), stampComponent)).filter(each.negate()).forEach(Task::delete);
	}



	public final NAR synch() {
		return this;
	}

	public final NALTask answerBelief(Term x, long when) {
		return answerBelief(x, when, when);
	}

	public final NALTask answerBelief(Term x, long start, long end) {
		return answer(x, BELIEF, start, end);
	}

	public final NALTask answerGoal(Term x, long when) {
		return answerGoal(x, when, when);
	}

	public final NALTask answerGoal(Term x, long start, long end) {
		return answer(x, GOAL, start, end);
	}


	public final <X> Stream<X> parts(Class<? extends X> nAgentClass) {
		return this.partStream().filter(nAgentClass::isInstance).map(x -> (X) x);
	}


	public Off stopIf(BooleanSupplier stopCondition) {
		return eventCycle.onWeak(n -> {
			if (stopCondition.getAsBoolean())
				stop();
		});
	}

	public final Term eval(Term x) {
		if (x instanceof Compound c) {
            var y = evaluator.first(c);
			if (y != null)
				return y;
		}
		return x;
	}

	public final Concept conceptualizeDynamic(Termed concept) {

		Term ct;
		Termed x;
		if (concept instanceof Concept) {
			ct = (x = concept).term();
		} else {
			if ((ct = conceptTerm(concept, true)) == null)
				return null;
			x = ct;
		}

        var y = concept(x);
		return y != null ? y :
				conceptBuilder.apply(ct, false, true);
	}

	public @Nullable TaskTable table(Termed concept, byte punc, boolean conceptualize) {
		return switch (punc) {
			case BELIEF -> table(concept, true, conceptualize);
			case GOAL ->   table(concept, false, conceptualize);
			default -> {
                var exist = concept(concept, conceptualize);
				yield exist instanceof TaskConcept tc ? tc.table(punc) : null;
			}
		};
	}

	public final @Nullable BeliefTable table(Termed c, boolean beliefOrGoal) {
		return table(c, beliefOrGoal, false);
	}

	public final @Nullable BeliefTable table(Termed c, boolean beliefOrGoal, boolean conceptualize) {
		@Nullable Term ct = conceptTerm(c, true);
		if (ct == null) return null;
		var y = _conceptualize(ct, conceptualize);
        return y != null ? tableActual(beliefOrGoal, y) : DynTables.tableDyn(ct, beliefOrGoal);
    }

	@Nullable private static BeliefTable tableActual(boolean beliefOrGoal, Concept y) {
		return y instanceof TaskConcept c ?
			(BeliefTable) c.table(beliefOrGoal ? BELIEF : GOAL)
			: null;
	}


	public final Off on(Every cycle, Consumer<NAR> c) {
		return cycle.on(this, 1, c);
	}

	public final Off on(Every cycle, float amount, Consumer<NAR> c) {
		return cycle.on(this, amount, c);
	}

	public static void proofPrint(TruthProjection r ) {
		for (var x : r) {
			proofPrint(x);
			System.out.println();
		}
	}

	@Nullable public final TaskConcept conceptualizeTask(Termed x) {

        var c = conceptualize(x);

		if (c instanceof TaskConcept C) {
			if (test.DEBUG_EXTRA && !x.term().equalsRoot(C.term()))
				throw new TermTransformException("conceptualization mismatch", x.term(), c.term());

			return C;
		} else {
			if (DEBUG && c!=null)
				throw new TermTransformException("conceptualized to non-TaskConcept: " + c.getClass(), x.term(), c.term());

			return null;
		}

	}

	public NALTask want(Term x, float freq, float conf, Tense tense) {
        var now = time(tense);
		return want(priDefault(GOAL), x, now, now, freq, conf);
	}

	public NALTask believe(Term term, Tense tense, float freq, float conf) {
		return believe(term, time(tense), freq, conf);
	}

	public NALTask believe(Term term, long when, float freq, float conf) {
		return believe(priDefault(BELIEF), term, when, freq, conf);
	}

	public NALTask believe(Term term, long when, float freq) {
		return believe(term, when, freq, confDefault(BELIEF));
	}

	public NALTask believe(Term term, float freq, float conf) {
		return believe(term, Tense.Eternal, freq, conf);
	}

	public NALTask want(Term term, float freq, float conf) {
		return want(term, freq, conf, Tense.Eternal);
	}

	public NALTask believe(String term, Tense tense, float freq, float conf) {
		try {
			return believe(priDefault(BELIEF), $(term), time(tense), freq, conf);
		} catch (NarseseException e) {
			throw new RuntimeException(e);
		}
	}

	public NALTask believe(String termString, float freq, float conf) {
		return believe(termString, freq, conf, ETERNAL, ETERNAL);
	}

	public NALTask believe(String termString, float freq, float conf, long start, long end) {
		return believe($$(termString), freq, conf, start, end);
	}

	public NALTask want(String termString, float freq, float conf, long start, long end) {
		return want($$(termString), freq, conf, start, end);
	}

	public NALTask want(String termString) {
		return want($$(termString));
	}

	public void believe(String... tt) throws NarseseException {
		for (var b : tt)
			believe(b, true);
	}

	public NALTask question(String termString) throws NarseseException {
		return question($(termString));
	}

	public NALTask question(Term c) {
		return ask(c, QUESTION);
	}

	public NALTask quest(Term c) {
		return ask(c, QUEST);
	}

	public NALTask input(float pri, Term term, byte punc, long start, long end, @Nullable Truth truth) throws TaskException {
        var z = NALTask.task(term, punc, truth, start, end, evidence());
		z.pri(pri);
		input(z);
		return z;
	}

	public NALTask believe(Term term, long start, long end) throws TaskException {
		return input(priDefault(BELIEF), term, BELIEF, start, end, $.t(1.0f, confDefault(BELIEF)));
	}

	public NALTask believe(Term term, float freq, float conf, long start, long end) throws TaskException {
		return input(priDefault(BELIEF), term, BELIEF, start, end, $.t(freq, conf));
	}

	public NALTask want(Term term, float freq, float conf, long start, long end) throws TaskException {
		return want(priDefault(GOAL), term, start, end, freq, conf);
	}

	public NALTask believe(float pri, Term term, long occurrenceTime, float freq, float conf) throws TaskException {
		return input(pri, term, BELIEF, occurrenceTime, occurrenceTime, $.t(freq, conf));
	}

	public NALTask want(float pri, Term goal, long start, long end, float freq, float conf) throws TaskException {
		return input(pri, goal, GOAL, start, end, $.t(freq, conf));
	}

	public NALTask ask(Term term, byte punc, long... startEnd) {
		assert startEnd.length == 2;
		return input(priDefault(punc), term, punc, startEnd[0], startEnd[1], null);
	}

	public NALTask ask(Term term, byte punc) {
		return ask(term, punc, ETERNAL, ETERNAL);
	}

	public NALTask believe(String termString, boolean isTrue) throws NarseseException {
		return believe($(termString), isTrue);
	}

	public NALTask believe(Term term) {
		return believe(term, true);
	}

	public NALTask believe(Term term, boolean trueOrFalse) {
		return believe(term, trueOrFalse, confDefault(BELIEF));
	}

	public NALTask want(Term term) {
		return want(term, 1.0f, confDefault(GOAL));
	}

	public NALTask believe(Term term, boolean trueOrFalse, float conf) {
		return believe(term, trueOrFalse ? 1.0f : 0.0f, conf);
	}

	public NALTask want(Term term, boolean trueOrFalse, float conf) {
		return want(term, trueOrFalse ? 1.0f : 0.0f, conf);
	}

	public NALTask believe(Term term, long when) throws TaskException {
		return believe(term, when, when);
	}

	public final float throttle() {
		return cpuThrottle.asFloat();
	}

	public final void throttle(float t) {
		cpuThrottle.set(t);
		Loop.logger.info("{} throttle={}", self(), t);
	}


	public final FloatRange cpuThrottle = FloatRange.unit(1);

	public final class NARLoop extends InstrumentedLoop {

		private final boolean async;

		public NARLoop(boolean asyncMode) {
			this.async = asyncMode;
		}

		@Override
		protected void starting() {
			super.starting();
			parts(Pausing.class).forEach(g->g.pause(false));
		}

		@Override
		protected void stopping() {
			parts(Pausing.class).forEach(g->g.pause(true));
			super.stopping();
		}


		private final Runnable ready = this::ready;

		@Override
		public final boolean next() {
            long cycleStartTime = Profiler.startTime();
            logger.debug("NARLoop.next() cycle start, time: {}", NAR.this.time());

            long now = NAR.this.time.next();

            long scheduledTasksStartTime = Profiler.startTime();
            exe.run(now);
            Profiler.recordTime("NARLoop.scheduledTasks", scheduledTasksStartTime);

            commitTime(now); // commitTime itself is not profiled, but it's part of the cycle

            if (narAttn != null) {
                long attentionCommitStartTime = Profiler.startTime();
                narAttn.commit();
                Profiler.recordTime("NARLoop.attentionCommit", attentionCommitStartTime);
            }

            long eventCycleProcessingStartTime = Profiler.startTime();
            if (async) {
                eventCycle.emitAsync(NAR.this, exe, ready);
            } else {
                eventCycle.accept(NAR.this);
            }
            Profiler.recordTime("NARLoop.eventCycleProcessing", eventCycleProcessingStartTime);

            logger.debug("NARLoop.next() cycle end, time: {}", NAR.this.time());
            Profiler.recordTime("NARLoop.nextCycleTotal", cycleStartTime); // Profile the entire next() method
			return true;
		}

		@Override
		protected boolean async() {
			return async;
		}

		public NAR nar() {
			return NAR.this;
		}

		public void startPeriodMS(int periodMS) {
			startThread(periodMS);
		}

	}

	public static final String VERSION = "NARchy v?.?";

    public void enableProfiling() {
        Profiler.enable();
    }

    public void disableProfiling() {
        Profiler.disable();
    }

    public void resetProfilingStats() {
        Profiler.reset();
    }

    public String getProfilingStats() {
        return Profiler.getStats();
    }


    public abstract class Attn {
        public final Bag<? extends Termed, PriReference<? extends Termed>> _bag;

        protected Attn(Bag bag) {
            this._bag = bag;
        }

        public abstract boolean activate(NALTask t);
        public abstract void sample(TaskBagAttentionSampler sampler, int iter, PuncBag punc, RandomBits rng);
        public abstract Stream<Concept> concepts();

        public final void clear() {
            _bag.clear();
        }

        public final void capacity(int capacity) {
            _bag.setCapacity(capacity);
        }

        public final void commit() {
            _bag.commit(attnUpdater.apply(_bag));
        }

        public final int capacity() {
            return _bag.capacity();
        }

        public final boolean isEmpty() {
            return _bag.isEmpty();
        }
    }

    public class TaskBagAttn extends Attn {
        public final TaskAttention bag;

        protected TaskBagAttn(NAR narInstance) {
            super((Bag)(this.bag = new TaskAttention(false, narInstance)).model);
        }

        @Override
        public boolean activate(NALTask t) {
            bag.accept(t);
            return true;
        }

        @Override
        public Stream<Concept> concepts() {
            return terms(s -> true)
                    .map(NAR.this::concept)
                    .filter(Objects::nonNull);
        }

        public final Stream<Term> terms(Predicate<Term> filter) {
            return bag.stream()
                    .unordered()
                    .map(x -> x.id.term())
                    .distinct()
                    .filter(filter);
        }

        @Override
        public void sample(TaskBagAttentionSampler sampler, int iter, PuncBag punc, RandomBits rng) {
            bag.seed(sampler, iter, punc, rng);
        }
    }
}