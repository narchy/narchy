package jcog.memoize;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

class MemoGraphTest {

    @Test void usage() {
        var ctx = new MemoGraph();

        ctx .put("data", () -> "x")
                .then("data", this::fn, "processed")
                .then("processed", this::fn, "analysis")
                .then("processed", this::fn, "viz")
                .thenAll(List.of("analysis", "viz"),
                        l -> fn(l.get(0), l.get(1)),
                        "report")
                .get("report", (result)->{
                    //assertEquals(5, ctx.proc.size());

                    var report = ctx.get("report");
                    assertNotNull(report);
                    assertArrayEquals(new Object[]{"x","x"}, (Object[])report);
                })
                .run();

        ctx.clear();
    }

    private Object fn(Object... o) {
        return o.length == 1 ? o[0] : o;
    }

    @Test void testThenAll() {
        var ctx = new MemoGraph()
                .put("dataA", () -> "A")
                .put("dataB", () -> "B")
                .then("dataA", x -> x + "1", "processedA")
                .then("dataB", x -> x + "2", "processedB")
                .thenAll(List.of("processedA", "processedB"),
                        inputs -> inputs.get(0) + "+" + inputs.get(1),
                        "combined");

        ctx.get("combined", (result)->{
            assertEquals(result, ctx.get("combined"));

            //assertEquals(5, ctx.proc.size());
            assertEquals("A1+B2", ctx.get("combined"));

            // Verify intermediate results
            assertEquals("A", ctx.get("dataA"));
            assertEquals("B", ctx.get("dataB"));
            assertEquals("A1", ctx.get("processedA"));
            assertEquals("B2", ctx.get("processedB"));
        }).run();
    }

    /** TODO */
    @Disabled @Test
    void testCycleDetection() {
        var ctx = new MemoGraph()
                .put("a", () -> "start")
                .then("a", x -> x + "1", "b")
                .then("b", x -> x + "2", "c");
        assertThrows(IllegalStateException.class,
                () -> ctx.then("c", x -> x + "3", "a").run(),
                "Expected cycle detection");
    }


    @Test void testDuplicateKey() {
        var ctx = new MemoGraph()
                .put("data", () -> "x")
                .then("data", x -> x + "1", "result");

        assertThrows(IllegalStateException.class,
                () -> ctx.then("data", x -> x + "2", "result"));
    }


    @Test
    void testPartialComputation() {
        var analysisResult = new Object[1];
        var vizResult = new Object[1];

        new MemoGraph()
                .put("data", () -> "start")
                .then("data", x -> x + "_processed", "processed")
                .then("processed", x -> x + "_analysis", "analysis")
                .then("processed", x -> x + "_viz", "viz")
                .get("analysis", v -> analysisResult[0] = v)
                .get("viz", v -> vizResult[0] = v)
                .run();

        assertEquals("start_processed_analysis", analysisResult[0]);
        assertEquals("start_processed_viz", vizResult[0]);
    }

    @Nested
    class DedupTest {

        /** simulates expensive term computation */
        static class Term {
            static int computeCount = 0;
            final String value;

            @Override
            public boolean equals(Object o) {
                if (!(o instanceof Term term)) return false;
                return Objects.equals(value, term.value);
            }

            @Override
            public int hashCode() {
                return Objects.hashCode(value);
            }

            Term(String v) {
                computeCount++;
                this.value = v;
            }

            public ConceptTerm concept() {
                return new ConceptTerm(value + "_concept");
            }
        }

        /** simulates concept term derivation */
        static class ConceptTerm {
            static int computeCount = 0;
            final String value;

            ConceptTerm(String v) {
                computeCount++;
                this.value = v;
            }
        }

        /** simulates final concept computation */
        static class Concept {
            static int computeCount = 0;
            final String value;

            Concept(String v) {
                computeCount++;
                this.value = v;
            }
        }

        /** tracks final remember operations */
        static class Memory {
            static int rememberCount = 0;
            List<String> remembered = new ArrayList<>();

            void remember(Term t, Concept c) {
                rememberCount++;
                remembered.add(t.value + "->" + c.value);
            }
        }

        @Test
        public void testPipelineDeduplication() {
            MemoGraph graph = new MemoGraph();
            Memory memory = new Memory();

            // Reset counters
            Term.computeCount = 0;
            ConceptTerm.computeCount = 0;
            Concept.computeCount = 0;
            Memory.rememberCount = 0;

            // Create two tasks with same term value
            Term t1 = new Term("A");
            Term t2 = new Term("A"); // same value, should dedupe
            Term t3 = new Term("B"); // different value

            // Process each task
            for (var t : List.of(t1, t2, t3)) {
                graph.chain(t,
                        (g, T) -> g.share(T, (gg,TT) -> {
                            ConceptTerm ct = TT.concept();
                            return gg.share(ct, (ggg,CT) ->
                                    new Concept(CT.value + "_final"));
                        }),
                        memory::remember
                );

                graph.run();
            }

            // Verify deduplication:
            assertEquals(3, Term.computeCount); // Terms are task-specific
            assertEquals(2, ConceptTerm.computeCount); // Should dedupe A's concept term
            assertEquals(2, Concept.computeCount); // Should dedupe A's concept
            assertEquals(2, Memory.rememberCount); // Each unique task needs its own remember

            // Verify correct results stored
            assertEquals(2, memory.remembered.size());
            assertEquals("A->A_concept_final", memory.remembered.get(0));
            assertEquals("B->B_concept_final", memory.remembered.get(1));
        }
    }

}