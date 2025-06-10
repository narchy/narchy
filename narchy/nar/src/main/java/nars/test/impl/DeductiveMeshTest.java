package nars.test.impl;

import nars.$;
import nars.NALTask;
import nars.NAR;
import nars.Term;
import nars.test.TestNAR;

import java.util.HashSet;
import java.util.Set;

/**
 * TODO abstract edge() for different relation types:
 * similarity
 * implication
 * etc
 */
public class DeductiveMeshTest {



    public final TestNAR test;

    public final Set<Term> coords;
    public final Set<Term> edges;


    public DeductiveMeshTest(NAR n, int... dims) {
        this(n, dims, -1);
    }

    public DeductiveMeshTest(NAR n, int[] dims, int timeLimit) {
        this(new TestNAR(n), dims, timeLimit);
    }

    public DeductiveMeshTest(TestNAR n, int[] dims, int timeLimit) {

        if (dims.length != 2)
            throw new UnsupportedOperationException("2-D only implemented");

        int sx = 0, sy = 0;
        int tx = dims[0] - 1;
        int ty = dims[1] - 1;

        coords = new HashSet();
        edges = new HashSet();

        for (int x = 0; x < dims[0]; x++) {
            for (int y = 0; y < dims[1]; y++) {
                coords.add($.p(x,y));

                /*if (x > y)*/ {
                    if (x > 0)
                        edges.add(edge(x, y, x - 1, y));
                    if (y > 0)
                        edges.add(edge(x, y, x, y - 1));
                    if (x < tx)
                        edges.add(edge(x, y, x + 1, y));
                    if (y < ty)
                        edges.add(edge(x, y, x, y + 1));
                }
            }
        }

        NAR nar = n.nar;
        for (Term edge : edges)
            nar.believe(edge);


        nar.believe(vertex(sx,sy));
        //nar.believe(edge(sx, sy, tx, ty).neg()); //contrast

        NALTask q = ask(n,
            vertex(tx,ty)
            //edge(0, 0, tx, ty);
        );

        if (timeLimit > 0) {
            n.mustBelieve(timeLimit, q.term().toString(), 1.0f, 1.0f, 0.01f, 1.0f);
        }

        this.test = n;
    }

    public static NALTask ask(TestNAR n, Term term) {
        return n.nar.question(term);

    }

    private Term edge(int x1, int y1, int x2, int y2) {
        Term a = vertex(x1, y1);
        Term b = vertex(x2, y2);
        return edge(a, b);
    }

    protected Term edge(Term a, Term b) {
        return $.sim(a, b);
    }

    private static Term vertex(int x1, int y1) {
        return $.p($.the(x1), $.the(y1));
    }


}