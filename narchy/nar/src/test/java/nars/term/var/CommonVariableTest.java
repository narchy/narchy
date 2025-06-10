package nars.term.var;

import nars.$;
import nars.Term;
import nars.io.IO;
import nars.unify.UnifyAny;
import org.junit.jupiter.api.Test;

import java.util.Set;
import java.util.TreeSet;

import static nars.$.$$;
import static nars.term.util.Testing.assertEq;
import static nars.term.var.CommonVariable.common;
import static org.junit.jupiter.api.Assertions.*;


/**
 * Created by me on 9/9/15.
 */
class CommonVariableTest {


    private static final Variable p1 = $.varDep(1);
    private static final Variable p2 = $.varDep(2);
    private static final Variable p3 = $.varDep(3);
    private static final Variable p4 = $.varDep(4);
    private static final Variable c12 = common(p1, p2);


    @Test
    void commonVariableTest1() {


        Variable p1p2 = common(p1, p2);
        assertEquals("##1#2", p1p2.toString());
        assertSerialize(p1p2);

        assertSame(p1p2, common(p1, p1p2)); //subsumed, same instance

        Variable  p2p1 = common(p2, p1);
        assertEquals("##1#2", p2p1.toString());
        assertSerialize(p2p1);

        Variable p2p3p1 = common(p2,  common(p3, p1));
        assertEquals("##1#2#3", p2p3p1.toString());
        assertSerialize(p2p3p1);

    }

    @Test
    void testInvalid() {
        assertThrows(Throwable.class, ()-> {
            Variable p1p1 = common(p1, p1);
        });
    }


    @Test
    void CommonVariableDirectionalityPreserved() {
        assertEquals(c12, common(p2, p1));
    }

    @Test
    void CommonVariableOfCommonVariable() {

        Variable c123 = common( c12,  p3);
        assertSerialize(c123);

        assertEquals("##1#2#3 class nars.term.var.CommonVariable", (c123 + " " + c123.getClass()));


        Variable c1232 = common(c123, p2);
        assertSerialize(c123);
        assertEquals("##1#2#3", c1232.toString());

    }

    private static void assertSerialize(Variable x) {
        assertEq(x, IO.bytesToTerm(x.bytes()));
    }

    @Test void UnifyIncludeCommonVariable() {
        Variable c123 = common( c12,  p3);
        {
            UnifyAny u = new UnifyAny();
            assertTrue(u.unifies(c123, p3));
            assertEquals("{#3=##1#2#3}$0", u.toString());
        }
        {
            UnifyAny u = new UnifyAny();
            assertTrue(u.unifies(c123, p4));
            assertEquals(
                    "{##1#2#3=##1#2#3#4, #4=##1#2#3#4}$0",
                    //"{#4=##1#2#3}$0",
                    u.toString());
        }
    }

    @Test void UnifyCommonVar_DepIndep() {
        String vx = "$", vy = "#";
        Set<String> uu = new TreeSet();
        for (int i = 0; i < 16; i++) {
            UnifyAny u = new UnifyAny();
            assertTrue(
                    u.uni($$("x(" + vx + "1," + vy + "1)"), $$("x(" + vy + "1," + vx + "1)"))
            );
            uu.add(u.toString());
            for (Term c : u.xy.values()) {
                assertSerialize((Variable) c);
            }
            //System.out.println(u);
        }

        assertEquals(1, uu.size());
        assertEquals("[{$2=#2, $1=#1}$0]"
            ,uu.toString());

//        assertEquals(1, uu.size());
//        assertEquals(
//                "[{#1=$1, #2=$2}$0]", uu.toString());


    }

}