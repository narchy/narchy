package nars.game;

import nars.$;
import nars.NAR;
import nars.NARS;
import nars.game.action.AbstractAction;
import nars.term.atom.Atomic;

import java.util.Arrays;

import static java.lang.System.out;
import static nars.term.atom.Atomic.atom;


/**
 * Created by me on 5/4/16.
 */
public class Line1DContinuous extends Game {

    static {

    }

    @FunctionalInterface
    public interface IntToFloatFunction {
        float valueOf(int i);
    }

    private final IntToFloatFunction targetFunc;
    private int size;
    private boolean print;
    private float yHidden;
    private float yEst;
    private final float speed = 5f;
    private final float[] ins;

    private Line1DContinuous(NAR n, int size, IntToFloatFunction target) {
        super("x", GameTime.durs(1));
        this.size = size;
        ins = new float[size * 2];
        this.targetFunc = target;

        yEst = size / 2f;
        yHidden = size / 2f;


        for (int i = 0; i < size; i++) {
            int ii = i;
            sense($.func("h", atom("x"), $.the(i)), () -> ins[ii]);
            sense($.func("e", atom("x"), $.the(i)), () -> ins[size + ii]);
        }

        AbstractAction a;

        actionBipolar($.inh(Atomic.atomic("move"), Atomic.atomic("x")), (v) -> {

            yEst += (v) * speed;

            return yEst;
        });

        reward(() -> {
            yHidden = Math.round(targetFunc.valueOf((int) nar.time()) * (size - 1));

            yHidden = Math.min(size - 1, Math.max(0, yHidden));
            yEst = Math.min(size - 1, Math.max(0, yEst));


            Arrays.fill(ins, 0f);
            float smoothing = 1 / 2f;
            for (int i = 0; i < size; i++) {
                ins[i] = Math.abs(yHidden - i) / (size * smoothing);
                ins[i + this.size] = Math.abs(yEst - i) / (size * smoothing);
            }


            float dist = Math.abs(yHidden - yEst) / this.size;


            float reward =
                    -dist * 2f + 1f;


            if (yEst > this.size - 1) yEst = this.size - 1;
            if (yEst < 0) yEst = 0;


            if (print) {


                int colActual = Math.round(yHidden);
                int colEst = Math.round(yEst);
                for (int i = 0; i < this.size; i++) {

                    char c;
                    if (i == colActual && i == colEst) {
                        c = '@';
                    } else if (i == colActual)
                        c = 'X';
                    else if (i == colEst)
                        c = '+';
                    else
                        c = '.';

                    out.print(c);
                }


                out.print(' ');
                out.print(summary());
                out.println();
            }

            return reward;

        });
    }


    public static IntToFloatFunction sine(float targetPeriod) {
        return (t) -> 0.5f + 0.5f * (float) Math.sin(t / (targetPeriod));
    }

    private static IntToFloatFunction random(float targetPeriod) {
        return (t) -> (((((int) (t / targetPeriod)) * 31) ^ 37) % 256) / 256.0f;
    }

    public static void main(String[] args) {

        NAR nar = new NARS().get();
        nar.complexMax.set(32);


        nar.beliefConfDefault.set(0.9f);
        nar.goalConfDefault.set(0.9f);


        Line1DContinuous l = new Line1DContinuous(nar, 6,
                random(16)
        );

        l.print = true;


    }


}