/*
 * This file is part of Beads. See http:
 */
package net.beadsproject.beads.ugens;

import net.beadsproject.beads.core.AudioContext;
import net.beadsproject.beads.core.Auvent;
import net.beadsproject.beads.core.UGen;
import net.beadsproject.beads.data.DataAuvent;

import java.util.Arrays;

/**
 * Implements a simple comb filter with both feed-forward and feed-back
 * components.
 * <p>
 * y(n) = a * x(n) + g * x(n - d) - h * y(n - d)
 *
 * @author Benito Crawford
 * @version 0.9.1
 * @beads.category filter
 */
public class CombFilter extends IIRFilter {

    private float a = 1;
    private float g = 0.2f;
    private float h = 0.2f;
    private int maxDelay = 1;
    private int delay = 1;
    private int ind;
    private UGen aUGen;
    private UGen gUGen;
    private UGen hUGen;
    private UGen delayUGen;
    private boolean isAStatic;
    private boolean isGStatic;
    private boolean isHStatic;
    private boolean isDelayStatic;
    private boolean areAllStatic;
    private final float[] xn;
    private final float[] yn;
    private int bufLen = 1;

    /**
     * Constructor.
     *
     * @param con    The audio context.
     * @param maxdel The maximum delay in samples.
     */
    public CombFilter(AudioContext con, int maxdel) {
        super(con, 1, 1);
        maxDelay = Math.max(maxdel, 1);
        bufLen = maxDelay + 1;
        yn = new float[bufLen];
        xn = new float[bufLen];
        setA(a).setG(g).setH(h).setDelay(1);
    }

    private void checkStaticStatus() {
        areAllStatic = isAStatic && isGStatic && isGStatic && isDelayStatic;
    }

    @Override
    public void gen() {

        float[] bi = bufIn[0];
        float[] bo = bufOut[0];

        if (areAllStatic) {
            for (int currsample = 0; currsample < bufferSize; currsample++) {
                int ind2 = (ind + bufLen - delay) % bufLen;
                bo[currsample] = yn[ind] = a * (xn[ind] = bi[currsample]) + g
                        * xn[ind2] - h * yn[ind2];
                ind = (ind + 1) % bufLen;
            }
        } else {
            aUGen.update();
            gUGen.update();
            hUGen.update();
            delayUGen.update();

            for (int currsample = 0; currsample < bufferSize; currsample++) {
                a = aUGen.getValue(0, currsample);
                g = gUGen.getValue(0, currsample);
                h = hUGen.getValue(0, currsample);
                delay = (int) delayUGen.getValue(0, currsample);
                if (delay < 1) {
                    delay = 1;
                } else if (delay >= maxDelay) {
                    delay = maxDelay;
                }

                int ind2 = (ind + bufLen - delay) % bufLen;
                bo[currsample] = yn[ind] = a * (xn[ind] = bi[currsample]) + g
                        * xn[ind2] - h * yn[ind2];
                ind = (ind + 1) % bufLen;
            }
        }
    }

    /**
     * Use this to reset the filter if it explodes.
     */
    public void reset() {
        Arrays.fill(yn, 0);
    }

    /**
     * Gets the maximum delay in samples.
     *
     * @return The maximum delay.
     */
    public int getMaxDelay() {
        return maxDelay;
    }

    /**
     * Gets the current delay in samples.
     *
     * @return The delay in samples.
     */
    public int getDelay() {
        return delay;
    }

    /**
     * Sets the delay time in samples. This will remove the delay UGen, if there
     * is one.
     *
     * @param delay The delay in samples.
     * @return This CombFilter instance.
     */
    private CombFilter setDelay(int delay) {
        this.delay = delay < 1 ? 1 : Math.min(delay, maxDelay);

        if (isDelayStatic) {
            delayUGen.setValue(delay);
        } else {
            delayUGen = new Static(context, delay);
            isDelayStatic = true;
            checkStaticStatus();
        }
        return this;
    }

    /**
     * Sets a UGen to specify the delay in samples (converted to ints).
     *
     * @param delay The delay UGen.
     * @return This CombFilter instance.
     */
    private CombFilter setDelay(UGen delay) {
        if (delay == null) {
            setDelay(this.delay);
        } else {
            delayUGen = delay;
            delay.update();
            this.delay = (int) delay.getValue();
            isDelayStatic = false;
            areAllStatic = false;
        }
        return this;
    }

    /**
     * Gets the delay UGen, if it exists.
     *
     * @return The delay UGen.
     */
    public UGen getDelayUGen() {
        return isDelayStatic ? null : delayUGen;
    }

    /**
     * Gets the g parameter.
     *
     * @return The g parameter.
     */
    public float getG() {
        return g;
    }

    /**
     * Sets the g parameter to a float value. This will remove the g UGen, if
     * there is one.
     *
     * @param g The g parameter.
     * @return This CombFilter instance.
     */
    private CombFilter setG(float g) {
        this.g = g;
        if (isGStatic) {
            gUGen.setValue(g);
        } else {
            gUGen = new Static(context, g);
            isGStatic = true;
            checkStaticStatus();
        }
        return this;
    }

    /**
     * Sets a UGen to specify the g parameter. Passing a null value will freeze
     * the parameter at its current value.
     *
     * @param g The g UGen.
     * @return This CombFilter instance.
     */
    private CombFilter setG(UGen g) {
        if (g == null) {
            setG(this.g);
        } else {
            gUGen = g;
            g.update();
            this.g = g.getValue();
            isGStatic = false;
            areAllStatic = false;
        }
        return this;
    }

    /**
     * Gets the g UGen, if it exists.
     *
     * @return The g UGen.
     */
    public UGen getGUGen() {
        return isGStatic ? null : gUGen;
    }

    /**
     * Gets the h parameter.
     *
     * @return The h parameter.
     */
    public float getH() {
        return h;
    }

    /**
     * Sets the h parameter to a float value. This will remove the h UGen, if
     * there is one.
     *
     * @param h The h parameter.
     * @return This CombFilter instance.
     */
    private CombFilter setH(float h) {
        this.h = h;
        if (isHStatic) {
            hUGen.setValue(h);
        } else {
            hUGen = new Static(context, h);
            isHStatic = true;
            checkStaticStatus();
        }
        return this;
    }

    /**
     * Sets a UGen to specify the h parameter. Passing a null value will freeze
     * the parameter at its current value.
     *
     * @param h The h UGen.
     * @return This CombFilter instance.
     */
    private CombFilter setH(UGen h) {
        if (h == null) {
            setH(this.h);
        } else {
            aUGen = h;
            h.update();
            this.h = h.getValue();
            isHStatic = false;
            areAllStatic = false;
        }
        return this;
    }

    /**
     * Gets the h UGen, if it exists.
     *
     * @return The h UGen.
     */
    public UGen getHUGen() {
        return isHStatic ? null : hUGen;
    }

    /**
     * Gets the 'a' parameter.
     *
     * @return The 'a' parameter.
     */
    public float getA() {
        return a;
    }

    /**
     * Sets the 'a' parameter to a float value. This will remove the 'a' UGen,
     * if there is one.
     *
     * @param a The 'a' parameter.
     * @return This CombFilter instance.
     */
    private CombFilter setA(float a) {
        this.a = a;
        if (isAStatic) {
            aUGen.setValue(a);
        } else {
            aUGen = new Static(context, a);
            isAStatic = true;
            checkStaticStatus();
        }
        return this;
    }

    /**
     * Sets a UGen to specify the 'a' parameter. Passing a null value will
     * freeze the parameter at its current value.
     *
     * @param a The 'a' UGen.
     * @return This CombFilter instance.
     */
    private CombFilter setA(UGen a) {
        if (a == null) {
            setA(this.a);
        } else {
            aUGen = a;
            a.update();
            this.a = a.getValue();
            isAStatic = false;
            areAllStatic = false;
        }
        return this;
    }

    /**
     * Gets the 'a' UGen, if it exists.
     *
     * @return The 'a' UGen.
     */
    public UGen getAUGen() {
        return isAStatic ? null : aUGen;
    }

    /**
     * Sets all the parameters at once. This will clear parameter UGens, if they
     * exist.
     *
     * @param delay The delay in samples.
     * @param a     The 'a' parameter.
     * @param g     The g parameter.
     * @param h     The h parameter.
     * @return This CombFilter instance.
     */
    public CombFilter setParams(int delay, float a, float g, float h) {
        setA(a);
        setG(g);
        setH(h);
        setDelay(delay);
        return this;
    }

    /**
     * Sets the parameter UGens. Passing null values will freeze the parameters
     * at their previous values.
     *
     * @param delUGen The delay UGen.
     * @param aUGen   The 'a' UGen.
     * @param gUGen   The g UGen.
     * @param hUGen   The h UGen.
     * @return This CombFilter instance.
     */
    public CombFilter setParams(UGen delUGen, UGen aUGen, UGen gUGen, UGen hUGen) {
        setDelay(delUGen);
        setA(aUGen);
        setG(gUGen);
        setH(hUGen);
        return this;
    }

    /**
     * Sets the filter parameters with a DataBead.
     * <p>
     * Use the following properties to specify filter parameters:
     * </p>
     * <ul>
     * <li>"a": (float or UGen)</li>
     * <li>"g": (float or UGen)</li>
     * <li>"h": (float or UGen)</li>
     * <li>"delay": (float or UGen)</li>
     * </ul>
     *
     * @param paramBead The DataBead specifying parameters.
     */
    private CombFilter setParams(DataAuvent paramBead) {
        if (paramBead != null) {
            Object o;

            if ((o = paramBead.get("a")) != null) {
                if (o instanceof UGen) {
                    setA((UGen) o);
                } else {
                    setA(paramBead.getFloat("a", a));
                }
            }

            if ((o = paramBead.get("g")) != null) {
                if (o instanceof UGen) {
                    setG((UGen) o);
                } else {
                    setG(paramBead.getFloat("g", g));
                }
            }

            if ((o = paramBead.get("h")) != null) {
                if (o instanceof UGen) {
                    setH((UGen) o);
                } else {
                    setH(paramBead.getFloat("h", h));
                }
            }

            if ((o = paramBead.get("delay")) != null) {
                if (o instanceof UGen) {
                    setDelay((UGen) o);
                } else {
                    setDelay((int) paramBead.getFloat("delay", delay));
                }
            }
        }
        return this;
    }

    @Override
    public void on(Auvent message) {
        if (message instanceof DataAuvent) {
            setParams((DataAuvent) message);
        }
    }

    /**
     * Gets a DataBead with properties "a", "g", "h", and "delay" set to the
     * corresponding filter parameters.
     *
     * @return The parameter DataBead.
     */
    public DataAuvent getParams() {
        DataAuvent db = new DataAuvent();
        if (isAStatic) {
            db.put("a", a);
        } else {
            db.put("a", aUGen);
        }

        if (isGStatic) {
            db.put("g", g);
        } else {
            db.put("g", gUGen);
        }

        if (isHStatic) {
            db.put("h", h);
        } else {
            db.put("h", hUGen);
        }

        if (isDelayStatic) {
            db.put("delay", delay);
        } else {
            db.put("delay", delayUGen);
        }

        return db;
    }

    /**
     * Gets a DataBead with properties "a", "g", "h", and "delay" set to static
     * float values corresponding to the current filter parameters.
     *
     * @return The static parameter DataBead.
     */
    public DataAuvent getStaticParams() {
        DataAuvent db = new DataAuvent();
        db.put("a", a);
        db.put("g", g);
        db.put("h", h);
        db.put("delay", delay);
        return db;
    }

    @Override
    public IIRFilterAnalysis getFilterResponse(float freq) {
        float[] bs = new float[delay + 1], as = new float[delay + 1];
        bs[0] = a;
        bs[delay] = g;
        as[0] = 1;
        as[delay] = h;
        return calculateFilterResponse(bs, as, freq, context.getSampleRate());
    }

    /*
     * public static void main(String[] args) { 
     * comparing the speed of this ParamUpdater 
     * new AudioContext(); for (int i = 0; i < 1000; i++) { CombFilter c = new
     * CombFilter(ac, 1000);
     *
     * 
     *
     * ac.out.addInput(c); } ac.start(); }
     */
}