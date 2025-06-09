/*
 * This file is part of Beads. See http:
 */
package net.beadsproject.beads.ugens;

import net.beadsproject.beads.core.AudioContext;
import net.beadsproject.beads.core.Auvent;
import net.beadsproject.beads.core.UGen;
import net.beadsproject.beads.core.UGenChain;
import net.beadsproject.beads.data.DataAuvent;
import net.beadsproject.beads.data.DataBeadReceiver;

/**
 * A generalized multi-channel wrapper for UGens. Can either be used to treat an
 * existing array of UGens as one, big, multi-channel UGen, or to create a
 * multi-channel UGen from newly created UGens.
 * <p>
 * <p>In the simple case, it just wraps N one-channel UGens into one N-channel UGen.
 *
 * @author Benito Crawford
 * @version 0.9.5
 */
public class MultiWrapper extends UGenChain implements DataBeadReceiver {
    private UGen[] ugens;
    private int channels;
    private int insPerChannel;
    private int outsPerChannel;

    /**
     * Constructor for an multi-channel wrapper for 1-in/1-out UGens on each
     * channel. {@link #buildUGens(int)} should be implemented to construct a
     * UGen for each channel.
     *
     * @param context  The audio context.
     * @param channels The number of channels.
     */
    public MultiWrapper(AudioContext context, int channels) {
        this(context, channels, 1, 1);
    }

    private MultiWrapper(AudioContext context, int numIns, int numOuts) {
        super(context, numIns, numOuts);
    }

    /**
     * Constructor for an n-channel wrapper for UGens with a certain number
     * inputs and a certain number outputs on each channel.
     * {@link #buildUGens(int)} should be implemented to construct a UGen for
     * each channel.
     *
     * @param context        The audio context.
     * @param channels       The number of channels.
     * @param insPerChannel  The number of inputs per channel UGen.
     * @param outsPerChannel The number of outputs per channel UGen.
     */
    private MultiWrapper(AudioContext context, int channels, int insPerChannel,
                         int outsPerChannel) {
        this(context, channels * insPerChannel, channels * outsPerChannel);

        this.insPerChannel = insPerChannel;
        this.outsPerChannel = outsPerChannel;
        this.channels = channels;

        ugens = new UGen[channels];

        for (int i = 0; i < channels; i++) {
            ugens[i] = buildUGens(i);
        }
        setupUGens();

    }

    /**
     * Constructor for a multi-channel wrapper for an array of UGens that
     * represent separate "channels".
     *
     * @param context        The audio context.
     * @param ugens          The array of UGens to wrap.
     * @param insPerChannel  The number of inputs per channel.
     * @param outsPerChannel The number of ouputs per channel.
     */
    public MultiWrapper(AudioContext context, UGen[] ugens, int insPerChannel,
                        int outsPerChannel) {
        this(context, ugens.length * insPerChannel, ugens.length
                * outsPerChannel);

        this.insPerChannel = insPerChannel;
        this.outsPerChannel = outsPerChannel;
        this.channels = ugens.length;
        this.ugens = ugens;

        setupUGens();

    }

    private void setupUGens() {

        for (int i = 0; i < channels; i++) {

            
            
            for (int j = 0; j < insPerChannel; j++) {
                if (j < ugens[i].getIns()) {
                    this.drawFromChainInput(i * insPerChannel + j, ugens[i], j);
                }
            }

            
            
            for (int j = 0; j < outsPerChannel; j++) {
                if (j < ugens[i].getOuts()) {
                    this.addToChainOutput(i * outsPerChannel + j, ugens[i], j);
                }
            }

        }
    }

    /**
     * Constructs and returns a UGen for each channel. If an array of UGens is
     * not provided to MCWrapper at construction, this method should be
     * overridden with code to create a UGen for each channel. To make the
     * channel UGens modifiable, the UGen should implement DataBeadReceiver
     * properly as this is the only way to change the channel UGens' parameters.
     *
     * @param channelIndex The index of the channel for which the UGen is being created.
     * @return The new channel UGen.
     */
    private UGen buildUGens(int channelIndex) {
        return new UGen(context, 1, 0) {
            @Override
            public void gen() {
            }
        };
    }

    /**
     * A convenience method of adding inputs to specific channel UGens. Calling
     * <p>
     * <code>addInput(c, i, sUGen, o)</code> is equivalent to calling
     * <p>
     * <code>addInput(c * insPerChannel + i, sUGen, o)</code>.
     *
     * @param channelIndex     The channel to call addInput on.
     * @param channelUGenInput The input index of the channel UGen.
     * @param sourceUGen       The source UGen.
     * @param sourceOutput     The output of the source UGen.
     */
    public void addInput(int channelIndex, int channelUGenInput,
                         UGen sourceUGen, int sourceOutput) {
        addInput(channelIndex * insPerChannel + channelUGenInput, sourceUGen,
                sourceOutput);
    }

    /**
     * Forwards a DataBead to all channel UGens. If the channel UGens are
     * DataBeadReceivers, it calls {@link DataBeadReceiver#sendData(DataAuvent)};
     * otherwise it calls {@link Auvent#accept(Auvent)}.
     *
     * @param db The DataBead.
     */
    @Override
    public DataBeadReceiver sendData(DataAuvent db) {
        for (int i = 0; i < channels; i++) {
            if (ugens[i] instanceof DataBeadReceiver) {
                ((DataBeadReceiver) ugens[i]).sendData(db);
            } else {
                ugens[i].accept(db);
                
            }
        }
        return this;
    }

    /**
     * Forwards a DataBead to a specific channel UGen. If the channel UGen is a
     * DataBeadReceiver, it calls {@link DataBeadReceiver#sendData(DataAuvent)};
     * otherwise it calls {@link Auvent#accept(Auvent)}.
     *
     * @param channel The index of the channel UGen to which to forward the
     *                DataBead.
     * @param db      The DataBead.
     */
    public DataBeadReceiver sendData(int channel, DataAuvent db) {
        for (int i = 0; i < channels; i++) {
            if (ugens[i] instanceof DataBeadReceiver) {
                ((DataBeadReceiver) ugens[i]).sendData(db);
            } else {
                ugens[i].accept(db);
            }
        }
        return this;
    }

    /**
     * Forwards Beads to channel UGens. If <code>message</code> is a DataBead
     * and the channel UGens are DataBeadReceivers, it calls
     * {@link DataBeadReceiver#sendData(DataAuvent)}; otherwise it calls
     * {@link Auvent#accept(Auvent)}.
     */
    @Override
    public void on(Auvent message) {
        if (message instanceof DataAuvent) {
            sendData((DataAuvent) message);
        } else {
            for (int i = 0; i < channels; i++) {
                ugens[i].accept(message);

            }
        }
    }

    /**
     * Gets the number of channels.
     *
     * @return The number of channels.
     */
    public int getChannels() {
        return channels;
    }

    /**
     * Gets the number of inputs per channel.
     *
     * @return The number of inputs per channel.
     */
    public int getInsPerChannel() {
        return insPerChannel;
    }

    /**
     * Gets the number of outs per channel.
     *
     * @return The number of outs per channel.
     */
    public int getOutsPerChannel() {
        return outsPerChannel;
    }

}
