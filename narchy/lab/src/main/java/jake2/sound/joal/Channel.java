/*
 * Created on Jun 19, 2004
 *
 * Copyright (C) 2003
 */
/*
Copyright (C) 1997-2001 Id Software, Inc.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

*/
package jake2.sound.joal;

import com.jogamp.openal.AL;
import com.jogamp.openal.ALConstants;
import com.jogamp.openal.ALException;
import jake2.Defines;
import jake2.Globals;
import jake2.client.CL_ents;
import jake2.game.entity_state_t;
import jake2.qcommon.Com;
import jake2.sound.Sound;
import jake2.sound.sfx_t;
import jake2.sound.sfxcache_t;
import jake2.util.Math3D;

import java.nio.ByteBuffer;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Channel
 *
 * @author cwei
 */
class Channel {

    static final int LISTENER = 0;
    static final int FIXED = 1;
    static final int DYNAMIC = 2;
    private static final int MAX_CHANNELS = 32;
    private static final float[] NULLVECTOR = {0, 0, 0};

    private static AL al;
    private static final Channel[] channels = new Channel[MAX_CHANNELS];
    private static final int[] sources = new int[MAX_CHANNELS];

    private static int[] buffers;
    private static final Map<Integer, Channel> looptable = new ConcurrentHashMap<>(MAX_CHANNELS);

    private static int numChannels;


    private static boolean streamingEnabled;
    private static int streamQueue;


    private int type;
    private int entnum;
    private int entchannel;
    private int bufferId;
    private final int sourceId;
    private float volume;
    private float rolloff;
    private final float[] origin = {0, 0, 0};


    private boolean autosound;
    private boolean active;
    private boolean modified;
    private boolean bufferChanged;
    private boolean volumeChanged;

    private Channel(int sourceId) {
        this.sourceId = sourceId;
        clear();
        this.volumeChanged = false;
        this.volume = 1;
    }

    private void clear() {
        entnum = entchannel = bufferId = -1;
        bufferChanged = false;
        rolloff = 0;
        autosound = false;
        active = false;
        modified = false;
    }

    private static int[] tmp;

    static int init(AL al, int[] buffers) {
        if (tmp == null)
            tmp = new int[1];

        Channel.al = al;
        Channel.buffers = buffers;

        numChannels = 0;
        for (int i = 0; i < MAX_CHANNELS; i++) {

            int sourceId;
            try {
                al.alGetError();
                al.alGenSources(1, tmp, 0);
                sourceId = tmp[0];
                int errorCode = al.alGetError();

                if (errorCode != ALConstants.AL_NO_ERROR) {
                    Com.Println("can't generate more sources: channel=" +
                            i + " sourceId=" + sourceId +
                            " alError:" + errorCode);
                    break;
                }
            } catch (ALException e) {

                Com.Println("can't generate more sources: " + e);
                break;
            }

            sources[i] = sourceId;

            channels[i] = new Channel(sourceId);
            numChannels++;


            al.alSourcef(sourceId, ALConstants.AL_GAIN, 1.0f);
            al.alSourcef(sourceId, ALConstants.AL_PITCH, 1.0f);
            al.alSourcei(sourceId, ALConstants.AL_SOURCE_RELATIVE, ALConstants.AL_FALSE);
            al.alSourcefv(sourceId, ALConstants.AL_VELOCITY, NULLVECTOR, 0);
            al.alSourcei(sourceId, ALConstants.AL_LOOPING, ALConstants.AL_FALSE);
            al.alSourcef(sourceId, ALConstants.AL_REFERENCE_DISTANCE, 200.0f);
            al.alSourcef(sourceId, ALConstants.AL_MIN_GAIN, 0.0005f);
            al.alSourcef(sourceId, ALConstants.AL_MAX_GAIN, 1.0f);
        }
        return numChannels;
    }

    static void reset() {
        for (int i = 0; i < numChannels; i++) {
            al.alSourceStop(sources[i]);
            al.alSourcei(sources[i], ALConstants.AL_BUFFER, 0);
            channels[i].clear();
        }
    }

    static void shutdown() {
        al.alDeleteSources(numChannels, sources, 0);
        numChannels = 0;
    }

    private static void enableStreaming() {
        if (streamingEnabled) return;


        numChannels--;
        streamingEnabled = true;
        streamQueue = 0;

        int source = channels[numChannels].sourceId;
        al.alSourcei(source, ALConstants.AL_SOURCE_RELATIVE, ALConstants.AL_TRUE);
        al.alSourcef(source, ALConstants.AL_GAIN, 1.0f);
        channels[numChannels].volumeChanged = true;

        Com.DPrintf("streaming enabled\n");
    }

    static void disableStreaming() {
        if (!streamingEnabled) return;
        unqueueStreams();
        al.alSourcei(channels[numChannels].sourceId, ALConstants.AL_SOURCE_RELATIVE, ALConstants.AL_FALSE);


        numChannels++;
        streamingEnabled = false;
        Com.DPrintf("streaming disabled\n");
    }

    private static void unqueueStreams() {
        if (tmp == null)
            tmp = new int[1];
        if (!streamingEnabled) return;
        int source = channels[numChannels].sourceId;


        al.alSourceStop(source);
        int[] tmpCount = {0};
        al.alGetSourcei(source, ALConstants.AL_BUFFERS_QUEUED, tmpCount, 0);
        int count = tmpCount[0];
        Com.DPrintf("unqueue " + count + " buffers\n");
        while (count-- > 0) {
            al.alSourceUnqueueBuffers(source, 1, tmp, 0);
        }
        streamQueue = 0;
    }

    static void updateStream(ByteBuffer samples, int count, int format, int rate) {
        if (tmp == null)
            tmp = new int[1];
        enableStreaming();
        int[] buffer = tmp;
        int source = channels[numChannels].sourceId;

        int[] tmp = {0};
        al.alGetSourcei(source, ALConstants.AL_BUFFERS_PROCESSED, tmp, 0);
        int processed = tmp[0];
        al.alGetSourcei(source, ALConstants.AL_SOURCE_STATE, tmp, 0);
        int state = tmp[0];
        boolean playing = (state == ALConstants.AL_PLAYING);
        boolean interupted = !playing && streamQueue > 2;

        if (interupted) {
            unqueueStreams();
            buffer[0] = buffers[Sound.MAX_SFX + streamQueue++];
            Com.DPrintf("queue " + (streamQueue - 1) + '\n');
        } else if (processed < 2) {

            if (streamQueue >= Sound.STREAM_QUEUE) return;
            buffer[0] = buffers[Sound.MAX_SFX + streamQueue++];
            Com.DPrintf("queue " + (streamQueue - 1) + '\n');
        } else {
            al.alSourceUnqueueBuffers(source, 1, buffer, 0);
        }

        samples.position(0);
        samples.limit(count);
        al.alBufferData(buffer[0], format, samples, count, rate);
        al.alSourceQueueBuffers(source, 1, buffer, 0);

        if (streamQueue > 1 && !playing) {
            Com.DPrintf("start sound\n");
            al.alSourcePlay(source);
        }
    }

    static void addPlaySounds() {
        while (assign(PlaySound.nextPlayableSound())) ;
    }

    private static boolean assign(PlaySound ps) {
        if (ps == null) return false;
        Channel ch = null;
        int i;
        for (i = 0; i < numChannels; i++) {
            ch = channels[i];

            if (ps.entchannel != 0 && ch.entnum == ps.entnum && ch.entchannel == ps.entchannel) {

                if (ch.bufferId != ps.bufferId)
                    al.alSourceStop(ch.sourceId);

                break;
            }


            if ((ch.entnum == Globals.cl.playernum + 1) && (ps.entnum != Globals.cl.playernum + 1) && ch.bufferId != -1)
                continue;


            if (!ch.active)
                break;

        }

        if (i == numChannels)
            return false;

        ch.type = ps.type;
        if (ps.type == FIXED)
            Math3D.VectorCopy(ps.origin, ch.origin);
        ch.entnum = ps.entnum;
        ch.entchannel = ps.entchannel;
        ch.bufferChanged = (ch.bufferId != ps.bufferId);
        ch.bufferId = ps.bufferId;
        ch.volumeChanged = (ch.volume != ps.volume);
        ch.volume = ps.volume;
        ch.rolloff = ps.attenuation * 2;
        ch.active = true;
        ch.modified = true;
        return true;
    }

    private static Channel pickForLoop(int bufferId, float attenuation) {
        for (int i = 0; i < numChannels; i++) {
            Channel ch = channels[i];
            if (!ch.active) {
                ch.entnum = 0;
                ch.entchannel = 0;
                ch.bufferChanged = (ch.bufferId != bufferId);
                ch.bufferId = bufferId;
                ch.volumeChanged = (ch.volume < 1);
                ch.volume = 1;
                ch.rolloff = attenuation * 2;
                ch.active = true;
                ch.modified = true;
                return ch;
            }
        }
        return null;
    }


    private static final float[] entityOrigin = {0, 0, 0};
    private static final float[] sourceOrigin = {0, 0, 0};

    static void playAllSounds(float[] listenerOrigin) {
        int[] tmp = {0};

        for (int i = 0; i < numChannels; i++) {
            Channel ch = channels[i];
            if (ch.active) {
                int sourceId = ch.sourceId;
                switch (ch.type) {
                    case LISTENER -> Math3D.VectorCopy(listenerOrigin, sourceOrigin);
                    case DYNAMIC -> {
                        CL_ents.GetEntitySoundOrigin(ch.entnum, entityOrigin);
                        convertVector(entityOrigin, sourceOrigin);
                    }
                    case FIXED -> convertVector(ch.origin, sourceOrigin);
                }

                if (ch.modified) {
                    if (ch.bufferChanged) {
                        try {
                            al.alSourcei(sourceId, ALConstants.AL_BUFFER, ch.bufferId);
                        } catch (ALException e) {

                            al.alSourceStop(sourceId);
                            al.alSourcei(sourceId, ALConstants.AL_BUFFER, ch.bufferId);
                        }
                    }
                    if (ch.volumeChanged) {
                        al.alSourcef(sourceId, ALConstants.AL_GAIN, ch.volume);
                    }
                    al.alSourcef(sourceId, ALConstants.AL_ROLLOFF_FACTOR, ch.rolloff);
                    al.alSourcefv(sourceId, ALConstants.AL_POSITION, sourceOrigin, 0);
                    al.alSourcePlay(sourceId);
                    ch.modified = false;
                } else {
                    al.alGetSourcei(sourceId, ALConstants.AL_SOURCE_STATE, tmp, 0);
                    int state = tmp[0];
                    if (state == ALConstants.AL_PLAYING) {
                        al.alSourcefv(sourceId, ALConstants.AL_POSITION, sourceOrigin, 0);
                    } else {
                        ch.clear();
                    }
                }
                ch.autosound = false;
            }
        }
    }

    /*
     * 	adddLoopSounds
     * 	Entities with a ->sound field will generated looped sounds
     * 	that are automatically started, stopped, and merged together
     * 	as the entities are sent to the client
     */
    static void addLoopSounds() {

        if ((Globals.cl_paused.value != 0.0f) || (Globals.cls.state != Defines.ca_active) || !Globals.cl.sound_prepped) {
            removeUnusedLoopSounds();
            return;
        }

        int sound;

        for (int i = 0; i < Globals.cl.frame.num_entities; i++) {
            int num = (Globals.cl.frame.parse_entities + i) & (Defines.MAX_PARSE_ENTITIES - 1);
            entity_state_t ent = Globals.cl_parse_entities[num];
            sound = ent.sound;

            if (sound == 0) continue;

            Integer key = ent.number;
            Channel ch = looptable.get(key);

            if (ch != null) {

                ch.autosound = true;
                Math3D.VectorCopy(ent.origin, ch.origin);
                continue;
            }

            sfx_t sfx = Globals.cl.sound_precache[sound];
            if (sfx == null)
                continue;

            sfxcache_t sc = sfx.cache;
            if (sc == null)
                continue;


            ch = pickForLoop(buffers[sfx.bufferId], 6);
            if (ch == null)
                break;

            ch.type = FIXED;
            Math3D.VectorCopy(ent.origin, ch.origin);
            ch.autosound = true;

            looptable.put(key, ch);
            al.alSourcei(ch.sourceId, ALConstants.AL_LOOPING, ALConstants.AL_TRUE);
        }

        removeUnusedLoopSounds();

    }

    private static void removeUnusedLoopSounds() {

        for (Iterator<Channel> iter = looptable.values().iterator(); iter.hasNext(); ) {
            Channel ch = iter.next();
            if (!ch.autosound) {
                al.alSourceStop(ch.sourceId);
                al.alSourcei(ch.sourceId, ALConstants.AL_LOOPING, ALConstants.AL_FALSE);
                iter.remove();
                ch.clear();
            }
        }
    }


    static void convertVector(float[] from, float[] to) {
        to[0] = from[0];
        to[1] = from[2];
        to[2] = -from[1];
    }

    static void convertOrientation(float[] forward, float[] up, float[] orientation) {
        orientation[0] = forward[0];
        orientation[1] = forward[2];
        orientation[2] = -forward[1];
        orientation[3] = up[0];
        orientation[4] = up[2];
        orientation[5] = -up[1];
    }
}