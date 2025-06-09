/*
 * SND_MEM.java
 * Copyright (C) 2004
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
package jake2.sound;

import jake2.Defines;
import jake2.qcommon.Com;
import jake2.qcommon.FS;

/**
 * SND_MEM
 */
public class WaveLoader {

	/** 
	 * The ResampleSfx can squeeze and stretch samples to a default sample rate. 
	 * Since Joal and lwjgl sound drivers support this, we don't need it and the samples
	 * can keep their original sample rate. Use this switch for reactivating resampling.
	 */
	private static final boolean DONT_DO_A_RESAMPLING_FOR_JOAL_AND_LWJGL = true;

	/**
	 * Loads a sound from a wav file. 
	 */
	public static sfxcache_t LoadSound(sfx_t s) {
		if (s.name == null || s.name.charAt(0) == '*')
			return null;

		
		sfxcache_t sc = s.cache;
		if (sc != null)
			return sc;

		String name;
		
		if (s.truename != null)
			name = s.truename;
		else
			name = s.name;

		String namebuffer;
		if (name.charAt(0) == '#')
			namebuffer = name.substring(1);
		else
			namebuffer = "sound/" + name;

		byte[] data = FS.LoadFile(namebuffer);

		if (data == null) {
			Com.DPrintf("Couldn't load " + namebuffer + '\n');
			return null;
		}
		
		int size = data.length;

		wavinfo_t info = GetWavinfo(s.name, data, size);

		if (info.channels != 1)
		{
			Com.Printf(s.name + " is a stereo sample - ignoring\n");
			return null;
		}

		float stepscale;
		if (DONT_DO_A_RESAMPLING_FOR_JOAL_AND_LWJGL)
			stepscale = 1; 
		else
			stepscale = (float)info.rate / S.getDefaultSampleRate();
		
		int len = (int) (info.samples / stepscale);
        len *= info.width * info.channels;

		
		/*
	  This is the maximum sample length in bytes which has to be replaced by
	  a configurable variable.
	 */
		int maxsamplebytes = 2048 * 1024;
		if (len >= maxsamplebytes)
		{
			Com.Printf(s.name + " is too long: " + len + " bytes?! ignoring.\n");
			return null;
		}

		sc = s.cache = new sfxcache_t(len);
		         
        sc.length = info.samples;
        sc.loopstart = info.loopstart;
        sc.speed = info.rate;
        sc.width = info.width;
        sc.stereo = info.channels;

		ResampleSfx(s, sc.speed, sc.width, data, info.dataofs);

		return sc;
	}


	/** 
	 * Converts sample data with respect to the endianess and adjusts 
	 * the sample rate of a loaded sample, see flag DONT_DO_A_RESAMPLING_FOR_JOAL_AND_LWJGL.
	 */
	private static void ResampleSfx(sfx_t sfx, int inrate, int inwidth, byte[] data, int offset)
	{

        sfxcache_t sc = sfx.cache;
        
        if (sc == null)
        	return;

        
        
        
        float stepscale;
        if (DONT_DO_A_RESAMPLING_FOR_JOAL_AND_LWJGL)
        	stepscale = 1;
        else
        	stepscale = (float)inrate / S.getDefaultSampleRate();
        int outcount = (int) (sc.length / stepscale);
        sc.length = outcount;
        
        if (sc.loopstart != -1)
            sc.loopstart /= stepscale;

        
        if (!DONT_DO_A_RESAMPLING_FOR_JOAL_AND_LWJGL)
        	sc.speed = S.getDefaultSampleRate();

        sc.width = inwidth;
        sc.stereo = 0;
        int samplefrac = 0;
        int fracstep = (int) (stepscale * 256);
        
        for (int i = 0; i < outcount; i++) {
            int srcsample = samplefrac >> 8;
            samplefrac += fracstep;

            int sample;
            if (inwidth == 2) {
                sample = (data[offset + srcsample * 2] & 0xff)
                        + (data[offset + srcsample * 2 + 1] << 8);
            } else {
                sample = ((data[offset + srcsample] & 0xff) - 128) << 8;
            }

            if (sc.width == 2) {
                if (Defines.LITTLE_ENDIAN) {
                    sc.data[i * 2] = (byte) (sample & 0xff);
                    sc.data[i * 2 + 1] = (byte) ((sample >>> 8) & 0xff);
                } else {
                    sc.data[i * 2] = (byte) ((sample >>> 8) & 0xff);
                    sc.data[i * 2 + 1] = (byte) (sample & 0xff);
                }
            } else {
                sc.data[i] = (byte) (sample >> 8);
            }
        }
    }


	private static byte[] data_b;
	private static int data_p;
	private static int iff_end;
	private static int last_chunk;
	private static int iff_data;


	private static short GetLittleShort() {
        int val = data_b[data_p] & 0xFF;
		data_p++;
		val |= ((data_b[data_p] & 0xFF) << 8);
		data_p++;
		return (short)val;
	}

	private static int GetLittleLong() {
        int val = data_b[data_p] & 0xFF;
		data_p++;
		val |= ((data_b[data_p] & 0xFF) << 8);
		data_p++;
		val |= ((data_b[data_p] & 0xFF) << 16);
		data_p++;
		val |= ((data_b[data_p] & 0xFF) << 24);
		data_p++;
		return val;
	}

	private static void FindNextChunk(String name) {
		while (true) {
			data_p = last_chunk;

			if (data_p >= iff_end) { 
				data_p = 0;
				return;
			}

			data_p += 4;

			int iff_chunk_len = GetLittleLong();
			
			if (iff_chunk_len < 0) {
				data_p = 0;
				return;
			}
			if (iff_chunk_len > 1024*1024) {
				Com.Println(" Warning: FindNextChunk: length is past the 1 meg sanity limit");
			}
			data_p -= 8;
			last_chunk = data_p + 8 + ((iff_chunk_len + 1) & ~1);
			String s = new String(data_b, data_p, 4);
			if (s.equals(name))
				return;
		}
	}

	private static void FindChunk(String name) {
		last_chunk = iff_data;
		FindNextChunk(name);
	}

	/*
	============
	GetWavinfo
	============
	*/
	private static wavinfo_t GetWavinfo(String name, byte[] wav, int wavlength) {
		wavinfo_t info = new wavinfo_t();

        if (wav == null)
			return info;

		iff_data = 0;
		iff_end = wavlength;
		data_b = wav;

		
		FindChunk("RIFF");
		String s = new String(data_b, data_p + 8, 4);
		if (!"WAVE".equals(s)) {
			Com.Printf("Missing RIFF/WAVE chunks\n");
			return info;
		}

		
		iff_data = data_p + 12;
		

		FindChunk("fmt ");
		if (data_p == 0) {
			Com.Printf("Missing fmt chunk\n");
			return info;
		}
		data_p += 8;
        int format = GetLittleShort();
		if (format != 1) {
			Com.Printf("Microsoft PCM format only\n");
			return info;
		}

		info.channels = GetLittleShort();
		info.rate = GetLittleLong();
		data_p += 4 + 2;
		info.width = GetLittleShort() / 8;

		
		FindChunk("cue ");
		if (data_p != 0) {
			data_p += 32;
			info.loopstart = GetLittleLong();
			

			
			FindNextChunk("LIST");
			if (data_p != 0) {
				if (data_b.length >= data_p + 32) {
					s = new String(data_b, data_p + 28, 4);
					if ("MARK".equals(s)) {
											
						data_p += 24;
                        int i = GetLittleLong();
                        info.samples = info.loopstart + i;
						
					}
				}
			}
		} else
			info.loopstart = -1;

		
		FindChunk("data");
		if (data_p == 0) {
			Com.Printf("Missing data chunk\n");
			return info;
		}

		data_p += 4;
        int samples = GetLittleLong() / info.width;

		if (info.samples != 0) {
			if (samples < info.samples)
				Com.Error(Defines.ERR_DROP, "Sound " + name + " has a bad loop length");
		} else {
			info.samples = samples;
			if (info.loopstart > 0) info.samples -= info.loopstart;
		}
		
		info.dataofs = data_p;

		return info;
	}

	static class wavinfo_t {
		int rate;
		int width;
		int channels;
		int loopstart;
		int samples;
		int dataofs; 
	}
}