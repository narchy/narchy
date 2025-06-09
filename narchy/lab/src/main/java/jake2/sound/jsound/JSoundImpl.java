/*
 * JSoundImpl.java
 * Copyright (C) 2004
 *
 * $Id: JSoundImpl.java,v 1.2 2005-12-04 17:26:55 cawe Exp $
 */
package jake2.sound.jsound;

import jake2.sound.S;
import jake2.sound.Sound;
import jake2.sound.sfx_t;

import java.nio.ByteBuffer;

/**
 * JSoundImpl
 */
public class JSoundImpl  implements Sound {
	
	static {
		S.register(new JSoundImpl());
	}

	@Override
    public boolean Init() {
		SND_DMA.Init();
		return SND_DMA.sound_started;
	}

	/* (non-Javadoc)
	 * @see jake2.sound.SoundImpl#Shutdown()
	 */
	@Override
    public void Shutdown() {
		SND_DMA.Shutdown();
	}

	/* (non-Javadoc)
	 * @see jake2.sound.SoundImpl#StartSound(float[], int, int, jake2.sound.sfx_t, float, float, float)
	 */
	@Override
    public void StartSound(float[] origin, int entnum, int entchannel, sfx_t sfx, float fvol, float attenuation, float timeofs) {
		SND_DMA.StartSound(origin, entnum, entchannel, sfx, fvol, attenuation, timeofs);
	}

	/* (non-Javadoc)
	 * @see jake2.sound.SoundImpl#StopAllSounds()
	 */
	@Override
    public void StopAllSounds() {
		SND_DMA.StopAllSounds();
	}

	/* (non-Javadoc)
	 * @see jake2.sound.SoundImpl#Update(float[], float[], float[], float[])
	 */
	@Override
    public void Update(float[] origin, float[] forward, float[] right, float[] up) {
		SND_DMA.Update(origin, forward, right, up);
	}

	/* (non-Javadoc)
	 * @see jake2.sound.Sound#getName()
	 */
	@Override
    public String getName() {
		return "jsound";
	}

	/* (non-Javadoc)
	 * @see jake2.sound.Sound#BeginRegistration()
	 */
	@Override
    public void BeginRegistration() {
		SND_DMA.BeginRegistration();
	}

	/* (non-Javadoc)
	 * @see jake2.sound.Sound#RegisterSound(java.lang.String)
	 */
	@Override
    public sfx_t RegisterSound(String sample) {
		return SND_DMA.RegisterSound(sample);
	}

	/* (non-Javadoc)
	 * @see jake2.sound.Sound#EndRegistration()
	 */
	@Override
    public void EndRegistration() {
		SND_DMA.EndRegistration();
	}

	/* (non-Javadoc)
	 * @see jake2.sound.Sound#StartLocalSound(java.lang.String)
	 */
	@Override
    public void StartLocalSound(String sound) {
		SND_DMA.StartLocalSound(sound);
	}

	/* (non-Javadoc)
	 * @see jake2.sound.Sound#RawSamples(int, int, int, int, byte[])
	 */
	@Override
    public void RawSamples(int samples, int rate, int width, int channels, ByteBuffer data) {
		SND_DMA.RawSamples(samples, rate, width, channels, data);
	}
    
    @Override
    public void disableStreaming() {
    }

}
