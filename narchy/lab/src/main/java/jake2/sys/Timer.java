/*
 * Timer.java
 * Copyright (C) 2005
 */
package jake2.sys;

import jake2.Globals;
import jake2.qcommon.Com;


public abstract class Timer {

	protected abstract long currentTimeMillis();
	private static Timer t;
	
	static {
		try {
			t = new NanoTimer();
		} catch (RuntimeException e) {
			/*try {
				t = new HighPrecisionTimer();
			} catch (Throwable e1) {*/
				t = new StandardTimer();
			
		}
		Com.Println("using " + t.getClass().getName());
	}
	
	public static int Milliseconds() {
		return Globals.curtime = (int)(t.currentTimeMillis());
	}
}