/*
 * Qcommon.java
 * Copyright 2003
 * 
 * $Id: Qcommon.java,v 1.23 2006-08-20 21:46:07 salomo Exp $
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
package jake2.qcommon;

import jake2.Globals;
import jake2.Jake2;
import jake2.client.CL;
import jake2.client.Key;
import jake2.client.SCR;
import jake2.game.Cmd;
import jake2.server.SV_MAIN;
import jake2.sys.NET;
import jake2.sys.Sys;
import jake2.sys.Timer;
import jake2.util.Vargs;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Qcommon contains some  basic routines for the game engine
 * namely initialization, shutdown and frame generation.
 */
public final class Qcommon extends Globals {

	private static final String BUILDSTRING = "Java " + System.getProperty("java.version");
	private static final String CPUSTRING = System.getProperty("os.arch");

	/**
	 * This function initializes the different subsystems of
	 * the game engine. The setjmp/longjmp mechanism of the original
	 * was replaced with exceptions.
	 * @param args the original unmodified command line arguments
	 */
	public static void Init(String[] args) {
		try {

			
			
			Com.InitArgv(args);

			Cbuf.Init();
			
			Cmd.Init();
			Cvar.Init();

			Key.Init();

			
			
			
			
			Cbuf.AddEarlyCommands(false);
			Cbuf.Execute();

			if (Globals.dedicated.value != 1.0f) {
				Q2DataTool.setStatus("initializing filesystem...");
	                }
			
			FS.InitFilesystem();

			if (Globals.dedicated.value != 1.0f) {
				Q2DataTool.setStatus("loading config...");
			}
			
			reconfigure(false);

			FS.setCDDir(); 
			FS.markBaseSearchPaths(); 

			if (Globals.dedicated.value != 1.0f) {
			    Jake2.q2DataTool.testQ2Data(); 
                        }
			
			reconfigure(true); 
			
			
			
			
			Cmd.AddCommand("error", Com.Error_f);

			Globals.host_speeds= Cvar.Get("host_speeds", "0", 0);
			Globals.log_stats= Cvar.Get("log_stats", "0", 0);
			Globals.developer= Cvar.Get("developer", "0", CVAR_ARCHIVE);
			Globals.timescale= Cvar.Get("timescale", "0", 0);
			Globals.fixedtime= Cvar.Get("fixedtime", "0", 0);
			Globals.logfile_active= Cvar.Get("logfile", "0", 0);
			Globals.showtrace= Cvar.Get("showtrace", "0", 0);
			Globals.dedicated= Cvar.Get("dedicated", "0", CVAR_NOSET);
			String s = Com.sprintf("%4.2f %s %s %s",
					new Vargs(4)
						.add(Globals.VERSION)
						.add(CPUSTRING)
						.add(Globals.__DATE__)
						.add(BUILDSTRING));

			Cvar.Get("version", s, CVAR_SERVERINFO | CVAR_NOSET);

			if (Globals.dedicated.value != 1.0f) {
				Q2DataTool.setStatus("initializing network subsystem...");
			}
			
			NET.Init();	
			Netchan.Netchan_Init();	

			if (Globals.dedicated.value != 1.0f) {			
				Q2DataTool.setStatus("initializing server subsystem...");
			}
			SV_MAIN.SV_Init();	
			
			if (Globals.dedicated.value != 1.0f) {
				Q2DataTool.setStatus("initializing client subsystem...");
			}
			
			CL.Init();

			
			if (!Cbuf.AddLateCommands()) {
				
			      if (Globals.dedicated.value == 0)
			          Cbuf.AddText ("d1\n");
			      else
			          Cbuf.AddText ("dedicated_start\n");
			          
				Cbuf.Execute();
			} else {
				
				
				SCR.EndLoadingPlaque();
			}

			Com.Printf("====== Quake2 Initialized ======\n\n");

			
			CL.WriteConfiguration();
						
			if (Globals.dedicated.value != 1.0f) {
			    Jake2.q2DataTool.dispose();
			}

		} catch (longjmpException e) {
			Sys.Error("Error during initialization");
		}
	}

	/**
	 * Trigger generation of a frame for the given time. The setjmp/longjmp
	 * mechanism of the original was replaced with exceptions.
	 * @param msec the current game time
	 */
	public static void Frame(int msec) {
		try {

			if (Globals.log_stats.modified) {
				Globals.log_stats.modified= false;

				if (Globals.log_stats.value != 0.0f) {

					if (Globals.log_stats_file != null) {
						try {
							Globals.log_stats_file.close();
						} catch (IOException e) {
						}
						Globals.log_stats_file= null;
					}

					try {
						Globals.log_stats_file= new FileWriter("stats.log");
					} catch (IOException e) {
						Globals.log_stats_file= null;
					}
					if (Globals.log_stats_file != null) {
						try {
							Globals.log_stats_file.write("entities,dlights,parts,frame time\n");
						} catch (IOException e) {
						}
					}

				} else {

					if (Globals.log_stats_file != null) {
						try {
							Globals.log_stats_file.close();
						} catch (IOException e) {
						}
						Globals.log_stats_file= null;
					}
				}
			}

			if (Globals.fixedtime.value != 0.0f) {
				msec= (int) Globals.fixedtime.value;
			} else if (Globals.timescale.value != 0.0f) {
				msec *= Globals.timescale.value;
				if (msec < 1)
					msec= 1;
			}

			if (Globals.showtrace.value != 0.0f) {
				Com.Printf("%4i traces  %4i points\n",
					new Vargs(2).add(Globals.c_traces)
								.add(Globals.c_pointcontents));

				
				Globals.c_traces= 0;
				Globals.c_brush_traces= 0;
				Globals.c_pointcontents= 0;
			}

			Cbuf.Execute();

			int time_before= 0;

            if (Globals.host_speeds.value != 0.0f)
				time_before= Timer.Milliseconds();
			
			Com.debugContext = "SV:";
			SV_MAIN.SV_Frame(msec);

            int time_between = 0;
            if (Globals.host_speeds.value != 0.0f)
				time_between= Timer.Milliseconds();
			
			Com.debugContext = "CL:";
			CL.Frame(msec);

			if (Globals.host_speeds.value != 0.0f) {
                int time_after = Timer.Milliseconds();

                int all= time_after - time_before;
				int sv= time_between - time_before;
				int cl= time_after - time_between;
				int gm= Globals.time_after_game - Globals.time_before_game;
				int rf= Globals.time_after_ref - Globals.time_before_ref;
				sv -= gm;
				cl -= rf;

				Com.Printf("all:%3i sv:%3i gm:%3i cl:%3i rf:%3i\n",
					new Vargs(5).add(all).add(sv).add(gm).add(cl).add(rf));
			}

		} catch (longjmpException e) {
			Com.DPrintf("longjmp exception:" + e);
		}
	}

	private static void reconfigure(boolean clear) {
		String dir = Cvar.Get("cddir", "", CVAR_ARCHIVE).string;
		Cbuf.AddText("exec default.cfg\n");
		Cbuf.AddText("bind MWHEELUP weapnext\n");
		Cbuf.AddText("bind MWHEELDOWN weapprev\n");
		Cbuf.AddText("bind w +forward\n");
		Cbuf.AddText("bind s +back\n");
		Cbuf.AddText("bind a +moveleft\n");
		Cbuf.AddText("bind d +moveright\n");
		Cbuf.Execute();
		Cvar.Set("vid_fullscreen", "0");
		Cbuf.AddText("exec config.cfg\n");

		Cbuf.AddEarlyCommands(clear);
		Cbuf.Execute();
		if (!(dir != null && dir.isEmpty())) Cvar.Set("cddir", dir);
	}
}