/*
 * VID.java
 * Copyright (C) 2003
 *
 * $Id: VID.java,v 1.21 2008-03-02 14:56:22 cawe Exp $
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
package jake2.client;

import com.jogamp.nativewindow.util.Dimension;
import com.jogamp.nativewindow.util.DimensionImmutable;
import com.jogamp.nativewindow.util.SurfaceSize;
import com.jogamp.newt.MonitorMode;
import jake2.Defines;
import jake2.Globals;
import jake2.game.Cmd;
import jake2.game.cvar_t;
import jake2.qcommon.Com;
import jake2.qcommon.Cvar;
import jake2.qcommon.xcommand_t;
import jake2.render.Renderer;
import jake2.sound.S;
import jake2.sys.IN;
import jake2.util.Vargs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * VID is a video driver.
 * 
 * source: client/vid.h linux/vid_so.c
 * 
 * @author cwei
 */
public class VID extends Globals {
	
	
	

	
	
	

	
	

	
	private static cvar_t vid_gamma;
	private static cvar_t vid_ref;
	private static cvar_t vid_xpos;
	private static cvar_t vid_ypos;
	private static cvar_t vid_width;
	private static cvar_t vid_height;
	private static cvar_t vid_fullscreen;

	
	
	private static boolean reflib_active;
	

	/*
	==========================================================================

	DLL GLUE

	==========================================================================
	*/

	public static void Printf(int print_level, String fmt) {
		Printf(print_level, fmt, null);	
	}

	public static void Printf(int print_level, String fmt, Vargs vargs) {
		
		if (print_level == Defines.PRINT_ALL)
			Com.Printf(fmt, vargs);
		else
			Com.DPrintf(fmt, vargs);
	}

	

	/*
	============
	VID_Restart_f

	Console command to re-start the video mode and refresh DLL. We do this
	simply by setting the modified flag for the vid_ref variable, which will
	cause the entire video mode and refresh DLL to be reset on the next frame.
	============
	*/
	private static void Restart_f() {
		vid_modes[11].width = (int) vid_width.value;
		vid_modes[11].height = (int) vid_height.value;

		vid_ref.modified = true;
	}

	/*
	** VID_GetModeInfo
	*/
	private static final vidmode_t[] vid_modes =
		{
			new vidmode_t("Mode 0: 320x240", 320, 240, 0),
			new vidmode_t("Mode 1: 400x300", 400, 300, 1),
			new vidmode_t("Mode 2: 512x384", 512, 384, 2),
			new vidmode_t("Mode 3: 640x480", 640, 480, 3),
			new vidmode_t("Mode 4: 800x600", 800, 600, 4),
			new vidmode_t("Mode 5: 960x720", 960, 720, 5),
			new vidmode_t("Mode 6: 1024x768", 1024, 768, 6),
			new vidmode_t("Mode 7: 1152x864", 1152, 864, 7),
			new vidmode_t("Mode 8: 1280x1024", 1280, 1024, 8),
			new vidmode_t("Mode 9: 1600x1200", 1600, 1200, 9),
			new vidmode_t("Mode 10: 2048x1536", 2048, 1536, 10),
			new vidmode_t("Mode 11: user", 640, 480, 11)};
    private static vidmode_t[] fs_modes;

	public static boolean GetModeInfo(Dimension dim, int mode) {
		if (fs_modes == null) initModeList();

		vidmode_t[] modes = vid_modes;
		if (vid_fullscreen.value != 0.0f) modes = fs_modes;
		
		if (mode < 0 || mode >= modes.length) 
			return false;
			
		dim.setWidth( modes[mode].width );
		dim.setHeight( modes[mode].height );
		
		return true;
	}

	/*
	** VID_NewWindow
	*/
	public static void NewWindow(int width, int height) {
		Globals.viddef.setSize(width, height);
	}

	private static void FreeReflib()
	{
		if (Globals.re != null) {
			Globals.re.getKeyboardHandler().Close();
			IN.Shutdown();
		}

		Globals.re = null;
		reflib_active = false;
	}

	/*
	==============
	VID_LoadRefresh
	==============
	*/
	private static boolean LoadRefresh(String name, boolean fast)
	{

		if ( reflib_active )
		{
			Globals.re.getKeyboardHandler().Close();
			IN.Shutdown();

			Globals.re.Shutdown();
			FreeReflib();
		}

		Com.Printf( "------- Loading " + name + " -------\n");

        String[] driverNames = Renderer.getDriverNames();
        boolean found = Arrays.asList(driverNames).contains(name);

		if (!found) {
			Com.Printf( "LoadLibrary(\"" + name +"\") failed\n");
			return false;
		}

		Com.Printf( "LoadLibrary(\"" + name +"\")\n" );
		Globals.re = Renderer.getDriver(name, fast);
		
		if (Globals.re == null)
		{
			Com.Error(Defines.ERR_FATAL, name + " can't load but registered");
		}

		if (Globals.re.apiVersion() != Defines.API_VERSION)
		{
			FreeReflib();
			Com.Error(Defines.ERR_FATAL, name + " has incompatible api_version");
		}

		IN.Real_IN_Init();

		if ( !Globals.re.Init((int)vid_xpos.value, (int)vid_ypos.value) )
		{
			Globals.re.Shutdown();
			FreeReflib();
			return false;
		}

		/* Init KBD */
		Globals.re.getKeyboardHandler().Init();

		Com.Printf( "------------------------------------\n");
		reflib_active = true;
		return true;
	}

	/*
	============
	VID_CheckChanges

	This function gets called once just before drawing each frame, and it's sole purpose in life
	is to check to see if any of the video mode parameters have changed, and if they have to 
	update the rendering DLL and/or video mode to match.
	============
	*/
	public static void CheckChanges()
	{
	    Globals.viddef.update();
	    
		if ( vid_ref.modified )
		{
			S.StopAllSounds();
		}

		while (vid_ref.modified)
		{
			/*
			** refresh has changed
			*/
			vid_ref.modified = false;
			vid_fullscreen.modified = true;
			Globals.cl.refresh_prepped = false;
			Globals.cls.disable_screen = 1.0f; 

			
			if ( !LoadRefresh( vid_ref.string, true ) )
			{
				String renderer;
				if (vid_ref.string.equals(Renderer.getPreferedName())) {
				    
				    renderer = Renderer.getDefaultName();
				} else {
				    
				    renderer = Renderer.getPreferedName();
				}
				if ( vid_ref.string.equals(Renderer.getDefaultName())) {
				    renderer = vid_ref.string;
					Com.Printf("Refresh failed\n");
					gl_mode = Cvar.Get( "gl_mode", "0", 0 );
					if (gl_mode.value != 0.0f) {
						Com.Printf("Trying mode 0\n");
						Cvar.SetValue("gl_mode", 0);
						if ( !LoadRefresh( vid_ref.string, false ) )
							Com.Error(Defines.ERR_FATAL, "Couldn't fall back to " + renderer +" refresh!");
					} else
						Com.Error(Defines.ERR_FATAL, "Couldn't fall back to " + renderer +" refresh!");
				}

				Cvar.Set("vid_ref", renderer);

				/*
				 * drop the console if we fail to load a refresh
				 */
				if ( Globals.cls.key_dest != Defines.key_console )
				{
					try {
						Console.ToggleConsole_f.execute();
					} catch (Exception e) {
					}
				}
			}
			Globals.cls.disable_screen = 0.0f; 
		}
	}

	/*
	============
	VID_Init
	============
	*/
	public static void Init()
	{
		/* Create the video variables so we know how to start the graphics drivers */
		vid_ref = Cvar.Get("vid_ref", Renderer.getPreferedName(), CVAR_ARCHIVE);
		vid_xpos = Cvar.Get("vid_xpos", "3", CVAR_ARCHIVE);
		vid_ypos = Cvar.Get("vid_ypos", "22", CVAR_ARCHIVE);
		vid_width = Cvar.Get("vid_width", "640", CVAR_ARCHIVE);
		vid_height = Cvar.Get("vid_height", "480", CVAR_ARCHIVE);
		vid_fullscreen = Cvar.Get("vid_fullscreen", "0", CVAR_ARCHIVE);
		vid_gamma = Cvar.Get( "vid_gamma", "1", CVAR_ARCHIVE );

		vid_modes[11].width = (int)vid_width.value;
		vid_modes[11].height = (int)vid_height.value;
		
		/* Add some console commands that we want to handle */
		Cmd.AddCommand ("vid_restart", new xcommand_t() {
			@Override
            public void execute() {
				Restart_f();
			}
		});
		
		/* Disable the 3Dfx splash screen */
		
		
		/* Start the graphics mode and load refresh DLL */
                vid_ref.modified = true;
		CheckChanges();
	}

	/*
	============
	VID_Shutdown
	============
	*/
	public static void Shutdown()
	{
		if ( reflib_active )
		{
			Globals.re.getKeyboardHandler().Close();
			IN.Shutdown();

			Globals.re.Shutdown();
			FreeReflib();
		}
	}

	
	
	
	
	

	static final int REF_OPENGL_JOGL = 0;
	static final int REF_OPENGL_FASTJOGL =1;
	static final int REF_OPENGL_LWJGL =2;

	private static cvar_t gl_mode;
	private static cvar_t gl_driver;
	private static cvar_t gl_picmip;
	private static cvar_t gl_ext_palettedtexture;
	private static cvar_t gl_swapinterval;

	/*
	====================================================================

	MENU INTERACTION

	====================================================================
	*/

	private static final Menu.menuframework_s	s_opengl_menu = new Menu.menuframework_s();
	private static Menu.menuframework_s s_current_menu;

	private static final Menu.menulist_s s_mode_list = new Menu.menulist_s();

	private static final Menu.menulist_s s_ref_list = new Menu.menulist_s();

	private static final Menu.menuslider_s s_tq_slider = new Menu.menuslider_s();
	private static final Menu.menuslider_s s_screensize_slider = new Menu.menuslider_s();

	private static final Menu.menuslider_s s_brightness_slider = new Menu.menuslider_s();

	private static final Menu.menulist_s s_fs_box = new Menu.menulist_s();

	static Menu.menulist_s s_stipple_box = new Menu.menulist_s();
	private static final Menu.menulist_s s_paletted_texture_box = new Menu.menulist_s();
	private static final Menu.menulist_s s_vsync_box = new Menu.menulist_s();
	static Menu.menulist_s s_windowed_mouse = new Menu.menulist_s();
	private static final Menu.menuaction_s s_apply_action = new Menu.menuaction_s();

	private static final Menu.menuaction_s s_defaults_action= new Menu.menuaction_s();

	private static void DriverCallback(Object unused)
	{
		s_current_menu = s_opengl_menu; 
	}

	private static void ScreenSizeCallback(Object s)
	{
		Menu.menuslider_s slider = (Menu.menuslider_s) s;

		Cvar.SetValue( "viewsize", slider.curvalue * 10 );
	}

	private static void BrightnessCallback(Object s)
	{
		Menu.menuslider_s slider = (Menu.menuslider_s) s;

		
		
		if ( "soft".equalsIgnoreCase(vid_ref.string) ||
			 "softx".equalsIgnoreCase(vid_ref.string) )
		{
			float gamma = ( 0.8f - ( slider.curvalue/10.0f - 0.5f ) ) + 0.5f;

			Cvar.SetValue( "vid_gamma", gamma );
		}
	}

	private static void ResetDefaults(Object unused)
	{
		MenuInit();
	}

	private static void ApplyChanges(Object unused)
	{

		/*
		** invert sense so greater = brighter, and scale to a range of 0.5 to 1.3
		*/
		
		
		
		float gamma = ( 0.4f - ( s_brightness_slider.curvalue/20.0f - 0.25f ) ) + 0.7f;
		
		float modulate = s_brightness_slider.curvalue * 0.2f;

		Cvar.SetValue( "vid_gamma", gamma );
		Cvar.SetValue( "gl_modulate", modulate);
		Cvar.SetValue( "gl_picmip", 3 - s_tq_slider.curvalue );
		Cvar.SetValue( "vid_fullscreen", s_fs_box.curvalue );
		Cvar.SetValue( "gl_swapinterval", s_vsync_box.curvalue);
		
		gl_swapinterval.modified = true;
		Cvar.SetValue( "gl_ext_palettedtexture", s_paletted_texture_box.curvalue );
		Cvar.SetValue( "gl_mode", s_mode_list.curvalue );

		Cvar.Set( "vid_ref", drivers[s_ref_list.curvalue] );
		Cvar.Set( "gl_driver", drivers[s_ref_list.curvalue] );
		if (gl_driver.modified)
			vid_ref.modified = true;
		
		Menu.ForceMenuOff();
	}

	private static final String[] resolutions =
	{
		"[320 240  ]",
		"[400 300  ]",
		"[512 384  ]",
		"[640 480  ]",
		"[800 600  ]",
		"[960 720  ]",
		"[1024 768 ]",
		"[1152 864 ]",
		"[1280 1024]",
		"[1600 1200]",
		"[2048 1536]",
		"user mode",
	};
	private static String[] fs_resolutions;
	private static int mode_x;
	
	private static String[] refs;
	private static String[] drivers;
	
	private static final String[] yesno_names =
	{
		"no",
		"yes",
	};

	private static void initModeList() {
	        List<MonitorMode> modes = re.getModeList();
                ArrayList<String> fs_resolutions_list = new ArrayList<>();
                ArrayList<vidmode_t> fs_modes_list = new ArrayList<>();
                HashSet<DimensionImmutable> resSet = new HashSet<>();
		for (int i = 0; i < modes.size(); i++) {
		    MonitorMode mm = modes.get(modes.size() - 1 - i);
                    SurfaceSize ss = mm.getSurfaceSize();
                    DimensionImmutable m = ss.getResolution();
                    if( resSet.add(m) ) {
                        StringBuilder sb = new StringBuilder();
			sb.append('[');
			sb.append(m.getWidth());
			sb.append(' ');
			sb.append(m.getHeight());
			while (sb.length() < 10) sb.append(' ');
			sb.append(']');
			fs_resolutions_list.add(sb.toString());
			sb.setLength(0);
			sb.append("Mode ");
			sb.append(i);
			sb.append(':');
			sb.append(m.getWidth());
			sb.append('x');
			sb.append(m.getHeight());
			fs_modes_list.add(new vidmode_t(sb.toString(), m.getWidth(), m.getHeight(), i));
                    }
		}
                fs_resolutions = new String[fs_resolutions_list.size()];
                fs_modes = new vidmode_t[fs_modes_list.size()];
                fs_resolutions_list.toArray(fs_resolutions);
                fs_modes_list.toArray(fs_modes);
	}
	
	private static void initRefs() {
		drivers = Renderer.getDriverNames();
		refs = new String[drivers.length];
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < drivers.length; i++) {
			sb.setLength(0);
			sb.append("[OpenGL ").append(drivers[i]);
			while (sb.length() < 16) sb.append(' ');
			sb.append(']');
			refs[i] = sb.toString();
		}
	}

	/*
	** VID_MenuInit
	*/
	public static void MenuInit() {
		
		initRefs();
		
		if ( gl_driver == null )
			gl_driver = Cvar.Get( "gl_driver", Renderer.getPreferedName(), 0 );
		if ( gl_picmip == null )
			gl_picmip = Cvar.Get( "gl_picmip", "0", 0 );
		if ( gl_mode == null)
			gl_mode = Cvar.Get( "gl_mode", "3", 0 );
		if ( gl_ext_palettedtexture == null )
			gl_ext_palettedtexture = Cvar.Get( "gl_ext_palettedtexture", "1", CVAR_ARCHIVE );

		if ( gl_swapinterval == null)
			gl_swapinterval = Cvar.Get( "gl_swapinterval", "0", CVAR_ARCHIVE );

		s_mode_list.curvalue = (int)gl_mode.value;
		if (vid_fullscreen.value != 0.0f) {
			s_mode_list.itemnames = fs_resolutions;
			if (s_mode_list.curvalue >= fs_resolutions.length - 1) {
				s_mode_list.curvalue = 0;
			}
			mode_x = fs_modes[s_mode_list.curvalue].width;
		} else {
			s_mode_list.itemnames = resolutions;
			if (s_mode_list.curvalue >= resolutions.length - 1) {
				s_mode_list.curvalue = 0;
			}
			mode_x = vid_modes[s_mode_list.curvalue].width;
		}

		if ( SCR.scr_viewsize == null )
			SCR.scr_viewsize = Cvar.Get ("viewsize", "100", CVAR_ARCHIVE);

		s_screensize_slider.curvalue = (int)(SCR.scr_viewsize.value/10);

		for (int i = 0; i < drivers.length; i++) {
			if (vid_ref.string.equals(drivers[i])) {
				s_ref_list.curvalue = i;
			}
		}

		s_opengl_menu.x = (int)(viddef.getWidth() * 0.50f);
		s_opengl_menu.nitems = 0;
		
		s_ref_list.type = MTYPE_SPINCONTROL;
		s_ref_list.name = "driver";
		s_ref_list.x = 0;
		s_ref_list.y = 0;
		s_ref_list.callback =  VID::DriverCallback;
		s_ref_list.itemnames = refs;

		s_mode_list.type = MTYPE_SPINCONTROL;
		s_mode_list.name = "video mode";
		s_mode_list.x = 0;
		s_mode_list.y = 10;

		s_screensize_slider.type	= MTYPE_SLIDER;
		s_screensize_slider.x		= 0;
		s_screensize_slider.y		= 20;
		s_screensize_slider.name	= "screen size";
		s_screensize_slider.minvalue = 3;
		s_screensize_slider.maxvalue = 12;
		s_screensize_slider.callback =  VID::ScreenSizeCallback;
		s_brightness_slider.type	= MTYPE_SLIDER;
		s_brightness_slider.x	= 0;
		s_brightness_slider.y	= 30;
		s_brightness_slider.name	= "brightness";
		s_brightness_slider.callback =  VID::BrightnessCallback;
		s_brightness_slider.minvalue = 5;
		s_brightness_slider.maxvalue = 13;
		s_brightness_slider.curvalue = ( 1.3f - vid_gamma.value + 0.5f ) * 10;

		s_fs_box.type = MTYPE_SPINCONTROL;
		s_fs_box.x	= 0;
		s_fs_box.y	= 40;
		s_fs_box.name	= "fullscreen";
		s_fs_box.itemnames = yesno_names;
		s_fs_box.curvalue = (int)vid_fullscreen.value;
		s_fs_box.callback = o -> {
			int fs = ((Menu.menulist_s)o).curvalue;
			if (fs == 0) {
				s_mode_list.itemnames = resolutions;
				int i = vid_modes.length - 2;
				while (i > 0 && vid_modes[i].width > mode_x) i--;
				s_mode_list.curvalue = i;
			} else {
				s_mode_list.itemnames = fs_resolutions;
				int i = fs_modes.length - 1;
				while (i > 0 && fs_modes[i].width > mode_x) i--;
				s_mode_list.curvalue = i;
			}
		};

		s_tq_slider.type	= MTYPE_SLIDER;
		s_tq_slider.x		= 0;
		s_tq_slider.y		= 60;
		s_tq_slider.name	= "texture quality";
		s_tq_slider.minvalue = 0;
		s_tq_slider.maxvalue = 3;
		s_tq_slider.curvalue = 3 - gl_picmip.value;

		s_paletted_texture_box.type = MTYPE_SPINCONTROL;
		s_paletted_texture_box.x	= 0;
		s_paletted_texture_box.y	= 70;
		s_paletted_texture_box.name	= "8-bit textures";
		s_paletted_texture_box.itemnames = yesno_names;
		s_paletted_texture_box.curvalue = (int)gl_ext_palettedtexture.value;


		s_vsync_box.type = MTYPE_SPINCONTROL;
		s_vsync_box.x	= 0;
		s_vsync_box.y	= 80;
		s_vsync_box.name	= "sync every frame";
		s_vsync_box.itemnames = yesno_names;
		s_vsync_box.curvalue = (int) gl_swapinterval.value;

		s_defaults_action.type = MTYPE_ACTION;
		s_defaults_action.name = "reset to default";
		s_defaults_action.x    = 0;
		s_defaults_action.y    = 100;
		s_defaults_action.callback =  VID::ResetDefaults;

		s_apply_action.type = MTYPE_ACTION;
		s_apply_action.name = "apply";
		s_apply_action.x    = 0;
		s_apply_action.y    = 110;
		s_apply_action.callback =  VID::ApplyChanges;

		Menu.Menu_AddItem( s_opengl_menu, s_ref_list );
		Menu.Menu_AddItem( s_opengl_menu, s_mode_list );
		Menu.Menu_AddItem( s_opengl_menu, s_screensize_slider );
		Menu.Menu_AddItem( s_opengl_menu, s_brightness_slider );
		Menu.Menu_AddItem( s_opengl_menu, s_fs_box );

		Menu.Menu_AddItem( s_opengl_menu, s_tq_slider );
		Menu.Menu_AddItem( s_opengl_menu, s_paletted_texture_box );
		Menu.Menu_AddItem( s_opengl_menu, s_vsync_box );

		Menu.Menu_AddItem( s_opengl_menu, s_defaults_action );
		Menu.Menu_AddItem( s_opengl_menu, s_apply_action );

		Menu.Menu_Center( s_opengl_menu );
		s_opengl_menu.x -= 8;
	}

	/*
	================
	VID_MenuDraw
	================
	*/
	static void MenuDraw()
	{
		s_current_menu = s_opengl_menu;

		/*
		** draw the banner
		*/
		Dimension dim = new Dimension();
		re.DrawGetPicSize( dim, "m_banner_video" );
		re.DrawPic( viddef.getWidth() / 2 - dim.getWidth() / 2, viddef.getHeight() /2 - 110, "m_banner_video" );

		/*
		** move cursor to a reasonable starting position
		*/
		Menu.Menu_AdjustCursor( s_current_menu, 1 );

		/*
		** draw the menu
		*/
		Menu.Menu_Draw( s_current_menu );
	}

	/*
	================
	VID_MenuKey
	================
	*/
	static String MenuKey( int key )
	{
		Menu.menuframework_s m = s_current_menu;

        switch (key) {
            case K_ESCAPE -> {
                Menu.PopMenu();
                return null;
            }
            case K_UPARROW -> {
                m.cursor--;
                Menu.Menu_AdjustCursor(m, -1);
            }
            case K_DOWNARROW -> {
                m.cursor++;
                Menu.Menu_AdjustCursor(m, 1);
            }
            case K_LEFTARROW -> Menu.Menu_SlideItem(m, -1);
            case K_RIGHTARROW -> Menu.Menu_SlideItem(m, 1);
            case K_ENTER -> Menu.Menu_SelectItem(m);
        }

		return "misc/menu1.wav";
	}

}