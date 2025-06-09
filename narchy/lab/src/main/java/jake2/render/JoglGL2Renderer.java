/*
 * JoglRenderer.java
 * Copyright (C) 2004
 *
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
package jake2.render;

import com.jogamp.nativewindow.util.Dimension;
import jake2.Defines;
import jake2.client.refdef_t;
import jake2.client.refexport_t;
import jake2.render.opengl.JoglGL2Driver;
import jake2.sys.KBD;
import jake2.sys.NEWTKBD;

/**
 * JoglRenderer
 * 
 * @author dsanders/cwei
 */
public final class JoglGL2Renderer extends JoglGL2Driver implements refexport_t, Ref {

    private static final String DRIVER_NAME = "joglgl2";

    private final KBD kbd = new NEWTKBD();

    
    private RenderAPI impl;

    static {
        Renderer.register(new JoglGL2Renderer());
    }

    private JoglGL2Renderer() {
        
    }

    
    
    
    
    

    /** 
     * @see refexport_t#Init()
     */
    @Override
    public boolean Init(int vid_xpos, int vid_ypos) {
        
        impl.setGLDriver(this);
        
        
        if (!impl.R_Init(vid_xpos, vid_ypos)) return false;
        
        activateGLContext(true);
        
        
        return impl.R_Init2();
    }

    /** 
     * @see refexport_t#Shutdown()
     */
    @Override
    public void Shutdown() {
        impl.R_Shutdown();
    }

    /** 
     * @see refexport_t#BeginRegistration(String)
     */
    @Override
    public final void BeginRegistration(String map) {
        activateGLContext(true);
        impl.R_BeginRegistration(map);
    }

    /** 
     * @see refexport_t#RegisterModel(String)
     */
    @Override
    public final model_t RegisterModel(String name) {
        activateGLContext(true);
        return impl.R_RegisterModel(name);
    }

    /** 
     * @see refexport_t#RegisterSkin(String)
     */
    @Override
    public final image_t RegisterSkin(String name) {
        activateGLContext(true);
        return impl.R_RegisterSkin(name);
    }

    /** 
     * @see refexport_t#RegisterPic(String)
     */
    @Override
    public final image_t RegisterPic(String name) {
        activateGLContext(true);
        return impl.Draw_FindPic(name);
    }
    /** 
     * @see refexport_t#SetSky(String, float, float[])
     */
    @Override
    public final void SetSky(String name, float rotate, float[] axis) {
        activateGLContext(true);
        impl.R_SetSky(name, rotate, axis);
    }

    /** 
     * @see refexport_t#EndRegistration()
     */
    @Override
    public final void EndRegistration() {
        activateGLContext(true);
        impl.R_EndRegistration();
    }

    /** 
     * @see refexport_t#RenderFrame(refdef_t)
     */
    @Override
    public final void RenderFrame(refdef_t fd) {
        impl.R_RenderFrame(fd);
    }

    /** 
     * @see refexport_t#DrawGetPicSize(Dimension, String)
     */
    @Override
    public final void DrawGetPicSize(Dimension dim, String name) {
        impl.Draw_GetPicSize(dim, name);
    }

    /** 
     * @see refexport_t#DrawPic(int, int, String)
     */
    @Override
    public final void DrawPic(int x, int y, String name) {
        impl.Draw_Pic(x, y, name);
    }

    /** 
     * @see refexport_t#DrawStretchPic(int, int, int, int, String)
     */
    @Override
    public final void DrawStretchPic(int x, int y, int w, int h, String name) {
        impl.Draw_StretchPic(x, y, w, h, name);
    }

    /** 
     * @see refexport_t#DrawChar(int, int, int)
     */
    @Override
    public final void DrawChar(int x, int y, int num) {
        activateGLContext(true);
        impl.Draw_Char(x, y, num);
    }

    /** 
     * @see refexport_t#DrawTileClear(int, int, int, int, String)
     */
    @Override
    public final void DrawTileClear(int x, int y, int w, int h, String name) {
        impl.Draw_TileClear(x, y, w, h, name);
    }

    /** 
     * @see refexport_t#DrawFill(int, int, int, int, int)
     */
    @Override
    public final void DrawFill(int x, int y, int w, int h, int c) {
        impl.Draw_Fill(x, y, w, h, c);
    }

    /** 
     * @see refexport_t#DrawFadeScreen()
     */
    @Override
    public final void DrawFadeScreen() {
        impl.Draw_FadeScreen();
    }

    /** 
     * @see refexport_t#DrawStretchRaw(int, int, int, int, int, int, byte[])
     */
    @Override
    public final void DrawStretchRaw(int x, int y, int w, int h, int cols, int rows, byte[] data) {
        impl.Draw_StretchRaw(x, y, w, h, cols, rows, data);
    }

    /** 
     * @see refexport_t#CinematicSetPalette(byte[])
     */
    @Override
    public final void CinematicSetPalette(byte[] palette) {
        impl.R_SetPalette(palette);
    }

    /** 
     * @see refexport_t#BeginFrame(float)
     */
    @Override
    public final boolean BeginFrame(float camera_separation) {
        return impl.R_BeginFrame(camera_separation);
    }

    /** 
     * @see refexport_t#EndFrame()
     */
    @Override
    public final void EndFrame() {
        endFrame();
    }

    /** 
     * @see refexport_t#AppActivate(boolean)
     */
    @Override
    public final void AppActivate(boolean activate) {
        appActivate(activate);
    }

    @Override
    public void screenshot() {
        activateGLContext(true);
        impl.GL_ScreenShot_f();
    }

    @Override
    public final int apiVersion() {
        return Defines.API_VERSION;
    }

    @Override
    public KBD getKeyboardHandler() {
        return kbd;
    }
    
    
    

    @Override
    public final String getName() {
        return DRIVER_NAME;
    }

    public final String toString() {
        return DRIVER_NAME;
    }

    @Override
    public final refexport_t GetRefAPI(RenderAPI renderer) {
        this.impl = renderer;
        return this;
    }

}
