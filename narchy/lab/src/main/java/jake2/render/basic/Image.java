/*
 * Image.java
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
package jake2.render.basic;

import com.jogamp.nativewindow.util.Dimension;
import jake2.Defines;
import jake2.client.VID;
import jake2.client.particle_t;
import jake2.game.cvar_t;
import jake2.qcommon.*;
import jake2.render.image_t;
import jake2.util.Lib;
import jake2.util.Vargs;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;






/**
 * Image
 * 
 * @author cwei
 */
public abstract class Image extends Main {

    image_t draw_chars;

    final image_t[] gltextures = new image_t[MAX_GLTEXTURES];

    
    int numgltextures;

    int base_textureid; 

    private final byte[] intensitytable = new byte[256];

    private final byte[] gammatable = new byte[256];


    private static final int gl_solid_format = 3;

    private static final int gl_alpha_format = 4;

    int gl_tex_solid_format = 3;

    int gl_tex_alpha_format = 4;

    int gl_filter_min = GL_LINEAR_MIPMAP_NEAREST;

    int gl_filter_max = GL_LINEAR;

    Image() {
        
        for (int i = 0; i < gltextures.length; i++) {
            gltextures[i] = new image_t(i);
        }
        numgltextures = 0;
    }

    @Override
    void GL_SetTexturePalette(int[] palette) {

        assert (palette != null && palette.length == 256) : "int palette[256] bug";

        if (qglColorTableEXT && gl_ext_palettedtexture.value != 0.0f) {
            ByteBuffer temptable = Lib.newByteBuffer(768);
            for (int i = 0; i < 256; i++) {
                temptable.put(i * 3 + 0, (byte) ((palette[i] >> 0) & 0xff));
                temptable.put(i * 3 + 1, (byte) ((palette[i] >> 8) & 0xff));
                temptable.put(i * 3 + 2, (byte) ((palette[i] >> 16) & 0xff));
            }

            gl.glColorTable(GL_SHARED_TEXTURE_PALETTE_EXT, GL_RGB, 256, GL_RGB, GL_UNSIGNED_BYTE, temptable);
        }
    }

    void GL_EnableMultitexture(boolean enable) {
        if (!qglSelectTextureSGIS && !qglActiveTextureARB)
            return;

        if (enable) {
            GL_SelectTexture(GL_TEXTURE1);
            gl.glEnable(GL_TEXTURE_2D);
        } else {
            GL_SelectTexture(GL_TEXTURE1);
            gl.glDisable(GL_TEXTURE_2D);
        }
        GL_TexEnv(GL_REPLACE);
        GL_SelectTexture(GL_TEXTURE0);
        GL_TexEnv(GL_REPLACE);
    }

    void GL_SelectTexture(int texture /* GLenum */) {

        if (!qglSelectTextureSGIS && !qglActiveTextureARB)
            return;

        int tmu;
        if (texture == GL_TEXTURE0) {
            tmu = 0;
        } else {
            tmu = 1;
        }

        if (tmu == gl_state.currenttmu) {
            return;
        }

        gl_state.currenttmu = tmu;

        if (qglSelectTextureSGIS) {
            
            gl.glActiveTextureARB(texture);
        } else if (qglActiveTextureARB) {
            gl.glActiveTextureARB(texture);
            gl.glClientActiveTextureARB(texture);
        }
    }

    private final int[] lastmodes = { -1, -1 };

    @Override
    void GL_TexEnv(int mode /* GLenum */
    ) {

        if (mode != lastmodes[gl_state.currenttmu]) {
            gl.glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, mode);
            lastmodes[gl_state.currenttmu] = mode;
        }
    }

    @Override
    void GL_Bind(int texnum) {

        if ((gl_nobind.value != 0) && (draw_chars != null)) {
            
            texnum = draw_chars.texnum;
        }
        if (gl_state.currenttextures[gl_state.currenttmu] == texnum)
            return;

        gl_state.currenttextures[gl_state.currenttmu] = texnum;
        gl.glBindTexture(GL_TEXTURE_2D, texnum);
    }

    void GL_MBind(int target /* GLenum */
    , int texnum) {
        GL_SelectTexture(target);
        if (target == GL_TEXTURE0) {
            if (gl_state.currenttextures[0] == texnum)
                return;
        } else {
            if (gl_state.currenttextures[1] == texnum)
                return;
        }
        GL_Bind(texnum);
    }

    
    static class glmode_t {
        final String name;

        final int minimize;
        final int maximize;

        glmode_t(String name, int minimize, int maximze) {
            this.name = name;
            this.minimize = minimize;
            this.maximize = maximze;
        }
    }

    private static final glmode_t[] modes = {
            new glmode_t("GL_NEAREST", GL_NEAREST, GL_NEAREST),
            new glmode_t("GL_LINEAR", GL_LINEAR, GL_LINEAR),
            new glmode_t("GL_NEAREST_MIPMAP_NEAREST",
                    GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST),
            new glmode_t("GL_LINEAR_MIPMAP_NEAREST",
                    GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR),
            new glmode_t("GL_NEAREST_MIPMAP_LINEAR",
                    GL_NEAREST_MIPMAP_LINEAR, GL_NEAREST),
            new glmode_t("GL_LINEAR_MIPMAP_LINEAR", GL_LINEAR_MIPMAP_LINEAR,
                    GL_LINEAR)};

    private static final int NUM_GL_MODES = modes.length;

    
    static class gltmode_t {
        final String name;

        final int mode;

        gltmode_t(String name, int mode) {
            this.name = name;
            this.mode = mode;
        }
    }

    private static final gltmode_t[] gl_alpha_modes = { new gltmode_t("default", 4),
            new gltmode_t("GL_RGBA", GL_RGBA),
            new gltmode_t("GL_RGBA8", GL_RGBA8),
            new gltmode_t("GL_RGB5_A1", GL_RGB5_A1),
            new gltmode_t("GL_RGBA4", GL_RGBA4),
            new gltmode_t("GL_RGBA2", GL_RGBA2), };

    private static final int NUM_GL_ALPHA_MODES = gl_alpha_modes.length;

    private static final gltmode_t[] gl_solid_modes = { new gltmode_t("default", 3),
            new gltmode_t("GL_RGB", GL_RGB),
            new gltmode_t("GL_RGB8", GL_RGB8),
            new gltmode_t("GL_RGB5", GL_RGB5),
            new gltmode_t("GL_RGB4", GL_RGB4),
            new gltmode_t("GL_R3_G3_B2", GL_R3_G3_B2),
    };

    private static final int NUM_GL_SOLID_MODES = gl_solid_modes.length;

    /*
     * =============== GL_TextureMode ===============
     */
    @Override
    void GL_TextureMode(String string) {

        int i;
        for (i = 0; i < NUM_GL_MODES; i++) {
            if (modes[i].name.equalsIgnoreCase(string))
                break;
        }

        if (i == NUM_GL_MODES) {
            VID
                    .Printf(Defines.PRINT_ALL, "bad filter name: [" + string
                            + "]\n");
            return;
        }

        gl_filter_min = modes[i].minimize;
        gl_filter_max = modes[i].maximize;

        for (i = 0; i < numgltextures; i++) {
            image_t glt = gltextures[i];

            if (glt.type != it_pic && glt.type != it_sky) {
                GL_Bind(glt.texnum);
                gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        gl_filter_min);
                gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                        gl_filter_max);
            }
        }
    }

    /*
     * =============== GL_TextureAlphaMode ===============
     */
    @Override
    void GL_TextureAlphaMode(String string) {

        int i;
        for (i = 0; i < NUM_GL_ALPHA_MODES; i++) {
            if (gl_alpha_modes[i].name.equalsIgnoreCase(string))
                break;
        }

        if (i == NUM_GL_ALPHA_MODES) {
            VID.Printf(Defines.PRINT_ALL, "bad alpha texture mode name: ["
                    + string + "]\n");
            return;
        }

        gl_tex_alpha_format = gl_alpha_modes[i].mode;
    }

    /*
     * =============== GL_TextureSolidMode ===============
     */
    @Override
    void GL_TextureSolidMode(String string) {
        int i;
        for (i = 0; i < NUM_GL_SOLID_MODES; i++) {
            if (gl_solid_modes[i].name.equalsIgnoreCase(string))
                break;
        }

        if (i == NUM_GL_SOLID_MODES) {
            VID.Printf(Defines.PRINT_ALL, "bad solid texture mode name: ["
                    + string + "]\n");
            return;
        }

        gl_tex_solid_format = gl_solid_modes[i].mode;
    }

    /*
     * =============== GL_ImageList_f ===============
     */
    @Override
    void GL_ImageList_f() {

        VID.Printf(Defines.PRINT_ALL, "------------------\n");
        int texels = 0;

        String[] palstrings = {"RGB", "PAL"};
        for (int i = 0; i < numgltextures; i++) {
            image_t image = gltextures[i];
            if (image.texnum <= 0)
                continue;

            texels += image.upload_width * image.upload_height;
            VID.Printf(Defines.PRINT_ALL, switch (image.type) {
                case it_skin -> "M";
                case it_sprite -> "S";
                case it_wall -> "W";
                case it_pic -> "P";
                default -> " ";
            });

            VID.Printf(Defines.PRINT_ALL, " %3i %3i %s: %s\n", new Vargs(4)
                    .add(image.upload_width).add(image.upload_height).add(
                            palstrings[(image.paletted) ? 1 : 0]).add(
                            image.name));
        }
        VID.Printf(Defines.PRINT_ALL,
                "Total texel count (not counting mipmaps): " + texels + '\n');
    }

    /*
     * =============================================================================
     * 
     * scrap allocation
     * 
     * Allocate all the little status bar objects into a single texture to
     * crutch up inefficient hardware / drivers
     * 
     * =============================================================================
     */

    private static final int MAX_SCRAPS = 1;

    private static final int BLOCK_WIDTH = 256;

    private static final int BLOCK_HEIGHT = 256;

    private final int[][] scrap_allocated = new int[MAX_SCRAPS][BLOCK_WIDTH];

    private final byte[][] scrap_texels = new byte[MAX_SCRAPS][BLOCK_WIDTH * BLOCK_HEIGHT];

    boolean scrap_dirty;

    static class pos_t {
        int x;
        int y;

        pos_t(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    
    private int Scrap_AllocBlock(int w, int h, pos_t pos) {

        for (int texnum = 0; texnum < MAX_SCRAPS; texnum++) {
            int best = BLOCK_HEIGHT;

            int i;
            for (i = 0; i < BLOCK_WIDTH - w; i++) {
                int best2 = 0;

                int j;
                for (j = 0; j < w; j++) {
                    if (scrap_allocated[texnum][i + j] >= best)
                        break;
                    if (scrap_allocated[texnum][i + j] > best2)
                        best2 = scrap_allocated[texnum][i + j];
                }
                if (j == w) { 
                    pos.x = i;
                    pos.y = best = best2;
                }
            }

            if (best + h > BLOCK_HEIGHT)
                continue;

            for (i = 0; i < w; i++)
                scrap_allocated[texnum][pos.x + i] = best + h;

            return texnum;
        }

        return -1;
        
    }

    private int scrap_uploads;

    void Scrap_Upload() {
        scrap_uploads++;
        GL_Bind(TEXNUM_SCRAPS);
        GL_Upload8(scrap_texels[0], BLOCK_WIDTH, BLOCK_HEIGHT, false, false);
        scrap_dirty = false;
    }

    /*
     * =================================================================
     * 
     * PCX LOADING
     * 
     * =================================================================
     */

    /*
     * ============== LoadPCX ==============
     */
    private static byte[] LoadPCX(String filename, byte[][] palette, Dimension dim) {


        byte[] raw = FS.LoadFile(filename);

        if (raw == null) {
            VID.Printf(Defines.PRINT_DEVELOPER, "Bad pcx file " + filename
                    + '\n');
            return null;
        }


        qfiles.pcx_t pcx = new qfiles.pcx_t(raw);

        if (pcx.manufacturer != 0x0a || pcx.version != 5 || pcx.encoding != 1
                || pcx.bits_per_pixel != 8 || pcx.xmax >= 640
                || pcx.ymax >= 480) {

            VID.Printf(Defines.PRINT_ALL, "Bad pcx file " + filename + '\n');
            return null;
        }

        int width = pcx.xmax - pcx.xmin + 1;
        int height = pcx.ymax - pcx.ymin + 1;

        if (palette != null) {
            palette[0] = new byte[768];
            System.arraycopy(raw, raw.length - 768, palette[0], 0, 768);
        }

        if (dim != null) {
            dim.setWidth(width);
            dim.setHeight(height);
        }

        
        
        
        int count = 0;
        byte dataByte;
        int runLength;

        byte[] pix = new byte[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width;) {

                dataByte = pcx.data.get();

                if ((dataByte & 0xC0) == 0xC0) {
                    runLength = dataByte & 0x3F;
                    dataByte = pcx.data.get();
                    
                    while (runLength-- > 0) {
                        pix[count++] = dataByte;
                        x++;
                    }
                } else {
                    
                    pix[count++] = dataByte;
                    x++;
                }
            }
        }
        return pix;
    }

    
    
    
    
    
    
    
    /*
     * ============= LoadTGA =============
     */
    private static byte[] LoadTGA(String name, Dimension dim) {


        byte[] raw = FS.LoadFile(name);

        if (raw == null) {
            VID.Printf(Defines.PRINT_DEVELOPER, "Bad tga file " + name + '\n');
            return null;
        }

        qfiles.tga_t targa_header = new qfiles.tga_t(raw);

        if (targa_header.image_type != 2 && targa_header.image_type != 10)
            Com.Error(Defines.ERR_DROP,
                    "LoadTGA: Only type 2 and 10 targa RGB images supported\n");

        if (targa_header.colormap_type != 0
                || (targa_header.pixel_size != 32 && targa_header.pixel_size != 24))
            Com
                    .Error(Defines.ERR_DROP,
                            "LoadTGA: Only 32 or 24 bit images supported (no colormaps)\n");

        int columns = targa_header.width;
        int rows = targa_header.height;

        if (dim != null) {
            dim.setWidth(columns);
            dim.setHeight(rows);
        }

        if (targa_header.id_length != 0)
            targa_header.data.position(targa_header.id_length);


        ByteBuffer buf_p = targa_header.data;

        byte green, blue, alphabyte;
        byte red = green = blue = alphabyte = 0;

        int numPixels = columns * rows;
        byte[] pic = new byte[numPixels * 4];
        int column;
        int row;
        int pixbuf;
        switch (targa_header.image_type) {
            case 2:
                for (row = rows - 1; row >= 0; row--) {

                    pixbuf = row * columns * 4;

                    for (column = 0; column < columns; column++) {
                        switch (targa_header.pixel_size) {
                            case 24 -> {
                                blue = buf_p.get();
                                green = buf_p.get();
                                red = buf_p.get();
                                pic[pixbuf++] = red;
                                pic[pixbuf++] = green;
                                pic[pixbuf++] = blue;
                                pic[pixbuf++] = (byte) 255;
                            }
                            case 32 -> {
                                blue = buf_p.get();
                                green = buf_p.get();
                                red = buf_p.get();
                                alphabyte = buf_p.get();
                                pic[pixbuf++] = red;
                                pic[pixbuf++] = green;
                                pic[pixbuf++] = blue;
                                pic[pixbuf++] = alphabyte;
                            }
                        }
                    }
                }
                break;
            case 10:

                for (row = rows - 1; row >= 0; row--) {

                    pixbuf = row * columns * 4;
                    try {

                        for (column = 0; column < columns; ) {

                            int packetHeader = buf_p.get() & 0xFF;
                            int packetSize = 1 + (packetHeader & 0x7f);

                            int j;
                            if ((packetHeader & 0x80) != 0) {
                                switch (targa_header.pixel_size) {
                                    case 24 -> {
                                        blue = buf_p.get();
                                        green = buf_p.get();
                                        red = buf_p.get();
                                        alphabyte = (byte) 255;
                                    }
                                    case 32 -> {
                                        blue = buf_p.get();
                                        green = buf_p.get();
                                        red = buf_p.get();
                                        alphabyte = buf_p.get();
                                    }
                                }

                                for (j = 0; j < packetSize; j++) {
                                    pic[pixbuf++] = red;
                                    pic[pixbuf++] = green;
                                    pic[pixbuf++] = blue;
                                    pic[pixbuf++] = alphabyte;
                                    column++;
                                    if (column == columns) {

                                        column = 0;
                                        if (row > 0)
                                            row--;
                                        else

                                            throw new longjmpException();

                                        pixbuf = row * columns * 4;
                                    }
                                }
                            } else {
                                for (j = 0; j < packetSize; j++) {
                                    switch (targa_header.pixel_size) {
                                        case 24 -> {
                                            blue = buf_p.get();
                                            green = buf_p.get();
                                            red = buf_p.get();
                                            pic[pixbuf++] = red;
                                            pic[pixbuf++] = green;
                                            pic[pixbuf++] = blue;
                                            pic[pixbuf++] = (byte) 255;
                                        }
                                        case 32 -> {
                                            blue = buf_p.get();
                                            green = buf_p.get();
                                            red = buf_p.get();
                                            alphabyte = buf_p.get();
                                            pic[pixbuf++] = red;
                                            pic[pixbuf++] = green;
                                            pic[pixbuf++] = blue;
                                            pic[pixbuf++] = alphabyte;
                                        }
                                    }
                                    column++;
                                    if (column == columns) {

                                        column = 0;
                                        if (row > 0)
                                            row--;
                                        else

                                            throw new longjmpException();

                                        pixbuf = row * columns * 4;
                                    }
                                }
                            }
                        }
                    } catch (longjmpException e) {

                    }
                }
                break;
        }
        return pic;
    }

    /*
     * ====================================================================
     * 
     * IMAGE FLOOD FILLING
     * 
     * ====================================================================
     */

    /*
     * ================= Mod_FloodFillSkin
     * 
     * Fill background pixels so mipmapping doesn't have haloes
     * =================
     */

    static class floodfill_t {
        short x;
        short y;
    }

    
    private static final int FLOODFILL_FIFO_SIZE = 0x1000;

    private static final int FLOODFILL_FIFO_MASK = FLOODFILL_FIFO_SIZE - 1;

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    private static final floodfill_t[] fifo = new floodfill_t[FLOODFILL_FIFO_SIZE];
    static {
        for (int j = 0; j < fifo.length; j++) {
            fifo[j] = new floodfill_t();
        }
    }

    
    
    private static void R_FloodFillSkin(byte[] skin, int skinwidth, int skinheight) {
        
        int fillcolor = skin[0] & 0xff;
        int filledcolor = -1;

        if (filledcolor == -1) {
            filledcolor = 0;
            
            for (int i = 0; i < 256; ++i)
                
                if (d_8to24table[i] == 0xFF000000) { 
                    
                    filledcolor = i;
                    break;
                }
        }

        
        
        if ((fillcolor == filledcolor) || (fillcolor == 255)) {
            return;
        }

        int inpt = 0;
        fifo[inpt].x = 0;
        fifo[inpt].y = 0;
        inpt = (inpt + 1) & FLOODFILL_FIFO_MASK;

        int outpt = 0;
        while (outpt != inpt) {
            int x = fifo[outpt].x;
            int y = fifo[outpt].y;
            int fdc = filledcolor;

            outpt = (outpt + 1) & FLOODFILL_FIFO_MASK;

            int off, dx, dy;

            int pos = x + skinwidth * y;
            if (x > 0) {
                
                off = -1;
                dx = -1;
                dy = 0;
                if (skin[pos + off] == (byte) fillcolor) {
                    skin[pos + off] = (byte) 255;
                    fifo[inpt].x = (short) (x + dx);
                    fifo[inpt].y = (short) (y + dy);
                    inpt = (inpt + 1) & FLOODFILL_FIFO_MASK;
                } else if (skin[pos + off] != (byte) 255)
                    fdc = skin[pos + off] & 0xff;
            }

            if (x < skinwidth - 1) {
                
                off = 1;
                dx = 1;
                dy = 0;
                if (skin[pos + off] == (byte) fillcolor) {
                    skin[pos + off] = (byte) 255;
                    fifo[inpt].x = (short) (x + dx);
                    fifo[inpt].y = (short) (y + dy);
                    inpt = (inpt + 1) & FLOODFILL_FIFO_MASK;
                } else if (skin[pos + off] != (byte) 255)
                    fdc = skin[pos + off] & 0xff;
            }

            if (y > 0) {
                
                off = -skinwidth;
                dx = 0;
                dy = -1;
                if (skin[pos + off] == (byte) fillcolor) {
                    skin[pos + off] = (byte) 255;
                    fifo[inpt].x = (short) (x + dx);
                    fifo[inpt].y = (short) (y + dy);
                    inpt = (inpt + 1) & FLOODFILL_FIFO_MASK;
                } else if (skin[pos + off] != (byte) 255)
                    fdc = skin[pos + off] & 0xff;
            }

            if (y < skinheight - 1) {
                
                off = skinwidth;
                dx = 0;
                dy = 1;
                if (skin[pos + off] == (byte) fillcolor) {
                    skin[pos + off] = (byte) 255;
                    fifo[inpt].x = (short) (x + dx);
                    fifo[inpt].y = (short) (y + dy);
                    inpt = (inpt + 1) & FLOODFILL_FIFO_MASK;
                } else if (skin[pos + off] != (byte) 255)
                    fdc = skin[pos + off] & 0xff;

            }

            skin[x + skinwidth * y] = (byte) fdc;
        }
    }

    

    /*
     * ================ GL_ResampleTexture ================
     */
    
    private static void GL_ResampleTexture(int[] in, int inwidth, int inheight, int[] out,
                                           int outwidth, int outheight) {
        
        
        
        
        
        

        
        throw new RuntimeException("Replace impl. w/ non AWT!");
        /**
         * NO AWT !
         * 
        BufferedImage image = new BufferedImage(inwidth, inheight,
                BufferedImage.TYPE_INT_ARGB);

        image.setRGB(0, 0, inwidth, inheight, in, 0, inwidth);

        AffineTransformOp op = new AffineTransformOp(AffineTransform
                .getScaleInstance(outwidth * 1.0 / inwidth, outheight * 1.0
                        / inheight), AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
        BufferedImage tmp = op.filter(image, null);

        tmp.getRGB(0, 0, outwidth, outheight, out, 0, outwidth);
        */

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    }

    /*
     * ================ GL_LightScaleTexture
     * 
     * Scale up the pixel values in a texture to increase the lighting range
     * ================
     */
    private void GL_LightScaleTexture(int[] in, int inwidth, int inheight,
                                      boolean only_gamma) {
        if (only_gamma) {

            int c = inwidth * inheight;
            for (int i = 0; i < c; i++) {
                int color = in[i];
                int r = (color >> 0) & 0xFF;

                r = gammatable[r] & 0xFF;
                int g = (color >> 8) & 0xFF;
                g = gammatable[g] & 0xFF;
                int b = (color >> 16) & 0xFF;
                b = gammatable[b] & 0xFF;

                in[i] = (r << 0) | (g << 8) | (b << 16) | (color & 0xFF000000);
            }
        } else {

            int c = inwidth * inheight;
            for (int i = 0; i < c; i++) {
                int color = in[i];
                int r = (color >> 0) & 0xFF;

                r = gammatable[intensitytable[r] & 0xFF] & 0xFF;
                int g = (color >> 8) & 0xFF;
                g = gammatable[intensitytable[g] & 0xFF] & 0xFF;
                int b = (color >> 16) & 0xFF;
                b = gammatable[intensitytable[b] & 0xFF] & 0xFF;

                in[i] = (r << 0) | (g << 8) | (b << 16) | (color & 0xFF000000);
            }

        }
    }

    /*
     * ================ GL_MipMap
     * 
     * Operates in place, quartering the size of the texture ================
     */
    private static void GL_MipMap(int[] in, int width, int height) {

        int inIndex = 0;
        int outIndex = 0;

        for (int i = 0; i < height; i += 2, inIndex += width) {
            for (int j = 0; j < width; j += 2, outIndex += 1, inIndex += 2) {

                int p1 = in[inIndex + 0];
                int p2 = in[inIndex + 1];
                int p3 = in[inIndex + width + 0];
                int p4 = in[inIndex + width + 1];

                int r = (((p1 >> 0) & 0xFF) + ((p2 >> 0) & 0xFF)
                        + ((p3 >> 0) & 0xFF) + ((p4 >> 0) & 0xFF)) >> 2;
                int g = (((p1 >> 8) & 0xFF) + ((p2 >> 8) & 0xFF)
                        + ((p3 >> 8) & 0xFF) + ((p4 >> 8) & 0xFF)) >> 2;
                int b = (((p1 >> 16) & 0xFF) + ((p2 >> 16) & 0xFF)
                        + ((p3 >> 16) & 0xFF) + ((p4 >> 16) & 0xFF)) >> 2;
                int a = (((p1 >> 24) & 0xFF) + ((p2 >> 24) & 0xFF)
                        + ((p3 >> 24) & 0xFF) + ((p4 >> 24) & 0xFF)) >> 2;

                in[outIndex] = (r << 0) | (g << 8) | (b << 16) | (a << 24);
            }
        }
    }

    /*
     * =============== GL_Upload32
     * 
     * Returns has_alpha ===============
     */
    private void GL_BuildPalettedTexture(ByteBuffer paletted_texture, int[] scaled,
                                         int scaled_width, int scaled_height) {

        int size = scaled_width * scaled_height;

        for (int i = 0; i < size; i++) {

            int r = (scaled[i] >> 3) & 31;
            int g = (scaled[i] >> 10) & 63;
            int b = (scaled[i] >> 19) & 31;

            int c = r | (g << 5) | (b << 11);

            paletted_texture.put(i, gl_state.d_16to8table[c]);
        }
    }

    private int upload_width;
    private int upload_height;

    private boolean uploaded_paletted;

    /*
     * =============== GL_Upload32
     * 
     * Returns has_alpha ===============
     */
    private final int[] scaled = new int[256 * 256];

    private final ByteBuffer paletted_texture = Lib.newByteBuffer(256*256);

    private final IntBuffer tex = Lib.newIntBuffer(512 * 256, ByteOrder.LITTLE_ENDIAN);

    private boolean GL_Upload32(int[] data, int width, int height, boolean mipmap) {

        Arrays.fill(scaled, 0);
        paletted_texture.clear();
        for (int j=0; j<256*256; j++) paletted_texture.put(j,(byte)0);
        
        uploaded_paletted = false;

        int scaled_width;
        for (scaled_width = 1; scaled_width < width; scaled_width <<= 1)
            ;
        if (gl_round_down.value > 0.0f && scaled_width > width && mipmap)
            scaled_width >>= 1;
        int scaled_height;
        for (scaled_height = 1; scaled_height < height; scaled_height <<= 1)
            ;
        if (gl_round_down.value > 0.0f && scaled_height > height && mipmap)
            scaled_height >>= 1;

        
        if (mipmap) {
            scaled_width >>= (int) gl_picmip.value;
            scaled_height >>= (int) gl_picmip.value;
        }

        
        if (scaled_width > 256)
            scaled_width = 256;
        if (scaled_height > 256)
            scaled_height = 256;

        if (scaled_width < 1)
            scaled_width = 1;
        if (scaled_height < 1)
            scaled_height = 1;

        upload_width = scaled_width;
        upload_height = scaled_height;

        if (scaled_width * scaled_height > 256 * 256)
            Com.Error(Defines.ERR_DROP, "GL_Upload32: too big");


        int c = width * height;
        int samples = gl_solid_format;

        for (int i = 0; i < c; i++) {
            if ((data[i] & 0xff000000) != 0xff000000) {
                samples = gl_alpha_format;
                break;
            }
        }

        int comp;
        switch (samples) {
            case gl_solid_format -> comp = gl_tex_solid_format;
            case gl_alpha_format -> comp = gl_tex_alpha_format;
            default -> {
                VID.Printf(Defines.PRINT_ALL,
                        "Unknown number of texture components " + samples + '\n');
                comp = samples;
            }
        }

        
        try {
            if (scaled_width == width && scaled_height == height) {
                if (!mipmap) {
                    if (qglColorTableEXT
                            && gl_ext_palettedtexture.value != 0.0f
                            && samples == gl_solid_format) {
                        uploaded_paletted = true;
                        GL_BuildPalettedTexture(paletted_texture, data,
                                scaled_width, scaled_height);
                        gl.glTexImage2D(GL_TEXTURE_2D, 0,
                                GL_COLOR_INDEX8_EXT, scaled_width,
                                scaled_height, 0, GL_COLOR_INDEX,
                                GL_UNSIGNED_BYTE, paletted_texture);
                    } else {
                        tex.clear();
                        tex.put(data, 0, scaled_width * scaled_height).flip();
                        gl.glTexImage2D(GL_TEXTURE_2D, 0, comp,
                                scaled_width, scaled_height, 0, GL_RGBA,
                                GL_UNSIGNED_BYTE, tex);
                    }
                    
                    throw new longjmpException();
                }
                
                System.arraycopy(data, 0, scaled, 0, width * height);

            } else
                GL_ResampleTexture(data, width, height, scaled, scaled_width,
                        scaled_height);

            GL_LightScaleTexture(scaled, scaled_width, scaled_height, !mipmap);

            if (qglColorTableEXT && gl_ext_palettedtexture.value != 0.0f
                    && (samples == gl_solid_format)) {
                uploaded_paletted = true;
                GL_BuildPalettedTexture(paletted_texture, scaled, scaled_width,
                        scaled_height);
                gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_COLOR_INDEX8_EXT,
                        scaled_width, scaled_height, 0, GL_COLOR_INDEX,
                        GL_UNSIGNED_BYTE, paletted_texture);
            } else {
                tex.clear();
                tex.put(scaled, 0, scaled_width * scaled_height).flip();
                gl.glTexImage2D(GL_TEXTURE_2D, 0, comp, scaled_width,
                        scaled_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
            }

            if (mipmap) {
                int miplevel = 0;
                while (scaled_width > 1 || scaled_height > 1) {
                    GL_MipMap(scaled, scaled_width, scaled_height);
                    scaled_width >>= 1;
                    scaled_height >>= 1;
                    if (scaled_width < 1)
                        scaled_width = 1;
                    if (scaled_height < 1)
                        scaled_height = 1;

                    miplevel++;
                    if (qglColorTableEXT
                            && gl_ext_palettedtexture.value != 0.0f
                            && samples == gl_solid_format) {
                        uploaded_paletted = true;
                        GL_BuildPalettedTexture(paletted_texture, scaled,
                                scaled_width, scaled_height);
                        gl.glTexImage2D(GL_TEXTURE_2D, miplevel,
                                GL_COLOR_INDEX8_EXT, scaled_width,
                                scaled_height, 0, GL_COLOR_INDEX,
                                GL_UNSIGNED_BYTE, paletted_texture);
                    } else {
                        tex.clear();
                        tex.put(scaled, 0, scaled_width * scaled_height).flip();
                        gl.glTexImage2D(GL_TEXTURE_2D, miplevel, comp,
                                scaled_width, scaled_height, 0, GL_RGBA,
                                GL_UNSIGNED_BYTE, tex);
                    }
                }
            }
            
        } catch (longjmpException e) {
            
        }

        if (mipmap) {
            gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    gl_filter_min);
        } else {
            gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    gl_filter_max);
        }
        gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                gl_filter_max);

        return (samples == gl_alpha_format);
    }

    /*
     * =============== GL_Upload8
     * 
     * Returns has_alpha ===============
     */

    private final int[] trans = new int[512 * 256];

    private boolean GL_Upload8(byte[] data, int width, int height, boolean mipmap,
                               boolean is_sky) {

        Arrays.fill(trans, 0);

        int s = width * height;

        if (s > trans.length)
            Com.Error(Defines.ERR_DROP, "GL_Upload8: too large");

        if (qglColorTableEXT && gl_ext_palettedtexture.value != 0.0f && is_sky) {
            gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_COLOR_INDEX8_EXT, width,
                    height, 0, GL_COLOR_INDEX, GL_UNSIGNED_BYTE, ByteBuffer.wrap(data));

            gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    gl_filter_max);
            gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    gl_filter_max);

            
            return false;
        } else {
            for (int i = 0; i < s; i++) {
                int p = data[i] & 0xff;
                trans[i] = d_8to24table[p];

                if (p == 255) { 
                    
                    
                    if (i > width && (data[i - width] & 0xff) != 255)
                        p = data[i - width] & 0xff;
                    else if (i < s - width && (data[i + width] & 0xff) != 255)
                        p = data[i + width] & 0xff;
                    else if (i > 0 && (data[i - 1] & 0xff) != 255)
                        p = data[i - 1] & 0xff;
                    else if (i < s - 1 && (data[i + 1] & 0xff) != 255)
                        p = data[i + 1] & 0xff;
                    else
                        p = 0;
                    

                    
                    
                    

                    trans[i] = d_8to24table[p] & 0x00FFFFFF; 
                }
            }

            return GL_Upload32(trans, width, height, mipmap);
        }
    }

    /*
     * ================ GL_LoadPic
     * 
     * This is also used as an entry point for the generated r_notexture
     * ================
     */
    image_t GL_LoadPic(String name, byte[] pic, int width, int height,
            int type, int bits) {
        image_t image;
        int i;

        
        for (i = 0; i < numgltextures; i++) {
            image = gltextures[i];
            if (image.texnum == 0)
                break;
        }

        if (i == numgltextures) {
            if (numgltextures == MAX_GLTEXTURES)
                Com.Error(Defines.ERR_DROP, "MAX_GLTEXTURES");

            numgltextures++;
        }
        image = gltextures[i];

        if (name.length() > Defines.MAX_QPATH)
            Com.Error(Defines.ERR_DROP, "Draw_LoadPic: \"" + name
                    + "\" is too long");

        image.name = name;
        image.registration_sequence = registration_sequence;

        image.width = width;
        image.height = height;
        image.type = type;

        if (type == it_skin && bits == 8)
            R_FloodFillSkin(pic, width, height);

        
        if (image.type == it_pic && bits == 8 && image.width < 64
                && image.height < 64) {
            pos_t pos = new pos_t(0, 0);

            int texnum = Scrap_AllocBlock(image.width, image.height, pos);

            if (texnum == -1) {
                

                image.scrap = false;

                image.texnum = TEXNUM_IMAGES + image.getId(); 
                                                              
                GL_Bind(image.texnum);

                if (8 == 8) {
                    image.has_alpha = GL_Upload8(pic, width, height,
                            (image.type != it_pic && image.type != it_sky),
                            image.type == it_sky);
                } else {
                    int[] tmp = new int[pic.length / 4];

                    for (i = 0; i < tmp.length; i++) {
                        tmp[i] = ((pic[4 * i + 0] & 0xFF) << 0); 
                                                                 
                        tmp[i] |= ((pic[4 * i + 1] & 0xFF) << 8); 
                                                                  
                        tmp[i] |= ((pic[4 * i + 2] & 0xFF) << 16); 
                                                                   
                        tmp[i] |= ((pic[4 * i + 3] & 0xFF) << 24); 
                                                                   
                    }

                    image.has_alpha = GL_Upload32(tmp, width, height,
                            (image.type != it_pic && image.type != it_sky));
                }

                image.upload_width = upload_width; 
                                                   
                image.upload_height = upload_height;
                image.paletted = uploaded_paletted;
                image.sl = 0;
                image.sh = 1;
                image.tl = 0;
                image.th = 1;

                return image;
            }

            scrap_dirty = true;


            int k = 0;
            for (i = 0; i < image.height; i++)
                for (int j = 0; j < image.width; j++, k++)
                    scrap_texels[texnum][(pos.y + i) * BLOCK_WIDTH + pos.x + j] = pic[k];

            image.texnum = TEXNUM_SCRAPS + texnum;
            image.scrap = true;
            image.has_alpha = true;
            image.sl = (pos.x + 0.01f) / BLOCK_WIDTH;
            image.sh = (pos.x + image.width - 0.01f) / BLOCK_WIDTH;
            image.tl = (pos.y + 0.01f) / BLOCK_WIDTH;
            image.th = (pos.y + image.height - 0.01f) / BLOCK_WIDTH;

        } else {
            

            image.scrap = false;

            image.texnum = TEXNUM_IMAGES + image.getId(); 
            GL_Bind(image.texnum);

            if (bits == 8) {
                image.has_alpha = GL_Upload8(pic, width, height,
                        (image.type != it_pic && image.type != it_sky),
                        image.type == it_sky);
            } else {
                int[] tmp = new int[pic.length / 4];

                for (i = 0; i < tmp.length; i++) {
                    tmp[i] = ((pic[4 * i + 0] & 0xFF) << 0); 
                    tmp[i] |= ((pic[4 * i + 1] & 0xFF) << 8); 
                    tmp[i] |= ((pic[4 * i + 2] & 0xFF) << 16); 
                    tmp[i] |= ((pic[4 * i + 3] & 0xFF) << 24); 
                }

                image.has_alpha = GL_Upload32(tmp, width, height,
                        (image.type != it_pic && image.type != it_sky));
            }
            image.upload_width = upload_width; 
            image.upload_height = upload_height;
            image.paletted = uploaded_paletted;
            image.sl = 0;
            image.sh = 1;
            image.tl = 0;
            image.th = 1;
        }
        return image;
    }

    /*
     * ================ GL_LoadWal ================
     */
    private image_t GL_LoadWal(String name) {

        byte[] raw = FS.LoadFile(name);
        if (raw == null) {
            VID.Printf(Defines.PRINT_ALL, "GL_FindImage: can't load " + name
                    + '\n');
            return r_notexture;
        }

        qfiles.miptex_t mt = new qfiles.miptex_t(raw);

        byte[] pix = new byte[mt.width * mt.height];
        System.arraycopy(raw, mt.offsets[0], pix, 0, pix.length);

        return GL_LoadPic(name, pix, mt.width, mt.height, it_wall, 8);
    }

    /*
     * =============== GL_FindImage
     * 
     * Finds or loads the given image ===============
     */
    image_t GL_FindImage(String name, int type) {


        name = name.toLowerCase();
        
        int index = name.indexOf('\0');
        if (index != -1)
            name = name.substring(0, index);

        if (name == null || name.length() < 5)
            return null;


        image_t image;
        for (int i = 0; i < numgltextures; i++) {
            image = gltextures[i];
            if (name.equals(image.name)) {
                image.registration_sequence = registration_sequence;
                return image;
            }
        }

        
        
        
        byte[] pic;
        Dimension dim = new Dimension();

        if (name.endsWith(".pcx")) {

            pic = LoadPCX(name, null, dim);
            if (pic == null)
                return null;
            image = GL_LoadPic(name, pic, dim.getWidth(), dim.getHeight(), type, 8);

        } else if (name.endsWith(".wal")) {

            image = GL_LoadWal(name);

        } else if (name.endsWith(".tga")) {

            pic = LoadTGA(name, dim);

            if (pic == null)
                return null;

            image = GL_LoadPic(name, pic, dim.getWidth(), dim.getHeight(), type, 32);

        } else
            return null;

        return image;
    }

    /*
     * =============== R_RegisterSkin ===============
     */
    @Override
    public image_t R_RegisterSkin(String name) {
        return GL_FindImage(name, it_skin);
    }

    private final IntBuffer texnum = Lib.newIntBuffer(1);

    /*
     * ================ GL_FreeUnusedImages
     * 
     * Any image that was not touched on this registration sequence will be
     * freed. ================
     */
    void GL_FreeUnusedImages() {

        
        r_notexture.registration_sequence = registration_sequence;
        r_particletexture.registration_sequence = registration_sequence;

        image_t image;

        for (int i = 0; i < numgltextures; i++) {
            image = gltextures[i];
            
            if (image.registration_sequence == registration_sequence)
                continue;
            
            if (image.registration_sequence == 0)
                continue;
            
            if (image.type == it_pic)
                continue;

            
            texnum.put(0, image.texnum);
            gl.glDeleteTextures(texnum);
            image.clear();
        }
    }

    /*
     * =============== Draw_GetPalette ===============
     */
    @Override
    protected void Draw_GetPalette() {
        byte[][] palette = new byte[1][];

        

        LoadPCX("pics/colormap.pcx", palette, new Dimension());

        if (palette[0] == null || palette[0].length != 768)
            Com.Error(Defines.ERR_FATAL, "Couldn't load pics/colormap.pcx");

        byte[] pal = palette[0];

        int j = 0;
        for (int i = 0; i < 256; i++) {
            int r = pal[j++] & 0xFF;
            int g = pal[j++] & 0xFF;
            int b = pal[j++] & 0xFF;

            d_8to24table[i] = (255 << 24) | (b << 16) | (g << 8) | (r << 0);
        }

        d_8to24table[255] &= 0x00FFFFFF; 

        particle_t.setColorPalette(d_8to24table);
    }

    /*
     * =============== GL_InitImages ===============
     */
    @Override
    void GL_InitImages() {
        float g = vid_gamma.value;

        registration_sequence = 1;


        cvar_t intensity = Cvar.Get("intensity", "2", 0);

        if (intensity.value <= 1)
            Cvar.Set("intensity", "1");

        gl_state.inverse_intensity = 1 / intensity.value;

        Draw_GetPalette();

        if (qglColorTableEXT) {
            gl_state.d_16to8table = FS.LoadFile("pics/16to8.dat");
            if (gl_state.d_16to8table == null)
                Com.Error(Defines.ERR_FATAL, "Couldn't load pics/16to8.pcx");
        }

        if ((gl_config.renderer & (GL_RENDERER_VOODOO | GL_RENDERER_VOODOO2)) != 0) {
            g = 1.0F;
        }

        int i;
        for (i = 0; i < 256; i++) {

            if (g == 1.0f) {
                gammatable[i] = (byte) i;
            } else {

                int inf = (int) (255.0f * Math.pow((i + 0.5) / 255.5, g) + 0.5);
                if (inf < 0)
                    inf = 0;
                if (inf > 255)
                    inf = 255;
                gammatable[i] = (byte) inf;
            }
        }

        for (i = 0; i < 256; i++) {
            int j = (int) (i * intensity.value);
            if (j > 255)
                j = 255;
            intensitytable[i] = (byte) j;
        }
    }

    /*
     * =============== GL_ShutdownImages ===============
     */
    @Override
    void GL_ShutdownImages() {

        for (int i = 0; i < numgltextures; i++) {
            image_t image = gltextures[i];

            if (image.registration_sequence == 0)
                continue; 
            
            texnum.put(0, image.texnum);
            gl.glDeleteTextures(texnum);
            image.clear();
        }
    }

}