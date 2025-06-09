/* RdesktopCanvas.java
 * Component: ProperJavaRDP
 *
 * Revision: $Revision: #2 $
 * Author: $Author: tvkelley $
 * Date: $Date: 2009/09/15 $
 *
 * Copyright (c) 2005 Propero Limited
 *
 * Purpose: Canvas component, handles drawing requests from server,
 *          and passes user input to Input class.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 * USA
 *
 * (See gpl.txt for details of the GNU General Public License.)
 *
 */
package net.propero.rdp;

import net.propero.rdp.keymapping.KeyCode;
import net.propero.rdp.keymapping.KeyCode_FileBased;
import net.propero.rdp.orders.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.awt.image.MemoryImageSource;
import java.util.Arrays;


public abstract class RdesktopCanvas extends Canvas {
    private static final int ROP2_COPY = 0xc;
    static final Logger logger = LoggerFactory.getLogger(RdesktopCanvas.class);
    private static final int ROP2_XOR = 0x6;

    
    private static final int ROP2_AND = 0x8;

    
    private static final int ROP2_NXOR = 0x9;
    private static final int ROP2_OR = 0xe;
    private static final int MIX_TRANSPARENT = 0;
    private static final int MIX_OPAQUE = 1;
    private static final int TEXT2_VERTICAL = 0x04;
    private static final int TEXT2_IMPLICIT_X = 0x20;
    private final int width;
    private final int height;

    public KeyCode keys;
    private KeyCode_FileBased fbKeys;
    public String sKeys;
    private Rdp rdp;
    public WrappedImage backstore;

    
    private Cursor previous_cursor; 
    private Input input;
    private Cache cache;
    
    
    private int top;

    private int left;

    private int right;

    private int bottom;

    /**
     * Initialise this canvas to specified width and height, also initialise
     * backstore
     *
     * @param width  Desired width of canvas
     * @param height Desired height of canvas
     */
    RdesktopCanvas(int width, int height) {
        
        this.width = width;
        this.height = height;
        this.right = width - 1; 
        this.bottom = height - 1; 
        setSize(width, height);

        backstore =  new WrappedImage(width, height, BufferedImage.TYPE_INT_RGB);
        setIgnoreRepaint(true);

        
    }

    /**
     * Parse a delta co-ordinate in polyline order form
     *
     * @param buffer
     * @param offset
     * @return
     */
    private static int parse_delta(byte[] buffer, int[] offset) {
        int value = buffer[offset[0]++] & 0xff;
        int two_byte = value & 0x80;

        if ((value & 0x40) != 0) /* sign bit */
            value |= ~0x3f;
        else
            value &= 0x3f;

        if (two_byte != 0)
            value = (value << 8) | (buffer[offset[0]++] & 0xff);

        return value;
    }

    @Override
    public void paint(Graphics g) {
        update(g);
    }

    @Override
    public abstract void update(Graphics g);

    /**
     * Register a colour palette with this canvas
     *
     * @param cm Colour model to be used with this canvas
     */
    public void registerPalette(IndexColorModel cm) {

        backstore.setIndexColorModel(cm);
    }

    /**
     * Register the Rdp layer to act as the communications interface to this
     * canvas
     *
     * @param rdp Rdp object controlling Rdp layer communication
     */
    public void registerCommLayer(Rdp rdp) {
        this.rdp = rdp;
        if (fbKeys != null)
            input = new Input_Localised(this, rdp, fbKeys);

    }

    /**
     * Register keymap
     *
     * @param keys Keymapping object for use in handling keyboard events
     */
    public void registerKeyboard(KeyCode_FileBased keys) {
        this.fbKeys = keys;
        if (rdp != null) {
            
            input = new Input_Localised(this, rdp, keys);
        }
    }

    /**
     * Set cache for this session
     *
     * @param cache Cache to be used in this session
     */
    public void registerCache(Cache cache) {
        this.cache = cache;
    }

    /**
     * Display a compressed bitmap direct to the backstore NOTE: Currently not
     * functioning correctly, see Bitmap.decompressImgDirect Does not call
     * repaint. Image is drawn to canvas on next update
     *
     * @param x      x coordinate within backstore for drawing of bitmap
     * @param y      y coordinate within backstore for drawing of bitmap
     * @param width  Width of bitmap
     * @param height Height of bitmap
     * @param size   Size (bytes) of compressed bitmap data
     * @param data   Packet containing compressed bitmap data at current read
     *               position
     * @param Bpp    Bytes-per-pixel for bitmap
     * @param cm     Colour model currently in use, if any
     * @throws RdesktopException
     */
    public void displayCompressed(int x, int y, int width, int height,
                                  int size, RdpPacket_Localised data, int Bpp, IndexColorModel cm)
            throws RdesktopException {
        backstore = Bitmap.decompressImgDirect(width, height, size, data, Bpp,
                cm, x, y, backstore);
    }

    /**
     * Draw an image object to the backstore, does not call repaint. Image is
     * drawn to canvas on next update.
     *
     * @param img Image to draw to backstore
     * @param x   x coordinate for drawing location
     * @param y   y coordinate for drawing location
     * @throws RdesktopException
     */
    public void displayImage(Image img, int x, int y) {

        Graphics g = backstore.getGraphics();
        g.drawImage(img, x, y, null);
        /* ********* Useful test for identifying image boundaries ************ */
        
        


    }

    /**
     * Draw an image (from an integer array of colour data) to the backstore,
     * does not call repaint. Image is drawn to canvas on next update.
     *
     * @param data Integer array of pixel colour information
     * @param w    Width of image
     * @param h    Height of image
     * @param x    x coordinate for drawing location
     * @param y    y coordinate for drawing location
     * @param cx   Width of drawn image (clips, does not scale)
     * @param cy   Height of drawn image (clips, does not scale)
     * @throws RdesktopException
     */
    public void displayImage(int[] data, int w, int h, int x, int y, int cx,
                             int cy) {

        backstore.setRGB(x, y, cx, cy, data, 0, w);

        /* ********* Useful test for identifying image boundaries ************ */
        
        
        
        
        
    }

    /**
     * Retrieve an image from the backstore, as integer pixel information
     *
     * @param x  x coordinate of image to retrieve
     * @param y  y coordinage of image to retrieve
     * @param cx width of image to retrieve
     * @param cy height of image to retrieve
     * @return Requested area of backstore, as an array of integer pixel colours
     */
    public int[] getImage(int x, int y, int cx, int cy) {

        int[] data = backstore.getRGB(x, y, cx, cy, null,

                0,
                cx);

        return data;
    }

    /**
     * Draw an image (from an integer array of colour data) to the backstore,
     * also calls repaint to draw image to canvas
     *
     * @param x    x coordinate at which to draw image
     * @param y    y coordinate at which to draw image
     * @param cx   Width of drawn image (clips, does not scale)
     * @param cy   Height of drawn image (clips, does not scale)
     * @param data Image to draw, represented as an array of integer pixel
     *             colours
     */
    public void putImage(int x, int y, int cx, int cy, int[] data) {

        backstore.setRGBNoConversion(x, y, cx, cy, data, 0, 
                
                
                cx);

        this.repaint(x, y, cx, cy);
    }

    /**
     * Reset clipping boundaries for canvas
     */
    public void resetClip() {
        Graphics g = this.getGraphics();
        Rectangle bounds = this.getBounds();
        g.setClip(bounds.x, bounds.y, bounds.width, bounds.height);
        this.top = 0;
        this.left = 0;
        this.right = this.width - 1; 
        this.bottom = this.height - 1; 
    }

    /**
     * Set clipping boundaries for canvas, based on a bounds order
     *
     * @param bounds Order defining new boundaries
     */
    public void setClip(BoundsOrder bounds) {
        Graphics g = this.getGraphics();
        g.setClip(bounds.getLeft(), bounds.getTop(), bounds.getRight()
                - bounds.getLeft(), bounds.getBottom() - bounds.getTop());
        this.top = bounds.getTop();
        this.left = bounds.getLeft();
        this.right = bounds.getRight();
        this.bottom = bounds.getBottom();
    }

    /**
     * Move the mouse pointer (only available in Java 1.3+)
     *
     * @param x x coordinate for mouse move
     * @param y y coordinate for mouse move
     */
    public void movePointer(int x, int y) {
        
    }

    /**
     * Draw a filled rectangle to the screen
     *
     * @param x     x coordinate (left) of rectangle
     * @param y     y coordinate (top) of rectangle
     * @param cx    Width of rectangle
     * @param cy    Height of rectangle
     * @param color Colour of rectangle
     */
    private void fillRectangle(int x, int y, int cx, int cy, int color) {
        
        if (x > this.right || y > this.bottom)
            return; 

        int Bpp = Options.Bpp;

        
        color = Bitmap.convertTo24(color);

        
        if (Bpp == 3)
            color = ((color & 0xFF) << 16) | (color & 0xFF00)
                    | ((color & 0xFF0000) >> 16);

        
        int clipright = x + cx - 1;
        if (clipright > this.right)
            clipright = this.right;
        if (x < this.left)
            x = this.left;
        cx = clipright - x + 1;

        
        int clipbottom = y + cy - 1;
        if (clipbottom > this.bottom)
            clipbottom = this.bottom;
        if (y < this.top)
            y = this.top;
        cy = clipbottom - y + 1;

        
        int[] rect = new int[cx * cy];
        Arrays.fill(rect, color);
        
        backstore.setRGB(x, y, cx, cy, rect, 0, cx);

        
        
        this.repaint(x, y, cx, cy); 
        
    }

    /**
     * Draw a line to the screen
     *
     * @param x1     x coordinate of start point of line
     * @param y1     y coordinate of start point of line
     * @param x2     x coordinate of end point of line
     * @param y2     y coordinate of end point of line
     * @param color  colour of line
     * @param opcode Operation code defining operation to perform on pixels within
     *               the line
     */
    private void drawLine(int x1, int y1, int x2, int y2, int color, int opcode) {
        
        color = Bitmap.convertTo24(color);

        if (x1 == x2 || y1 == y2) {
            drawLineVerticalHorizontal(x1, y1, x2, y2, color, opcode);
            return;
        }

        int xinc1, xinc2;

        if (x2 >= x1) { 
            xinc1 = 1;
            xinc2 = 1;
        } else { 
            xinc1 = -1;
            xinc2 = -1;
        }

        int yinc2;
        int yinc1;
        if (y2 >= y1) {
            yinc1 = 1;
            yinc2 = 1;
        } else { 
            yinc1 = -1;
            yinc2 = -1;
        }

        int numpixels;
        int numadd;
        int den;
        int num;
        int deltay = Math.abs(y2 - y1);
        int deltax = Math.abs(x2 - x1);
        if (deltax >= deltay) {
            
            xinc1 = 0; 
            yinc2 = 0; 
            den = deltax;
            num = deltax / 2;
            numadd = deltay;
            numpixels = deltax; 
        } else { 
            xinc2 = 0; 
            yinc1 = 0; 
            den = deltay;
            num = deltay / 2;
            numadd = deltax;
            numpixels = deltay; 
        }

        int y = y1;
        int x = x1;
        for (int curpixel = 0; curpixel <= numpixels; curpixel++) {
            setPixel(opcode, x, y, color); 
            num += numadd; 
            if (num >= den) { 
                num -= den; 
                x += xinc1; 
                y += yinc1; 
            }
            x += xinc2; 
            y += yinc2; 
        }

        int x_min = Math.min(x1, x2);
        int x_max = Math.max(x1, x2);
        int y_min = Math.min(y1, y2);
        int y_max = Math.max(y1, y2);

        this.repaint(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1);
    }

    /**
     * Helper function for drawLine, draws a horizontal or vertical line using a
     * much faster method than used for diagonal lines
     *
     * @param x1     x coordinate of start point of line
     * @param y1     y coordinate of start point of line
     * @param x2     x coordinate of end point of line
     * @param y2     y coordinate of end point of line
     * @param color  colour of line
     * @param opcode Operation code defining operation to perform on pixels within
     *               the line
     */
    private void drawLineVerticalHorizontal(int x1, int y1, int x2, int y2,
                                            int color, int opcode) {
        int pbackstore;
        int i;
        
        if (y1 == y2) { 
            if (y1 >= this.top && y1 <= this.bottom) { 
                if (x2 > x1) { 
                    if (x1 < this.left)
                        x1 = this.left;
                    if (x2 > this.right)
                        x2 = this.right;
                    pbackstore = y1 * this.width + x1;
                    for (i = 0; i < x2 - x1; i++) {
                        RasterOp.do_pixel(opcode, backstore, x1 + i, y1, color);
                        pbackstore++;
                    }
                    repaint(x1, y1, x2 - x1 + 1, 1);
                } else { 
                    if (x2 < this.left)
                        x2 = this.left;
                    if (x1 > this.right)
                        x1 = this.right;
                    pbackstore = y1 * this.width + x1;
                    for (i = 0; i < x1 - x2; i++) {
                        RasterOp.do_pixel(opcode, backstore, x2 + i, y1, color);
                        pbackstore--;
                    }
                    repaint(x2, y1, x1 - x2 + 1, 1);
                }
            }
        } else { 
            if (x1 >= this.left && x1 <= this.right) { 
                if (y2 > y1) { 
                    if (y1 < this.top)
                        y1 = this.top;
                    if (y2 > this.bottom)
                        y2 = this.bottom;
                    pbackstore = y1 * this.width + x1;
                    for (i = 0; i < y2 - y1; i++) {
                        RasterOp.do_pixel(opcode, backstore, x1, y1 + i, color);
                        pbackstore += this.width;
                    }
                    repaint(x1, y1, 1, y2 - y1 + 1);
                } else { 
                    if (y2 < this.top)
                        y2 = this.top;
                    if (y1 > this.bottom)
                        y1 = this.bottom;
                    pbackstore = y1 * this.width + x1;
                    for (i = 0; i < y1 - y2; i++) {
                        RasterOp.do_pixel(opcode, backstore, x1, y2 + i, color);
                        pbackstore -= this.width;
                    }
                    repaint(x1, y2, 1, y1 - y2 + 1);
                }
            }
        }
        
        
    }

    /**
     * Draw a line to the screen
     *
     * @param line LineOrder describing line to be drawn
     */
    public void drawLineOrder(LineOrder line) {
        int x1 = line.getStartX();
        int y1 = line.getStartY();
        int x2 = line.getEndX();
        int y2 = line.getEndY();

        int fgcolor = line.getPen().getColor();

        int opcode = line.getOpcode() - 1;
        drawLine(x1, y1, x2, y2, fgcolor, opcode);
    }

    /**
     * Perform a dest blt
     *
     * @param destblt DestBltOrder describing the blit to be performed
     */
    public void drawDestBltOrder(DestBltOrder destblt) {
        int x = destblt.getX();
        int y = destblt.getY();

        if (x > this.right || y > this.bottom)
            return; 

        int cx = destblt.getCX();
        int cy = destblt.getCY();

        int clipright = x + cx - 1;
        if (clipright > this.right)
            clipright = this.right;
        if (x < this.left)
            x = this.left;
        cx = clipright - x + 1;

        int clipbottom = y + cy - 1;
        if (clipbottom > this.bottom)
            clipbottom = this.bottom;
        if (y < this.top)
            y = this.top;
        cy = clipbottom - y + 1;

        RasterOp.do_array(destblt.getOpcode(), backstore, this.width, x, y, cx, cy,
                null, 0, 0, 0);
        this.repaint(x, y, cx, cy);

    }

    /**
     * Perform a screen blit
     *
     * @param screenblt ScreenBltOrder describing the blit to be performed
     */
    public void drawScreenBltOrder(ScreenBltOrder screenblt) {
        int x = screenblt.getX();
        int y = screenblt.getY();

        if (x > this.right || y > this.bottom)
            return; 

        int cx = screenblt.getCX();
        int cy = screenblt.getCY();
        int srcx = screenblt.getSrcX();
        int srcy = screenblt.getSrcY();

        int clipright = x + cx - 1;
        if (clipright > this.right)
            clipright = this.right;
        if (x < this.left)
            x = this.left;
        cx = clipright - x + 1;

        int clipbottom = y + cy - 1;
        if (clipbottom > this.bottom)
            clipbottom = this.bottom;
        if (y < this.top)
            y = this.top;
        cy = clipbottom - y + 1;

        srcx += x - screenblt.getX();
        srcy += y - screenblt.getY();

        RasterOp.do_array(screenblt.getOpcode(), backstore, this.width, x, y, cx,
                cy, null, this.width, srcx, srcy);
        this.repaint(x, y, cx, cy);

    }

    /**
     * Perform a memory blit
     *
     * @param memblt MemBltOrder describing the blit to be performed
     */
    public void drawMemBltOrder(MemBltOrder memblt) {
        int x = memblt.getX();
        int y = memblt.getY();

        if (x > this.right || y > this.bottom)
            return; 

        int cx = memblt.getCX();
        int cy = memblt.getCY();
        int srcx = memblt.getSrcX();
        int srcy = memblt.getSrcY();

        
        int clipright = x + cx - 1;
        if (clipright > this.right)
            clipright = this.right;
        if (x < this.left)
            x = this.left;
        cx = clipright - x + 1;

        
        int clipbottom = y + cy - 1;
        if (clipbottom > this.bottom)
            clipbottom = this.bottom;
        if (y < this.top)
            y = this.top;
        cy = clipbottom - y + 1;

        srcx += x - memblt.getX();
        srcy += y - memblt.getY();

        if (logger.isInfoEnabled())
            logger.info("MEMBLT x={} y={} cx={} cy={} srcx={} srcy={} opcode={}", x, y, cx, cy, srcx, srcy, memblt.getOpcode());
        try {
            Bitmap bitmap = cache.getBitmap(memblt.getCacheID(), memblt
                    .getCacheIDX());
            
            
            RasterOp.do_array(memblt.getOpcode(), backstore, this.width, x, y, cx,
                    cy, bitmap.getBitmapData(), bitmap.getWidth(), srcx, srcy);

            this.repaint(x, y, cx, cy);
        } catch (RdesktopException e) {
        }
    }

    /**
     * Draw a pattern to the screen (pattern blit)
     *
     * @param opcode  Code defining operation to be performed
     * @param x       x coordinate for left of blit area
     * @param y       y coordinate for top of blit area
     * @param cx      Width of blit area
     * @param cy      Height of blit area
     * @param fgcolor Foreground colour for pattern
     * @param bgcolor Background colour for pattern
     * @param brush   Brush object defining pattern to be drawn
     */
    private void patBltOrder(int opcode, int x, int y, int cx, int cy,
                             int fgcolor, int bgcolor, Brush brush) {

        
        fgcolor = Bitmap.convertTo24(fgcolor);
        bgcolor = Bitmap.convertTo24(bgcolor);

        
        int clipright = x + cx - 1;
        if (clipright > this.right)
            clipright = this.right;
        if (x < this.left)
            x = this.left;
        cx = clipright - x + 1;

        
        int clipbottom = y + cy - 1;
        if (clipbottom > this.bottom)
            clipbottom = this.bottom;
        if (y < this.top)
            y = this.top;
        cy = clipbottom - y + 1;

        int i;
        int[] src = null;
        /*
         *
         * byte[8]; for(i=0;i<ipattern.length;i++) {
         * ipattern[ipattern.length-1-i] = pattern[i]; }
         */
        switch (brush.getStyle()) {
            case 0 -> {
                src = new int[cx * cy];
                for (i = 0; i < src.length; i++)
                    src[i] = fgcolor;
                RasterOp.do_array(opcode, backstore, this.width, x, y, cx, cy, src, cx,
                        0, 0);
                this.repaint(x, y, cx, cy);
            }
            case 2 -> System.out.println("hatch");
            case 3 -> {
                int brushx = brush.getXOrigin();
                int brushy = brush.getYOrigin();
                byte[] pattern = brush.getPattern();
                byte[] ipattern = pattern;
                src = new int[cx * cy];
                int psrc = 0;
                for (i = 0; i < cy; i++) {
                    for (int j = 0; j < cx; j++) {
                        src[psrc] = (ipattern[(i + brushy) % 8] & (0x01 << ((j + brushx) % 8))) == 0 ? fgcolor : bgcolor;
                        psrc++;
                    }
                }
                RasterOp.do_array(opcode, backstore, this.width, x, y, cx, cy, src, cx,
                        0, 0);
                this.repaint(x, y, cx, cy);
            }
            default -> logger.warn("Unsupported brush style {}", brush.getStyle());
        }
    }

    /**
     * Perform a pattern blit on the screen
     *
     * @param patblt PatBltOrder describing the blit to be performed
     */
    public void drawPatBltOrder(PatBltOrder patblt) {
        Brush brush = patblt.getBrush();
        int x = patblt.getX();
        int y = patblt.getY();

        if (x > this.right || y > this.bottom)
            return; 

        int cx = patblt.getCX();
        int cy = patblt.getCY();
        int fgcolor = patblt.getForegroundColor();
        int bgcolor = patblt.getBackgroundColor();
        int opcode = patblt.getOpcode();

        patBltOrder(opcode, x, y, cx, cy, fgcolor, bgcolor, brush);
    }

    /**
     * Perform a tri blit on the screen
     *
     * @param triblt TriBltOrder describing the blit
     */
    public void drawTriBltOrder(TriBltOrder triblt) {
        int x = triblt.getX();
        int y = triblt.getY();

        if (x > this.right || y > this.bottom)
            return; 

        int cx = triblt.getCX();
        int cy = triblt.getCY();
        int srcx = triblt.getSrcX();
        int srcy = triblt.getSrcY();
        int fgcolor = triblt.getForegroundColor();
        int bgcolor = triblt.getBackgroundColor();
        Brush brush = triblt.getBrush();

        
        fgcolor = Bitmap.convertTo24(fgcolor);
        bgcolor = Bitmap.convertTo24(bgcolor);

        
        int clipright = x + cx - 1;
        if (clipright > this.right)
            clipright = this.right;
        if (x < this.left)
            x = this.left;
        cx = clipright - x + 1;

        
        int clipbottom = y + cy - 1;
        if (clipbottom > this.bottom)
            clipbottom = this.bottom;
        if (y < this.top)
            y = this.top;
        cy = clipbottom - y + 1;

        try {
            Bitmap bitmap = cache.getBitmap(triblt.getCacheID(), triblt
                    .getCacheIDX());
            switch (triblt.getOpcode()) {
                case 0x69 -> {
                    RasterOp.do_array(ROP2_XOR, backstore, this.width, x, y, cx, cy,
                            bitmap.getBitmapData(), bitmap.getWidth(), srcx, srcy);
                    patBltOrder(ROP2_NXOR, x, y, cx, cy, fgcolor, bgcolor, brush);
                }
                case 0xb8 -> {
                    patBltOrder(ROP2_XOR, x, y, cx, cy, fgcolor, bgcolor, brush);
                    RasterOp.do_array(ROP2_AND, backstore, this.width, x, y, cx, cy,
                            bitmap.getBitmapData(), bitmap.getWidth(), srcx, srcy);
                    patBltOrder(ROP2_XOR, x, y, cx, cy, fgcolor, bgcolor, brush);
                }
                case 0xc0 -> {
                    RasterOp.do_array(ROP2_COPY, backstore, this.width, x, y, cx, cy,
                            bitmap.getBitmapData(), bitmap.getWidth(), srcx, srcy);
                    patBltOrder(ROP2_AND, x, y, cx, cy, fgcolor, bgcolor, brush);
                }
                default -> {
                    logger
                            .warn("Unimplemented Triblt opcode:{}", triblt.getOpcode());
                    RasterOp.do_array(ROP2_COPY, backstore, this.width, x, y, cx, cy,
                            bitmap.getBitmapData(), bitmap.getWidth(), srcx, srcy);
                }
            }
        } catch (RdesktopException e) {
        }
    }

    /**
     * Draw a multi-point set of lines to the screen
     *
     * @param polyline PolyLineOrder describing the set of lines to draw
     */
    public void drawPolyLineOrder(PolyLineOrder polyline) {
        int x = polyline.getX();
        int y = polyline.getY();
        int fgcolor = polyline.getForegroundColor();
        int datasize = polyline.getDataSize();
        byte[] databytes = polyline.getData();
        int lines = polyline.getLines();

        
        fgcolor = Bitmap.convertTo24(fgcolor);

        
        
        

        int[] data = new int[1];
        data[0] = ((lines - 1) / 4) + 1;
        int flags = 0;
        int index = 0;

        int opcode = polyline.getOpcode() - 1;

        for (int line = 0; (line < lines) && (data[0] < datasize); line++) {
            int xfrom = x;
            int yfrom = y;

            if (line % 4 == 0)
                flags = databytes[index++];

            if ((flags & 0xc0) == 0)
                flags |= 0xc0; /* none = both */

            if ((flags & 0x40) != 0)
                x += parse_delta(databytes, data);

            if ((flags & 0x80) != 0)
                y += parse_delta(databytes, data);
            
            

            drawLine(xfrom, yfrom, x, y, fgcolor, opcode);
            flags <<= 2;
        }
    }

    /**
     * Draw a rectangle to the screen
     *
     * @param rect RectangleOrder defining the rectangle to be drawn
     */
    public void drawRectangleOrder(RectangleOrder rect) {
        
        fillRectangle(rect.getX(), rect.getY(), rect.getCX(), rect.getCY(),
                rect.getColor());
    }

    /**
     * Perform an operation on a pixel in the backstore
     *
     * @param opcode ID of operation to perform
     * @param x      x coordinate of pixel
     * @param y      y coordinate of pixel
     * @param color  Colour value to be used in operation
     */
    private void setPixel(int opcode, int x, int y, int color) {
        int Bpp = Options.Bpp;

        
        if (Bpp == 3)
            color = ((color & 0xFF) << 16) | (color & 0xFF00)
                    | ((color & 0xFF0000) >> 16);

        if ((x < this.left) || (x > this.right) || (y < this.top)
                || (y > this.bottom)) { 
        } else {
            RasterOp.do_pixel(opcode, backstore, x, y, color);
        }
    }

    /**
     * Draw a single glyph to the screen
     *
     * @param mixmode 0 for transparent background, specified colour for background
     *                otherwide
     * @param x       x coordinate on screen at which to draw glyph
     * @param y       y coordinate on screen at which to draw glyph
     * @param cx      Width of clipping area for glyph
     * @param cy      Height of clipping area for glyph
     * @param data    Set of values defining glyph's pattern
     * @param bgcolor Background colour for glyph pattern
     * @param fgcolor Foreground colour for glyph pattern
     */
    public void drawGlyph(int mixmode, int x, int y, int cx, int cy,
                          byte[] data, int bgcolor, int fgcolor) {

        int Bpp = Options.Bpp;

        
        fgcolor = Bitmap.convertTo24(fgcolor);
        bgcolor = Bitmap.convertTo24(bgcolor);

        
        if (Bpp == 3) {
            fgcolor = ((fgcolor & 0xFF) << 16) | (fgcolor & 0xFF00)
                    | ((fgcolor & 0xFF0000) >> 16);
            bgcolor = ((bgcolor & 0xFF) << 16) | (bgcolor & 0xFF00)
                    | ((bgcolor & 0xFF0000) >> 16);
        }

        

        if (x > this.right || y > this.bottom)
            return; 

        int clipright = x + cx - 1;
        if (clipright > this.right)
            clipright = this.right;
        int newx = Math.max(x, this.left);
		int newcx = clipright - x + 1;

        int clipbottom = y + cy - 1;
        if (clipbottom > this.bottom)
            clipbottom = this.bottom;
        int top = this.top;


        int newy = y;

        int newcy = clipbottom - newy + 1;

        int pbackstore = (newy * this.width) + x;
        int bytes_per_row = (cx - 1) / 8 + 1;
        int pdata = bytes_per_row * (newy - y);

        int index = 0x80;
        if (mixmode == MIX_TRANSPARENT) {
            for (int i = 0; i < newcy; i++) {
                for (int j = 0; j < newcx; j++) {
                    if (index == 0) { 
                        pdata++;
                        index = 0x80;
                    }

                    if ((data[pdata] & index) != 0) {
                        if ((x + j >= newx) && (newx + j > 0) && (newy + i > 0) && (y + i >= top))
                            
                            backstore.setRGB(x + j, newy + i, fgcolor);
                    }
                    index >>= 1;
                }
                pdata++;
                index = 0x80;
                pbackstore += this.width;
                if (pdata == data.length) {
                    pdata = 0;
                }
            }
        } else { 
            for (int i = 0; i < newcy; i++) {
                for (int j = 0; j < newcx; j++) {
                    if (index == 0) { 
                        pdata++;
                        index = 0x80;
                    }

                    if (x + j >= newx && y + i >= top) {
                        if ((x + j > 0) && (y + i > 0)) {
                            if ((data[pdata] & index) != 0)
                                backstore.setRGB(x + j, y + i, fgcolor);
                            else
                                backstore.setRGB(x + j, y + i, bgcolor);
                        }
                    }
                    index >>= 1;
                }
                pdata++;
                index = 0x80;
                pbackstore += this.width;
                if (pdata == data.length) {
                    pdata = 0;
                }
            }
        }

        
        
        this.repaint(newx, newy, newcx, newcy);
    }

    /**
     * Create an AWT Cursor object
     *
     * @param x
     * @param y
     * @param w
     * @param h
     * @param andmask
     * @param xormask
     * @param cache_idx
     * @return Created Cursor
     */
    public Cursor createCursor(int cache_idx, int x, int y, int w, int h, byte[] andmask,
                               byte[] xormask, int bpp) {
        Point p = new Point(x, y);
        int delta;
        int scanline = (w + 7) / 8;
        int offset = scanline * h;

        byte[] cursor = new byte[offset];
        byte[] mask = new byte[offset];

        if (bpp == 1) {
            offset = 0;
            delta = scanline;
        } else {
            offset = scanline * h - scanline;
            delta = -scanline;
        }

        int k = 0;
        int j;
        int i;
        int pandmask = 0;
        int pcursor = 0;
        int pmask = 0;
        for (i = 0; i < h; i++) {
            pcursor = offset;
            pmask = offset;
            for (j = 0; j < scanline; j++) {
                for (int nextbit = 0x80; nextbit != 0; nextbit >>= 1) {
                    
                    int rv = 0;
                    int pxormask = 0;
                    int s8 = 0;
                    switch (bpp) {
                        case 1 -> {
                            s8 = pxormask + (k / 8);
                            rv = xormask[s8] & (0x80 >> (k % 8));
                            rv = rv != 0 ? 0xffffff : 0;
                            k += 1;
                        }
                        case 8 -> {
                            s8 = pxormask + k;
                            rv = xormask[s8];
                            rv = rv != 0 ? 0xffffff : 0;
                            k += 1;
                        }
                        case 15 -> {
                            int temp = (xormask[k] << 8) | xormask[k + 1];
                            int red = (((temp >> 7) & 0xf8) | ((temp >> 12) & 0x7));
                            int green = (((temp >> 2) & 0xf8) | ((temp >> 8) & 0x7));
                            int blue = (((temp << 3) & 0xf8) | ((temp >> 2) & 0x7));
                            rv = (red << 16) | (green << 8) | blue;
                            k += 1;
                        }
                        case 16 -> {
                            int temp = (xormask[k] << 8) | xormask[k + 1];
                            int red = ((temp >> 8) & 0xf8) | ((temp >> 13) & 0x7);
                            int green = ((temp >> 3) & 0xfc) | ((temp >> 9) & 0x3);
                            int blue = ((temp << 3) & 0xf8) | ((temp >> 2) & 0x7);
                            rv = (red << 16) | (green << 8) | blue;
                            k += 1;
                        }
                        case 24 -> {
                            s8 = pxormask + k;
                            rv = (xormask[s8] << 16) | (xormask[s8 + 1] << 8) | xormask[s8 + 2];
                            k += 3;
                        }
                        case 32 -> {
                            s8 = pxormask + k;
                            rv = (xormask[s8 + 1] << 16) | (xormask[s8 + 2] << 8) | xormask[s8 + 3];
                            k += 4;
                        }
                        default -> System.out.println("get_next_xor_pixel未知bpp" + bpp);
                    }
                    if (rv != 0) {
                        cursor[pcursor] |= (~(andmask[pandmask]) & nextbit);
                        mask[pmask] |= nextbit;
                    } else {
                        cursor[pcursor] |= ((andmask[pandmask]) & nextbit);
                        mask[pmask] |= (~(andmask[pandmask]) & nextbit);
                    }
                    

                }
                pandmask++;
                pcursor++;
                pmask++;
            }
            offset += delta;
        }

        pmask = 0;
        pcursor = 0;
        int bg = 0xFF000000;
        int fg = 0xFFFFFFFF;
        int pcursormap = 0;
        int[] cursormap = new int[w * h];
        for (i = 0; i < h; i++) {
            for (j = 0; j < scanline; j++) {
                for (int nextbit = 0x80; nextbit != 0; nextbit >>= 1) {
                    boolean isCursor = (cursor[pcursor] & nextbit) != 0;
                    boolean isMask = (mask[pmask] & nextbit) != 0;
                    if (isMask) {
						cursormap[pcursormap] = isCursor ? fg : bg;
                    }
                    pcursormap++;
                }
                pcursor++;
                pmask++;
            }
        }

        Image wincursor = this.createImage(new MemoryImageSource(w, h, cursormap, 0, w));
        return createCustomCursor(wincursor, p, "", cache_idx);
    }

    /**
     * Create an AWT Cursor from an image
     *
     * @param wincursor
     * @param p
     * @param s
     * @param cache_idx
     * @return Generated Cursor object
     */
    Cursor createCustomCursor(Image wincursor, Point p, String s,
                              int cache_idx) {
        if (cache_idx == 1)
            return Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR);
        return Cursor.getDefaultCursor();
    }

    /**
     * Handle the window losing focus, notify input classes
     */
    public void lostFocus() {
        if (input != null)
            input.lostFocus();
    }

    /**
     * Handle the window gaining focus, notify input classes
     */
    public void gainedFocus() {
        if (input != null)
            input.gainedFocus();
    }

    /**
     * Notify the input classes that the connection is ready for sending
     * messages
     */
    public void triggerReadyToSend() {
        input.triggerReadyToSend();
    }
}
