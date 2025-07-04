/* Bitmap.java
 * Component: ProperJavaRDP
 *
 * Revision: $Revision: #2 $
 * Author: $Author: tvkelley $
 * Date: $Date: 2009/09/15 $
 *
 * Copyright (c) 2005 Propero Limited
 *
 * Purpose: Provide a class for storage of Bitmap images, along with
 *          static methods for decompression and conversion of bitmaps.
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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;

public class Bitmap {

    protected static Logger logger = LoggerFactory.getLogger(Bitmap.class);
    private static int bmpCount;
    private final int[] highdata;
    private final int width;
    private final int height;
    private final int x;
    private final int y;
    public int usage;

    /**
     * Constructor for Bitmap based on integer pixel values
     *
     * @param data   Array of pixel data, one integer per pixel. Should have a
     *               length of width*height.
     * @param width  Width of bitmap represented by data
     * @param height Height of bitmap represented by data
     * @param x      Desired x-coordinate of bitmap
     * @param y      Desired y-coordinate of bitmap
     */
    public Bitmap(int[] data, int width, int height, int x, int y) {
        this.highdata = data;
        this.width = width;
        this.height = height;
        this.x = x;
        this.y = y;
    }

    /**
     * Constructor for Bitmap based on
     *
     * @param data   Array of pixel data, each pixel represented by Bpp bytes.
     *               Should have a length of width*height*Bpp.
     * @param width  Width of bitmap represented by data
     * @param height Height of bitmap represented by data
     * @param x      Desired x-coordinate of bitmap
     * @param y      Desired y-coordinate of bitmap
     * @param Bpp    Number of bytes per pixel in image represented by data
     */
    public Bitmap(byte[] data, int width, int height, int x, int y, int Bpp) {
        this.highdata = Bitmap.convertImage(data, Bpp);
        this.width = width;
        this.height = height;
        this.x = x;
        this.y = y;
    }

    public static int convertTo24(int colour) {
        if (Options.server_bpp == 15)
            return convert15to24(colour);
        if (Options.server_bpp == 16)
            return convert16to24(colour);
        return colour;
    }

    private static int convert15to24(int colour16) {
        int r24 = (colour16 >> 7) & 0xF8;

        r24 |= r24 >> 5;
        int g24 = (colour16 >> 2) & 0xF8;
        g24 |= g24 >> 5;
        int b24 = (colour16 << 3) & 0xFF;
        b24 |= b24 >> 5;

        return (r24 << 16) | (g24 << 8) | b24;
    }

    private static int convert16to24(int colour16) {
        int r24 = (colour16 >> 8) & 0xF8;

        r24 |= r24 >> 5;
        int g24 = (colour16 >> 3) & 0xFC;
        g24 |= g24 >> 6;
        int b24 = (colour16 << 3) & 0xFF;
        b24 |= b24 >> 5;

        return (r24 << 16) | (g24 << 8) | b24;
    }

    /**
     * Read integer of a specified byte-length from byte array
     *
     * @param data   Array to read from
     * @param offset Offset in array to read from
     * @param Bpp    Number of bytes to read
     * @return
     */
    private static int cvalx(byte[] data, int offset, int Bpp) {
        int rv = 0;
        switch (Options.server_bpp) {
            case 15: {
                int lower = data[offset] & 0xFF;
                int full = (data[offset + 1] & 0xFF) << 8 | lower;

                int r24 = (full >> 7) & 0xF8;
                r24 |= r24 >> 5;
                int g24 = (full >> 2) & 0xF8;
                g24 |= g24 >> 5;
                int b24 = (lower << 3) & 0xFF;
                b24 |= b24 >> 5;

                return (r24 << 16) | (g24 << 8) | b24;

            }
            case 16:
                int lower = data[offset] & 0xFF;
                int full = (data[offset + 1] & 0xFF) << 8 | lower;

                int r24 = (full >> 8) & 0xF8;
                r24 |= r24 >> 5;
                int g24 = (full >> 3) & 0xFC;
                g24 |= g24 >> 6;
                int b24 = (lower << 3) & 0xFF;
                b24 |= b24 >> 5;

                return (r24 << 16) | (g24 << 8) | b24;

            default:
                for (int i = (Bpp - 1); i >= 0; i--) {
                    rv <<= 8;
                    rv |= data[offset + i] & 0xFF;
                }
                break;
        }

        return rv;
    }

    /**
     * @param input
     * @param startOffset
     * @param offset
     * @param Bpp
     * @return
     */
    private static int getli(byte[] input, int startOffset, int offset, int Bpp) {
        int rv = 0;

        int rOffset = startOffset + (offset * Bpp);
        for (int i = 0; i < Bpp; i++) {
            rv <<= 8;
            rv |= (input[rOffset + (Bpp - i - 1)]) & 0xFF;
        }
        return rv;
    }

    /**
     * @param input
     * @param startlocation
     * @param offset
     * @param value
     * @param Bpp
     */
    private static void setli(byte[] input, int startlocation, int offset, int value,
                              int Bpp) {
        int location = startlocation + offset * Bpp;

        input[location] = (byte) (value & 0xFF);
        if (Bpp > 1)
            input[location + 1] = (byte) ((value & 0xFF00) >> 8);
        if (Bpp > 2)
            input[location + 2] = (byte) ((value & 0xFF0000) >> 16);
    }

    /**
     * Convert byte array representing a bitmap into integer array of pixels
     *
     * @param bitmap Byte array of bitmap data
     * @param Bpp    Bytes-per-pixel for bitmap
     * @return Integer array of pixel data representing input image data
     */
    static int[] convertImage(byte[] bitmap, int Bpp) {
        int[] out = new int[bitmap.length / Bpp];

        for (int i = 0; i < out.length; i++) {
            switch (Bpp) {
                case 1 -> out[i] = bitmap[i] & 0xFF;
                case 2 -> out[i] = ((bitmap[i * 2 + 1] & 0xFF) << 8)
                        | (bitmap[i * Bpp] & 0xFF);
                case 3 -> out[i] = ((bitmap[i * 3 + 2] & 0xFF) << 16)
                        | ((bitmap[i * Bpp + 1] & 0xFF) << 8)
                        | (bitmap[i * Bpp] & 0xFF);
            }
            out[i] = Bitmap.convertTo24(out[i]);
        }
        return out;
    }

    /**
     * Decompress bitmap data from packet and output directly to supplied image
     * object
     *
     * @param width  Width of bitmap to decompress
     * @param height Height of bitmap to decompress
     * @param size   Size of compressed data in bytes
     * @param data   Packet containing bitmap data
     * @param Bpp    Bytes per-pixel for bitmap
     * @param cm     Colour model of bitmap
     * @param left   X offset for drawing bitmap
     * @param top    Y offset for drawing bitmap
     * @param w      Image to draw bitmap to
     * @return Original image object, with decompressed bitmap drawn at
     * specified coordinates
     * @throws RdesktopException
     */
    public static WrappedImage decompressImgDirect(int width, int height,
                                                   int size, RdpPacket_Localised data, int Bpp, IndexColorModel cm,
                                                   int left, int top, WrappedImage w) throws RdesktopException {

        

        byte[] compressed_pixel = new byte[size];
        data.copyToByteArray(compressed_pixel, 0, data.getPosition(), size);
        data.incrementPosition(size);

        int previous = -1, line = 0, prevY = 0;
        int input = 0, end = size;
        int opcode = 0, count = 0, offset = 0, x = width;
        int lastopcode = -1, fom_mask = 0;
        int code = 0, color1 = 0, color2 = 0;
        byte mixmask = 0;
        int mask = 0;
        int mix = 0xffffffff;

        boolean insertmix = false, bicolor = false, isfillormix = false;

        while (input < end) {
            fom_mask = 0;
            code = (compressed_pixel[input++] & 0x000000ff);
            opcode = code >> 4;

            /* Handle different opcode forms */
            switch (opcode) {
                case 0xc, 0xd, 0xe -> {
                    opcode -= 6;
                    count = code & 0xf;
                    offset = 16;
                }
                case 0xf -> {
                    opcode = code & 0xf;
                    if (opcode < 9) {
                        count = (compressed_pixel[input++] & 0xff);
                        count |= ((compressed_pixel[input++] & 0xff) << 8);
                    } else {
                        count = (opcode < 0xb) ? 8 : 1;
                    }
                    offset = 0;
                }
                default -> {
                    opcode >>= 1;
                    count = code & 0x1f;
                    offset = 32;
                }
            }

            /* Handle strange cases for counts */
            if (offset != 0) {
                isfillormix = ((opcode == 2) || (opcode == 7));

                if (count == 0) {
                    count = (compressed_pixel[input++] & 0x000000ff) + (isfillormix ? 1 : offset);
                } else if (isfillormix) {
                    count <<= 3;
                }
            }

            switch (opcode) {
                case 0: /* Fill */
                    if ((lastopcode == opcode)
                            && !((x == width) && (previous == -1)))
                        insertmix = true;
                    break;
                case 8: /* Bicolor */
                    color1 = cvalx(compressed_pixel, input, Bpp);
                    
                    input += Bpp;
                case 3: /* Color */
                    color2 = cvalx(compressed_pixel, input, Bpp);
                    
                    input += Bpp;
                    break;
                case 6: /* SetMix/Mix */
                case 7: /* SetMix/FillOrMix */
                    
                    mix = cvalx(compressed_pixel, input, Bpp);
                    input += Bpp;
                    opcode -= 5;
                    break;
                case 9: /* FillOrMix_1 */
                    mask = 0x03;
                    opcode = 0x02;
                    fom_mask = 3;
                    break;
                case 0x0a: /* FillOrMix_2 */
                    mask = 0x05;
                    opcode = 0x02;
                    fom_mask = 5;
                    break;

            }

            lastopcode = opcode;
            mixmask = 0;

            /* Output body */
            while (count > 0) {
                if (x >= width) {
                    if (height <= 0)
                        throw new RdesktopException(
                                "Decompressing bitmap failed! Height = "
                                        + height);
                    x = 0;
                    height--;

                    previous = line;
                    prevY = previous / width;
                    line = height * width;
                }

                switch (opcode) {
                    case 0: /* Fill */
                        if (insertmix) {
                            if (previous == -1) {
                                
                                w.setRGB(left + x, top + height, mix);
                            } else {
                                w.setRGB(left + x, top + height, w.getRGB(left + x,
                                        top + prevY)
                                        ^ mix);
                                
                            }

                            insertmix = false;
                            count--;
                            x++;
                        }

                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(left + x, top + height, 0);
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(left + x, top + height, 0);
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(left + x, top + height, w.getRGB(left
                                            + x, top + prevY));
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(left + x, top + height, w.getRGB(left + x,
                                        top + prevY));
                                count--;
                                x++;
                            }
                        }
                        break;

                    case 1: /* Mix */
                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(left + x, top + height, mix);
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(left + x, top + height, mix);
                                count--;
                                x++;
                            }
                        } else {

                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(left + x, top + height, w.getRGB(left
                                            + x, top + prevY)
                                            ^ mix);
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(left + x, top + height, w.getRGB(left + x,
                                        top + prevY)
                                        ^ mix);
                                count--;
                                x++;
                            }

                        }
                        break;
                    case 2: /* Fill or Mix */
                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    if ((mask & mixmask) != 0) {
                                        
                                        w
                                                .setRGB(left + x, top + height,
                                                        (byte) mix);
                                    } else {
                                        
                                        w.setRGB(left + x, top + height, 0);
                                    }
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                if ((mask & mixmask) != 0) {
                                    
                                    w.setRGB(left + x, top + height, mix);
                                } else {
                                    
                                    w.setRGB(left + x, top + height, 0);
                                }
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    if ((mask & mixmask) != 0) {
                                        
                                        
                                        w.setRGB(left + x, top + height, w.getRGB(
                                                left + x, prevY + top)
                                                ^ mix);
                                    } else {
                                        
                                        w.setRGB(left + x, top + height, w.getRGB(
                                                left + x, prevY + top));
                                    }
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                if ((mask & mixmask) != 0) {
                                    
                                    
                                    w.setRGB(left + x, top + height, w.getRGB(left
                                            + x, prevY + top)
                                            ^ mix);
                                } else {
                                    
                                    w.setRGB(left + x, top + height, w.getRGB(left
                                            + x, prevY + top));
                                }
                                count--;
                                x++;
                            }

                        }
                        break;

                    case 3: /* Color */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                w.setRGB(left + x, top + height, color2);
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            w.setRGB(left + x, top + height, color2);
                            count--;
                            x++;
                        }

                        break;

                    case 4: /* Copy */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                
                                w.setRGB(left + x, top + height, cvalx(
                                        compressed_pixel, input, Bpp));
                                input += Bpp;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            
                            w.setRGB(left + x, top + height, cvalx(
                                    compressed_pixel, input, Bpp));
                            input += Bpp;
                            count--;
                            x++;
                        }
                        break;

                    case 8: /* Bicolor */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                if (bicolor) {
                                    
                                    w.setRGB(left + x, top + height, color2);
                                    bicolor = false;
                                } else {
                                    
                                    w.setRGB(left + x, top + height, color1);
                                    bicolor = true;
                                    count++;
                                }
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            if (bicolor) {
                                
                                w.setRGB(left + x, top + height, color2);
                                bicolor = false;
                            } else {
                                
                                w.setRGB(left + x, top + height, color1);
                                bicolor = true;
                                count++;
                            }
                            count--;
                            x++;
                        }

                        break;

                    case 0xd: /* White */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                w.setRGB(left + x, top + height, 0xffffff);
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            w.setRGB(left + x, top + height, 0xffffff);
                            count--;
                            x++;
                        }
                        break;

                    case 0xe: /* Black */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                w.setRGB(left + x, top + height, 0x00);
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            w.setRGB(left + x, top + height, 0x00);
                            count--;
                            x++;
                        }

                        break;
                    default:
                        throw new RdesktopException(
                                "Unimplemented decompress opcode " + opcode);
                }
            }
        }

        /*
         * if(Options.server_bpp == 16){ for(int i = 0; i < pixel.length; i++)
         * pixel[i] = Bitmap.convert16to24(pixel[i]); }
         */

        return w;
    }

    /**
     * Decompress bitmap data from packet and output as an Image
     *
     * @param width  Width of bitmap
     * @param height Height of bitmap
     * @param size   Size of compressed data in bytes
     * @param data   Packet containing bitmap data
     * @param Bpp    Bytes per-pixel for bitmap
     * @param cm     Colour model for bitmap (if using indexed palette)
     * @return Decompressed bitmap as Image object
     * @throws RdesktopException
     */
    public static Image decompressImg(int width, int height, int size,
                                      RdpPacket_Localised data, int Bpp, IndexColorModel cm)
            throws RdesktopException {

        byte[] compressed_pixel = new byte[size];
        data.copyToByteArray(compressed_pixel, 0, data.getPosition(), size);
        data.incrementPosition(size);

        WrappedImage w = null;
        w = cm == null ? new WrappedImage(width, height, BufferedImage.TYPE_INT_RGB) : new WrappedImage(width, height, BufferedImage.TYPE_INT_RGB, cm);

        boolean isfillormix = false;
        boolean bicolor = false;
        boolean insertmix = false;
        int mix = 0xffffffff;
        int mask = 0;
        byte mixmask = 0;
        int color2 = 0;
        int color1 = 0;
        int code = 0;
        int fom_mask = 0;
        int lastopcode = -1;
        int x = width;
        int offset = 0;
        int count = 0;
        int opcode = 0;
        int end = size;
        int input = 0;
        int prevY = 0;
        int line = 0;
        int previous = -1;
        while (input < end) {
            fom_mask = 0;
            code = (compressed_pixel[input++] & 0x000000ff);
            opcode = code >> 4;

            /* Handle different opcode forms */
            switch (opcode) {
                case 0xc, 0xd, 0xe -> {
                    opcode -= 6;
                    count = code & 0xf;
                    offset = 16;
                }
                case 0xf -> {
                    opcode = code & 0xf;
                    if (opcode < 9) {
                        count = (compressed_pixel[input++] & 0xff);
                        count |= ((compressed_pixel[input++] & 0xff) << 8);
                    } else {
                        count = (opcode < 0xb) ? 8 : 1;
                    }
                    offset = 0;
                }
                default -> {
                    opcode >>= 1;
                    count = code & 0x1f;
                    offset = 32;
                }
            }

            /* Handle strange cases for counts */
            if (offset != 0) {
                isfillormix = ((opcode == 2) || (opcode == 7));

                if (count == 0) {
                    count = (compressed_pixel[input++] & 0x000000ff) + (isfillormix ? 1 : offset);
                } else if (isfillormix) {
                    count <<= 3;
                }
            }

            switch (opcode) {
                case 0: /* Fill */
                    if ((lastopcode == opcode)
                            && !((x == width) && (previous == -1)))
                        insertmix = true;
                    break;
                case 8: /* Bicolor */
                    color1 = cvalx(compressed_pixel, input, Bpp);
                    
                    input += Bpp;
                case 3: /* Color */
                    color2 = cvalx(compressed_pixel, input, Bpp);
                    
                    input += Bpp;
                    break;
                case 6: /* SetMix/Mix */
                case 7: /* SetMix/FillOrMix */
                    
                    mix = cvalx(compressed_pixel, input, Bpp);
                    input += Bpp;
                    opcode -= 5;
                    break;
                case 9: /* FillOrMix_1 */
                    mask = 0x03;
                    opcode = 0x02;
                    fom_mask = 3;
                    break;
                case 0x0a: /* FillOrMix_2 */
                    mask = 0x05;
                    opcode = 0x02;
                    fom_mask = 5;
                    break;

            }

            lastopcode = opcode;
            mixmask = 0;

            /* Output body */
            while (count > 0) {
                if (x >= width) {
                    if (height <= 0)
                        throw new RdesktopException(
                                "Decompressing bitmap failed! Height = "
                                        + height);
                    x = 0;
                    height--;

                    previous = line;
                    prevY = previous / width;
                    line = height * width;
                }

                switch (opcode) {
                    case 0: /* Fill */
                        if (insertmix) {
                            if (previous == -1) {
                                
                                w.setRGB(x, height, mix);
                            } else {
                                w.setRGB(x, height, w.getRGB(x, prevY) ^ mix);
                                
                            }

                            insertmix = false;
                            count--;
                            x++;
                        }

                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(x, height, 0);
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(x, height, 0);
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(x, height, w.getRGB(x, prevY));
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(x, height, w.getRGB(x, prevY));
                                count--;
                                x++;
                            }
                        }
                        break;

                    case 1: /* Mix */
                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(x, height, mix);
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(x, height, mix);
                                count--;
                                x++;
                            }
                        } else {

                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    w.setRGB(x, height, w.getRGB(x, prevY) ^ mix);
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                w.setRGB(x, height, w.getRGB(x, prevY) ^ mix);
                                count--;
                                x++;
                            }

                        }
                        break;
                    case 2: /* Fill or Mix */
                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    if ((mask & mixmask) != 0) {
                                        
                                        w.setRGB(x, height, (byte) mix);
                                    } else {
                                        
                                        w.setRGB(x, height, 0);
                                    }
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                if ((mask & mixmask) != 0) {
                                    
                                    w.setRGB(x, height, mix);
                                } else {
                                    
                                    w.setRGB(x, height, 0);
                                }
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    if ((mask & mixmask) != 0) {
                                        
                                        
                                        w.setRGB(x, height, w.getRGB(x, prevY)
                                                ^ mix);
                                    } else {
                                        
                                        w.setRGB(x, height, w.getRGB(x, prevY));
                                    }
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                if ((mask & mixmask) != 0) {
                                    
                                    
                                    w.setRGB(x, height, w.getRGB(x, prevY) ^ mix);
                                } else {
                                    
                                    w.setRGB(x, height, w.getRGB(x, prevY));
                                }
                                count--;
                                x++;
                            }

                        }
                        break;

                    case 3: /* Color */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                w.setRGB(x, height, color2);
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            w.setRGB(x, height, color2);
                            count--;
                            x++;
                        }

                        break;

                    case 4: /* Copy */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                
                                w.setRGB(x, height, cvalx(compressed_pixel, input,
                                        Bpp));
                                input += Bpp;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            
                            w
                                    .setRGB(x, height, cvalx(compressed_pixel,
                                            input, Bpp));
                            input += Bpp;
                            
                            count--;
                            x++;
                        }
                        break;

                    case 8: /* Bicolor */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                if (bicolor) {
                                    
                                    w.setRGB(x, height, color2);
                                    bicolor = false;
                                } else {
                                    
                                    w.setRGB(x, height, color1);
                                    bicolor = true;
                                    count++;
                                }
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            if (bicolor) {
                                
                                w.setRGB(x, height, color2);
                                bicolor = false;
                            } else {
                                
                                w.setRGB(x, height, color1);
                                bicolor = true;
                                count++;
                            }
                            count--;
                            x++;
                        }

                        break;

                    case 0xd: /* White */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                w.setRGB(x, height, 0xffffff);
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            w.setRGB(x, height, 0xffffff);
                            count--;
                            x++;
                        }
                        break;

                    case 0xe: /* Black */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                w.setRGB(x, height, 0x00);
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            w.setRGB(x, height, 0x00);
                            count--;
                            x++;
                        }

                        break;
                    default:
                        throw new RdesktopException(
                                "Unimplemented decompress opcode " + opcode);
                }
            }
        }

        /*
         * if(Options.server_bpp == 16){ for(int i = 0; i < pixel.length; i++)
         * pixel[i] = Bitmap.convert16to24(pixel[i]); }
         */

        return w.getBufferedImage();
    }

    /**
     * Decompress bitmap data from packet and store in array of integers
     *
     * @param width  Width of bitmap
     * @param height Height of bitmap
     * @param size   Size of compressed data in bytes
     * @param data   Packet containing bitmap data
     * @param Bpp    Bytes per-pixel for bitmap
     * @return Integer array of pixels containing decompressed bitmap data
     * @throws RdesktopException
     */
    public static int[] decompressInt(int width, int height, int size,
                                      RdpPacket_Localised data, int Bpp) throws RdesktopException {

        byte[] compressed_pixel = new byte[size];
        data.copyToByteArray(compressed_pixel, 0, data.getPosition(), size);
        data.incrementPosition(size);

        int previous = -1, line = 0;
        int input = 0, output = 0, end = size;
        int opcode = 0, count = 0, offset = 0, x = width;
        int lastopcode = -1, fom_mask = 0;
        int code = 0, color1 = 0, color2 = 0;
        byte mixmask = 0;
        int mask = 0;
        int mix = 0xffffffff;

        boolean insertmix = false, bicolor = false, isfillormix = false;

        int[] pixel = new int[width * height];
        while (input < end) {
            fom_mask = 0;
            code = (compressed_pixel[input++] & 0x000000ff);
            opcode = code >> 4;

            /* Handle different opcode forms */
            switch (opcode) {
                case 0xc, 0xd, 0xe -> {
                    opcode -= 6;
                    count = code & 0xf;
                    offset = 16;
                }
                case 0xf -> {
                    opcode = code & 0xf;
                    if (opcode < 9) {
                        count = (compressed_pixel[input++] & 0xff);
                        count |= ((compressed_pixel[input++] & 0xff) << 8);
                    } else {
                        count = (opcode < 0xb) ? 8 : 1;
                    }
                    offset = 0;
                }
                default -> {
                    opcode >>= 1;
                    count = code & 0x1f;
                    offset = 32;
                }
            }

            /* Handle strange cases for counts */
            if (offset != 0) {
                isfillormix = ((opcode == 2) || (opcode == 7));

                if (count == 0) {
                    count = (compressed_pixel[input++] & 0x000000ff) + (isfillormix ? 1 : offset);
                } else if (isfillormix) {
                    count <<= 3;
                }
            }

            switch (opcode) {
                case 0: /* Fill */
                    if ((lastopcode == opcode)
                            && !((x == width) && (previous == -1)))
                        insertmix = true;
                    break;
                case 8: /* Bicolor */
                    color1 = cvalx(compressed_pixel, input, Bpp);
                    
                    input += Bpp;
                case 3: /* Color */
                    color2 = cvalx(compressed_pixel, input, Bpp);
                    
                    input += Bpp;
                    break;
                case 6: /* SetMix/Mix */
                case 7: /* SetMix/FillOrMix */
                    
                    mix = cvalx(compressed_pixel, input, Bpp);
                    input += Bpp;
                    opcode -= 5;
                    break;
                case 9: /* FillOrMix_1 */
                    mask = 0x03;
                    opcode = 0x02;
                    fom_mask = 3;
                    break;
                case 0x0a: /* FillOrMix_2 */
                    mask = 0x05;
                    opcode = 0x02;
                    fom_mask = 5;
                    break;

            }

            lastopcode = opcode;
            mixmask = 0;

            /* Output body */
            while (count > 0) {
                if (x >= width) {
                    if (height <= 0)
                        throw new RdesktopException(
                                "Decompressing bitmap failed! Height = "
                                        + height);

                    x = 0;
                    height--;

                    previous = line;
                    line = output + height * width;
                }

                switch (opcode) {
                    case 0: /* Fill */
                        if (insertmix) {
                            pixel[line + x] = previous == -1 ? mix : pixel[previous + x] ^ mix;

                            insertmix = false;
                            count--;
                            x++;
                        }

                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    pixel[line + x] = 0;
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                pixel[line + x] = 0;
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    pixel[line + x] = pixel[previous + x];
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                pixel[line + x] = pixel[previous + x];
                                count--;
                                x++;
                            }
                        }
                        break;

                    case 1: /* Mix */
                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    pixel[line + x] = mix;
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                pixel[line + x] = mix;
                                count--;
                                x++;
                            }
                        } else {

                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    
                                    pixel[line + x] = pixel[previous + x] ^ mix;
                                    
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                pixel[line + x] = pixel[previous + x] ^ mix;
                                
                                count--;
                                x++;
                            }

                        }
                        break;
                    case 2: /* Fill or Mix */
                        if (previous == -1) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    pixel[line + x] = (mask & mixmask) != 0 ? (byte) mix : 0;
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                pixel[line + x] = (mask & mixmask) != 0 ? mix : 0;
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    pixel[line + x] = (mask & mixmask) != 0 ? pixel[previous + x] ^ mix : pixel[previous + x];
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                pixel[line + x] = (mask & mixmask) != 0 ? pixel[previous + x] ^ mix : pixel[previous + x];
                                count--;
                                x++;
                            }

                        }
                        break;

                    case 3: /* Color */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                pixel[line + x] = color2;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            pixel[line + x] = color2;
                            count--;
                            x++;
                        }

                        break;

                    case 4: /* Copy */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                
                                
                                pixel[line + x] = cvalx(compressed_pixel, input,
                                        Bpp);
                                input += Bpp;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            
                            pixel[line + x] = cvalx(compressed_pixel, input, Bpp);
                            input += Bpp;
                            
                            count--;
                            x++;
                        }
                        break;

                    case 8: /* Bicolor */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                if (bicolor) {
                                    
                                    pixel[line + x] = color2;
                                    bicolor = false;
                                } else {
                                    
                                    pixel[line + x] = color1;
                                    bicolor = true;
                                    count++;
                                }
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            if (bicolor) {
                                
                                pixel[line + x] = color2;
                                bicolor = false;
                            } else {
                                
                                pixel[line + x] = color1;
                                bicolor = true;
                                count++;
                            }
                            count--;
                            x++;
                        }

                        break;

                    case 0xd: /* White */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                pixel[line + x] = 0xffffff;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            pixel[line + x] = 0xffffff;
                            count--;
                            x++;
                        }
                        break;

                    case 0xe: /* Black */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                pixel[line + x] = 0x00;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            pixel[line + x] = 0x00;
                            count--;
                            x++;
                        }

                        break;
                    default:
                        throw new RdesktopException(
                                "Unimplemented decompress opcode " + opcode);
                }
            }
        }

        /*
         * if(Options.server_bpp == 16){ for(int i = 0; i < pixel.length; i++)
         * pixel[i] = Bitmap.convert16to24(pixel[i]); }
         */

        return pixel;
    }

    /**
     * Decompress bitmap data from packet and store in array of bytes
     *
     * @param width  Width of bitmap
     * @param height Height of bitmap
     * @param size   Size of compressed data in bytes
     * @param data   Packet containing bitmap data
     * @param Bpp    Bytes per-pixel for bitmap
     * @return Byte array of pixels containing decompressed bitmap data
     * @throws RdesktopException
     */
    public static byte[] decompress(int width, int height, int size,
                                    RdpPacket_Localised data, int Bpp) throws RdesktopException {

        byte[] compressed_pixel = new byte[size];
        data.copyToByteArray(compressed_pixel, 0, data.getPosition(), size);
        data.incrementPosition(size);

        int previous = 0, line = 0;
        int input = 0, output = 0, end = size;
        int opcode = 0, count = 0, offset = 0, x = width;
        int lastopcode = -1, fom_mask = 0;
        int code = 0, color1 = 0, color2 = 0;
        byte mixmask = 0;
        int mask = 0;
        int mix = 0xffffffff;

        boolean insertmix = false, bicolor = false, isfillormix = false;

        byte[] pixel = new byte[width * height];
        while (input < end) {
            fom_mask = 0;
            code = (compressed_pixel[input++] & 0x000000ff);
            opcode = code >> 4;

            /* Handle different opcode forms */
            switch (opcode) {
                case 0xc, 0xd, 0xe -> {
                    opcode -= 6;
                    count = code & 0xf;
                    offset = 16;
                }
                case 0xf -> {
                    opcode = code & 0xf;
                    if (opcode < 9) {
                        count = (compressed_pixel[input++] & 0xff);
                        count |= ((compressed_pixel[input++] & 0xff) << 8);
                    } else {
                        count = (opcode < 0xb) ? 8 : 1;
                    }
                    offset = 0;
                }
                default -> {
                    opcode >>= 1;
                    count = code & 0x1f;
                    offset = 32;
                }
            }

            /* Handle strange cases for counts */
            if (offset != 0) {
                isfillormix = ((opcode == 2) || (opcode == 7));

                if (count == 0) {
                    count = (compressed_pixel[input++] & 0x000000ff) + (isfillormix ? 1 : offset);
                } else if (isfillormix) {
                    count <<= 3;
                }
            }

            switch (opcode) {
                case 0: /* Fill */
                    if ((lastopcode == opcode)
                            && !((x == width) && (previous == 0)))
                        insertmix = true;
                    break;
                case 8: /* Bicolor */
                    
                    
                    color1 = (compressed_pixel[input++] & 0x000000ff);
                    
                case 3: /* Color */
                    
                    color2 = (compressed_pixel[input++] & 0x000000ff);
                    
                    break;
                case 6: /* SetMix/Mix */
                case 7: /* SetMix/FillOrMix */
                    mix = compressed_pixel[input++];
                    
                    
                    opcode -= 5;
                    break;
                case 9: /* FillOrMix_1 */
                    mask = 0x03;
                    opcode = 0x02;
                    fom_mask = 3;
                    break;
                case 0x0a: /* FillOrMix_2 */
                    mask = 0x05;
                    opcode = 0x02;
                    fom_mask = 5;
                    break;

            }

            lastopcode = opcode;
            mixmask = 0;

            /* Output body */
            while (count > 0) {
                if (x >= width) {
                    if (height <= 0)
                        throw new RdesktopException(
                                "Decompressing bitmap failed! Height = "
                                        + height);

                    x = 0;
                    height--;

                    previous = line;
                    line = output + height * width;
                }

                switch (opcode) {
                    case 0: /* Fill */
                        if (insertmix) {
                            pixel[line + x] = (byte) (previous == 0 ? mix : pixel[previous + x] ^ (byte) mix);

                            insertmix = false;
                            count--;
                            x++;
                        }

                        if (previous == 0) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    pixel[line + x] = 0;
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                pixel[line + x] = 0;
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    
                                    pixel[line + x] = pixel[previous + x];
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                
                                pixel[line + x] = pixel[previous + x];
                                count--;
                                x++;
                            }
                        }
                        break;

                    case 1: /* Mix */
                        if (previous == 0) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    
                                    pixel[line + x] = (byte) mix;
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                
                                pixel[line + x] = (byte) mix;
                                count--;
                                x++;
                            }
                        } else {

                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    setli(pixel, line, x, getli(pixel, previous, x,
                                            1)
                                            ^ mix, 1);
                                    
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                setli(pixel, line, x, getli(pixel, previous, x, 1)
                                        ^ mix, 1);
                                
                                count--;
                                x++;
                            }

                        }
                        break;
                    case 2: /* Fill or Mix */
                        if (previous == 0) {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    pixel[line + x] = (mask & mixmask) != 0 ? (byte) mix : 0;
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                pixel[line + x] = (mask & mixmask) != 0 ? (byte) mix : 0;
                                count--;
                                x++;
                            }
                        } else {
                            while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                                for (int i = 0; i < 8; i++) {
                                    mixmask <<= 1;
                                    if (mixmask == 0) {
                                        mask = (fom_mask != 0) ? (byte) fom_mask
                                                : compressed_pixel[input++];
                                        mixmask = 1;
                                    }
                                    pixel[line + x] = (mask & mixmask) != 0 ? (byte) (pixel[previous
                                        + x] ^ (byte) mix) : pixel[previous + x];
                                    count--;
                                    x++;
                                }
                            }
                            while ((count > 0) && (x < width)) {
                                mixmask <<= 1;
                                if (mixmask == 0) {
                                    mask = (fom_mask != 0) ? (byte) fom_mask
                                            : compressed_pixel[input++];
                                    mixmask = 1;
                                }
                                pixel[line + x] = (mask & mixmask) != 0 ? (byte) (pixel[previous + x] ^ (byte) mix) : pixel[previous + x];
                                count--;
                                x++;
                            }

                        }
                        break;

                    case 3: /* Color */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                pixel[line + x] = (byte) color2;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            pixel[line + x] = (byte) color2;
                            count--;
                            x++;
                        }

                        break;

                    case 4: /* Copy */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                
                                pixel[line + x] = compressed_pixel[input++];
                                
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            
                            
                            pixel[line + x] = compressed_pixel[input++];
                            count--;
                            x++;
                        }
                        break;

                    case 8: /* Bicolor */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                if (bicolor) {
                                    
                                    pixel[line + x] = (byte) color2;
                                    bicolor = false;
                                } else {
                                    
                                    pixel[line + x] = (byte) color1;
                                    bicolor = true;
                                    count++;
                                }
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            if (bicolor) {
                                
                                pixel[line + x] = (byte) color2;
                                bicolor = false;
                            } else {
                                
                                pixel[line + x] = (byte) color1;
                                bicolor = true;
                                count++;
                            }
                            count--;
                            x++;
                        }

                        break;

                    case 0xd: /* White */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                pixel[line + x] = (byte) 0xff;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            pixel[line + x] = (byte) 0xff;
                            count--;
                            x++;
                        }
                        break;

                    case 0xe: /* Black */
                        while (((count & ~0x7) != 0) && ((x + 8) < width)) {
                            for (int i = 0; i < 8; i++) {
                                
                                pixel[line + x] = 0x00;
                                count--;
                                x++;
                            }
                        }
                        while ((count > 0) && (x < width)) {
                            
                            pixel[line + x] = 0x00;
                            count--;
                            x++;
                        }

                        break;
                    default:
                        throw new RdesktopException(
                                "Unimplemented decompress opcode " + opcode);
                }
            }
        }

        bmpCount++;

        return pixel;
    }

    /**
     * Retrieve data representing this Bitmap, as an array of integer pixel
     * values
     *
     * @return Bitmap pixel data
     */
    public int[] getBitmapData() {
        return this.highdata;
    }

    /**
     * Retrieve width of the bitmap represented by this object
     *
     * @return Bitmap width
     */
    public int getWidth() {
        return this.width;
    }

    /**
     * Retrieve height of the bitmap represented by this object
     *
     * @return Bitmap height
     */
    public int getHeight() {
        return this.height;
    }

    /**
     * Retrieve desired x-coordinate of the bitmap represented by this object
     *
     * @return x-coordinate of this bitmap
     */
    public int getX() {
        return this.x;
    }

    /**
     * Retrieve desired y-coordinate of the bitmap represented by this object
     *
     * @return y-coordinate of this bitmap
     */
    public int getY() {
        return this.y;
    }
}
