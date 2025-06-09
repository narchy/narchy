/* RdesktopCanvas_Localised.java
 * Component: ProperJavaRDP
 *
 * Revision: $Revision: #2 $
 * Author: $Author: tvkelley $
 * Date: $Date: 2009/09/15 $
 *
 * Copyright (c) 2005 Propero Limited
 *
 * Purpose: Java 1.4 specific extension of RdesktopCanvas class
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

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;



public class RdesktopCanvas_Localised extends RdesktopCanvas {

    private static final long serialVersionUID = -6806580381785981945L;
    private Robot robot;

    RdesktopCanvas_Localised(int width, int height) {
        super(width, height);
        BufferedImage apex_backstore = new BufferedImage(width, height,
                BufferedImage.TYPE_INT_RGB);
    }

    private static void saveToFile(Image image) {
        if (Options.server_bpp == 8)
            return;

        BufferedImage img = new BufferedImage(image.getWidth(null), image.getHeight(null),
                BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        g.drawImage(image, 0, 0, null);

        
        try {
            
            File file = new File("./testimages/" + Options.imgCount + ".jpg");
            Options.imgCount++;
            ImageIO.write(img, "jpg", file);
        } catch (IOException e) {
        }

        g.dispose();
    }

    @Override
    public void movePointer(int x, int y) {
        Point p = this.getLocationOnScreen();
        x += p.x;
        y += p.y;
        robot.mouseMove(x, y);
    }

    @Override
    protected Cursor createCustomCursor(Image wincursor, Point p, String s,
                                        int cache_idx) {
        return Toolkit.getDefaultToolkit().createCustomCursor(wincursor, p, "");
    }

    @Override
    public void addNotify() {
        super.addNotify();

        if (robot == null) {
            try {
                robot = new Robot();
            } catch (AWTException e) {
                logger.warn("Pointer movement not allowed");
            }
        }
    }

    @Override
    public void update(Graphics g) {

        Rectangle r = g.getClipBounds();
        g.drawImage(backstore.getSubimage(r.x, r.y, r.width, r.height), r.x,
                r.y, null);

        if (Options.save_graphics) {
            RdesktopCanvas_Localised.saveToFile(backstore.getSubimage(r.x, r.y,
                    r.width, r.height));
        }

        

    }

}
