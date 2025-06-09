/*******************************************************************************
 * Copyright (c) 2013, Daniel Murphy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 	* Redistributions of source code must retain the above copyright notice,
 * 	  this list of conditions and the following disclaimer.
 * 	* Redistributions in binary form must reproduce the above copyright notice,
 * 	  this list of conditions and the following disclaimer in the documentation
 * 	  and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/
package spacegraph.space2d.phys.common;

import jcog.math.v2;

/**
 * Orientated bounding box viewport transform
 *
 * @author Daniel Murphy
 */
public class OBBViewportTransform implements IViewportTransform {

    final Mat22 R = new Mat22();
    final v2 center = new v2(), extents = new v2();

    private boolean yFlip;

    public OBBViewportTransform() {
        R.setIdentity();
    }

    public void set(OBBViewportTransform vpt) {
        center.set(vpt.center);
        extents.set(vpt.extents);
        R.set(vpt.R);
        yFlip = vpt.yFlip;
    }

    public void setCamera(float x, float y, float scale) {
        center.set(x, y);
        Mat22.createScaleTransform(scale, R);
    }

    public v2 extents() {
        return extents;
    }

    @Override
    public Mat22 getMat22Representation() {
        return R;
    }

    public void setExtents(v2 argExtents) {
        extents.set(argExtents);
    }

    public void setExtents(float halfWidth, float halfHeight) {
        extents.set(halfWidth, halfHeight);
    }

    public v2 center() {
        return center;
    }

    public void setCenter(v2 argPos) {
        center.set(argPos);
    }

    public void setCenter(float x, float y) {
        center.set(x, y);
    }

    /**
     * Gets the transform of the viewport, transforms around the center. Not a copy.
     */
    public Mat22 getTransform() {
        return R;
    }


    public boolean isYFlip() {
        return yFlip;
    }

    public void setYFlip(boolean yFlip) {
        this.yFlip = yFlip;
    }

    public void getScreenVectorToWorld(v2 screen, v2 world) {
        Mat22 inv = new Mat22();
        R.invertToOut(inv);
        inv.mulToOut(screen, world);
        if (yFlip)
            yFlipMat.mulToOut(world, world);
    }

    public void getWorldVectorToScreen(v2 world, v2 screen) {
        R.mulToOut(world, screen);
        if (yFlip)
            yFlipMat.mulToOut(screen, screen);
    }

    public void getWorldToScreen(v2 world, v2 screen) {
        screen.setSub(world, center);

        R.mulToOut(screen, screen);

        if (yFlip) yFlipMat.mulToOut(screen, screen);

        screen.added(extents);
    }


    public void getScreenToWorld(v2 screen, v2 world) {
        world.setSub(screen, extents);
        if (yFlip) yFlipMat.mulToOut(world, world);

        Mat22 inv2 = new Mat22();
        R.invertToOut(inv2);
        inv2.mulToOut(world, world);

        world.added(center);
    }

    private static final Mat22 yFlipMat = new Mat22(1, 0, 0, -1);

}