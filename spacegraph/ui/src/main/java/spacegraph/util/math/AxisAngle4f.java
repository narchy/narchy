/*
 * $RCSfile$
 *
 * Copyright 1997-2008 Sun Microsystems, Inc.  All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Sun designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Sun in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Sun Microsystems, Inc., 4150 Network Circle, Santa Clara,
 * CA 95054 USA or visit www.sun.com if you need additional information or
 * have any questions.
 *
 * $Revision: 127 $
 * $Date: 2008-02-28 17:18:51 -0300 (Thu, 28 Feb 2008) $
 * $State$
 */

package spacegraph.util.math;

import com.jogamp.opengl.math.Quaternion;
import jcog.math.VecMathUtil;
import jcog.math.v3;

import java.io.Serializable;

/**
 * A four-element axis angle represented by single-precision floating point
 * x,y,z,angle components.  An axis angle is a rotation of angle (radians)
 * about the vector (x,y,z).
 */
public class AxisAngle4f implements Serializable, Cloneable {


    /**
     * The x coordinate.
     */
    public float x;

    /**
     * The y coordinate.
     */
    public float y;

    /**
     * The z coordinate.
     */
    public float z;

    /**
     * The angle of rotation in radians.
     */
    public float angle;

    private static final float EPS = 0.000001f;

    /**
     * Constructs and initializes a AxisAngle4f from the specified xyzw coordinates.
     *
     * @param x     the x coordinate
     * @param y     the y coordinate
     * @param z     the z coordinate
     * @param angle the angle of rotation in radians
     */
    public AxisAngle4f(float x, float y, float z, float angle) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.angle = angle;
    }


    /**
     * Constructs and initializes an AxisAngle4f from the array of length 4.
     *
     * @param a the array of length 4 containing x,y,z,angle in order
     */
    public AxisAngle4f(float[] a) {
        this.x = a[0];
        this.y = a[1];
        this.z = a[2];
        this.angle = a[3];
    }


    /**
     * Constructs and initializes an AxisAngle4f from the specified
     * AxisAngle4f.
     *
     * @param a1 the AxisAngle4f containing the initialization x y z angle data
     */
    public AxisAngle4f(AxisAngle4f a1) {
        this.x = a1.x;
        this.y = a1.y;
        this.z = a1.z;
        this.angle = a1.angle;
    }


    /**
     * Constructs and initializes an AxisAngle4f from the specified
     * axis and angle.
     *
     * @param axis  the axis
     * @param angle the angle of rotation in radians
     * @since vecmath 1.2
     */
    public AxisAngle4f(v3 axis, float angle) {
        this.x = axis.x;
        this.y = axis.y;
        this.z = axis.z;
        this.angle = angle;
    }


    /**
     * Constructs and initializes an AxisAngle4f to (0,0,1,0).
     */
    public AxisAngle4f() {
        this.x = 0.0f;
        this.y = 0.0f;
        this.z = 1.0f;
        this.angle = 0.0f;
    }


    /**
     * Sets the value of this axis-angle to the specified x,y,z,angle.
     *
     * @param x     the x coordinate
     * @param y     the y coordinate
     * @param z     the z coordinate
     * @param angle the angle of rotation in radians
     */
    public final void set(float x, float y, float z, float angle) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.angle = angle;
    }


    /**
     * Sets the value of this axis-angle to the specified values in the
     * array of length 4.
     *
     * @param a the array of length 4 containing x,y,z,angle in order
     */
    public final void set(float[] a) {
        this.x = a[0];
        this.y = a[1];
        this.z = a[2];
        this.angle = a[3];
    }


    /**
     * Sets the value of this axis-angle to the value of axis-angle a1.
     *
     * @param a1 the axis-angle to be copied
     */
    public final void set(AxisAngle4f a1) {
        this.x = a1.x;
        this.y = a1.y;
        this.z = a1.z;
        this.angle = a1.angle;
    }


    /**
     * Sets the value of this AxisAngle4f to the specified
     * axis and angle.
     *
     * @param axis  the axis
     * @param angle the angle of rotation in radians
     * @since vecmath 1.2
     */
    public final void set(v3 axis, float angle) {
        this.x = axis.x;
        this.y = axis.y;
        this.z = axis.z;
        this.angle = angle;
    }


    /**
     * Copies the value of this axis-angle into the array a.
     *
     * @param a the array
     */
    public final void get(float[] a) {
        a[0] = this.x;
        a[1] = this.y;
        a[2] = this.z;
        a[3] = this.angle;
    }

    public final void get(v3 angle) {
        angle.set(x, y, z);
    }


    /**
     * Sets the value of this axis-angle to the rotational equivalent
     * of the passed quaternion.
     * If the specified quaternion has no rotational component, the value
     * of this AxisAngle4f is set to an angle of 0 about an axis of (0,1,0).
     *
     * @param q1 the Quat4f
     */
    public final void set(Quaternion q1, boolean angle) {
        float qz = q1.z();
        float qy = q1.y() * q1.y();
        float qx = q1.x() * q1.x();
        float mag = qx + qy + qz * qz;
        if (mag > EPS) {
            mag = (float) Math.sqrt(mag);
            x = qx / mag;
            y = qy / mag;
            z = qz / mag;
            this.angle = angle ? (float) (2 * Math.atan2(mag, q1.w())) : 0;
        } else {
            x = 0f;
            y = 1f;
            z = 0f;
            this.angle = 0f;
        }
    }


    /**
     * Returns a string that contains the values of this AxisAngle4f.
     * The form is (x,y,z,angle).
     *
     * @return the String representation
     */
    public String toString() {
        return "(" + this.x + ", " + this.y + ", " + this.z + ", " + this.angle + ')';
    }


    /**
     * Returns true if all of the data members of AxisAngle4f a1 are
     * equal to the corresponding data members in this AxisAngle4f.
     *
     * @param a1 the axis-angle with which the comparison is made
     * @return true or false
     */
    public boolean equals(AxisAngle4f a1) {
        try {
            return (this.x == a1.x && this.y == a1.y && this.z == a1.z
                    && this.angle == a1.angle);
        } catch (NullPointerException e2) {
            return false;
        }

    }

    /**
     * Returns true if the Object o1 is of type AxisAngle4f and all of the
     * data members of o1 are equal to the corresponding data members in
     * this AxisAngle4f.
     *
     * @param o1 the object with which the comparison is made
     * @return true or false
     */
    public boolean equals(Object o1) {
        try {
            AxisAngle4f a2 = (AxisAngle4f) o1;
            return (this.x == a2.x && this.y == a2.y && this.z == a2.z
                    && this.angle == a2.angle);
        } catch (NullPointerException | ClassCastException e2) {
            return false;
        }

    }

    /**
     * Returns true if the L-infinite distance between this axis-angle
     * and axis-angle a1 is less than or equal to the epsilon parameter,
     * otherwise returns false.  The L-infinite
     * distance is equal to
     * MAX[abs(x1-x2), abs(y1-y2), abs(z1-z2), abs(angle1-angle2)].
     *
     * @param a1      the axis-angle to be compared to this axis-angle
     * @param epsilon the threshold value
     */
    public boolean epsilonEquals(AxisAngle4f a1, float epsilon) {

        float diff = x - a1.x;
        if ((diff < 0 ? -diff : diff) > epsilon) return false;

        diff = y - a1.y;
        if ((diff < 0 ? -diff : diff) > epsilon) return false;

        diff = z - a1.z;
        if ((diff < 0 ? -diff : diff) > epsilon) return false;

        diff = angle - a1.angle;
        return (diff < 0 ? -diff : diff) <= epsilon;

    }


    /**
     * Returns a hash code value based on the data values in this
     * object.  Two different AxisAngle4f objects with identical data values
     * (i.e., AxisAngle4f.equals returns true) will return the same hash
     * code value.  Two objects with different data members may return the
     * same hash value, although this is not likely.
     *
     * @return the integer hash code value
     */
    public int hashCode() {
        long bits = 1L;
        bits = 31L * bits + VecMathUtil.floatToIntBits(x);
        bits = 31L * bits + VecMathUtil.floatToIntBits(y);
        bits = 31L * bits + VecMathUtil.floatToIntBits(z);
        bits = 31L * bits + VecMathUtil.floatToIntBits(angle);
        return (int) (bits ^ (bits >> 32));
    }

    /**
     * Creates a new object of the same class as this object.
     *
     * @return a clone of this instance.
     * @throws OutOfMemoryError if there is not enough memory.
     * @see Cloneable
     * @since vecmath 1.3
     */
    @Override
    public Object clone() {

        try {
            return super.clone();
        } catch (CloneNotSupportedException e) {

            throw new InternalError();
        }
    }


    /**
     * Get the axis angle, in radians.<br>
     * An axis angle is a rotation angle about the vector (x,y,z).
     *
     * @return Returns the angle, in radians.
     * @since vecmath 1.5
     */
    public final float getAngle() {
        return angle;
    }


    /**
     * Set the axis angle, in radians.<br>
     * An axis angle is a rotation angle about the vector (x,y,z).
     *
     * @param angle The angle to setAt, in radians.
     * @since vecmath 1.5
     */
    public final void setAngle(float angle) {
        this.angle = angle;
    }


    /**
     * Get value of <i>x</i> coordinate.
     *
     * @return the <i>x</i> coordinate.
     * @since vecmath 1.5
     */
    public final float getX() {
        return x;
    }


    /**
     * Set a new value for <i>x</i> coordinate.
     *
     * @param x the <i>x</i> coordinate.
     * @since vecmath 1.5
     */
    public final void setX(float x) {
        this.x = x;
    }


    /**
     * Get value of <i>y</i> coordinate.
     *
     * @return the <i>y</i> coordinate
     * @since vecmath 1.5
     */
    public final float getY() {
        return y;
    }


    /**
     * Set a new value for <i>y</i> coordinate.
     *
     * @param y the <i>y</i> coordinate.
     * @since vecmath 1.5
     */
    public final void setY(float y) {
        this.y = y;
    }


    /**
     * Get  value of <i>z</i> coordinate.
     *
     * @return the <i>z</i> coordinate.
     * @since vecmath 1.5
     */
    public final float getZ() {
        return z;
    }


    /**
     * Set a new value for <i>z</i> coordinate.
     *
     * @param z the <i>z</i> coordinate.
     * @since vecmath 1.5
     */
    public final void setZ(float z) {
        this.z = z;
    }

}
