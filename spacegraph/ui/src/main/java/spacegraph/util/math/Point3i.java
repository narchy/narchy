/*
 * $RCSfile$
 *
 * Copyright 1999-2008 Sun Microsystems, Inc.  All Rights Reserved.
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

import java.io.Serializable;

/**
 * A 3 element point represented by signed integer x,y,z
 * coordinates.
 *
 * @since vecmath 1.2
 */
public class Point3i implements Serializable, Cloneable {

    /**
     * The x coordinate.
     */
    private int x;
    /**
     * The y coordinate.
     */
    private int y;
    /**
     * The z coordinate.
     */
    private int z;

    /**
     * Constructs and initializes a Point3i from the specified
     * x, y, and z coordinates.
     * @param x the x coordinate
     * @param y the y coordinate
     * @param z the z coordinate
     */
    public Point3i(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }


    /**
     * Constructs and initializes a Point3i from the array of length 3.
     * @param t the array of length 3 containing x, y, and z in order.
     */
    public Point3i(int[] t) {
        this.x = t[0];
        this.y = t[1];
        this.z = t[2];
    }


    /**
     * Constructs and initializes a Point3i from the specified Tuple3i.
     * @param t1 the Tuple3i containing the initialization x, y, and z
     * data.
     */
    public Point3i(Point3i t1) {
        this.x = t1.x;
        this.y = t1.y;
        this.z = t1.z;
    }


    /**
     * Constructs and initializes a Point3i to (0,0,0).
     */
    public Point3i() {
        this.x = 0;
        this.y = 0;
        this.z = 0;
    }

    /**
     * Sets the value of this tuple to the specified x, y, and z
     * coordinates.
     * @param x the x coordinate
     * @param y the y coordinate
     * @param z the z coordinate
     */
    public final void set(int x, int y, int z) {
	this.x = x;
	this.y = y;
	this.z = z;
    }

    /**
     * Sets the value of this tuple to the specified coordinates in the
     * array of length 3.
     * @param t the array of length 3 containing x, y, and z in order.
     */
    public final void set(int[] t) {
	this.x = t[0];
	this.y = t[1];
	this.z = t[2];
    }

    /**
     * Sets the value of this tuple to the value of tuple t1.
     * @param t1 the tuple to be copied
     */
    public final void set(Point3i t1) {
	this.x = t1.x;
	this.y = t1.y;
	this.z = t1.z;
    }

    /**
     * Copies the values of this tuple into the array t.
     * @param t is the array
     */
    public final void get(int[] t) {
	t[0] = this.x;
	t[1] = this.y;
	t[2] = this.z;
    }

    /**
     * Copies the values of this tuple into the tuple t.
     * @param t is the target tuple
     */
    public final void get(Point3i t) {
	t.x = this.x;
	t.y = this.y;
	t.z = this.z;
    }

    /**
     * Sets the value of this tuple to the sum of tuples t1 and t2.
     * @param t1 the first tuple
     * @param t2 the second tuple
     */
    public final void add(Point3i t1, Point3i t2) {
	this.x = t1.x + t2.x;
	this.y = t1.y + t2.y;
	this.z = t1.z + t2.z;
    }

    /**
     * Sets the value of this tuple to the sum of itself and t1.
     * @param t1 the other tuple
     */
    public final void add(Point3i t1) {
	this.x += t1.x;
	this.y += t1.y;
	this.z += t1.z;
    }

    /**
     * Sets the value of this tuple to the difference
     * of tuples t1 and t2 (this = t1 - t2).
     * @param t1 the first tuple
     * @param t2 the second tuple
     */
    public final void sub(Point3i t1, Point3i t2) {
	this.x = t1.x - t2.x;
	this.y = t1.y - t2.y;
	this.z = t1.z - t2.z;
    }

    /**
     * Sets the value of this tuple to the difference
     * of itself and t1 (this = this - t1).
     * @param t1 the other tuple
     */
    public final void sub(Point3i t1) {
	this.x -= t1.x;
	this.y -= t1.y;
	this.z -= t1.z;
    }

    /**
     * Sets the value of this tuple to the negation of tuple t1.
     * @param t1 the source tuple
     */
    public final void negate(Point3i t1) {
	this.x = -t1.x;
	this.y = -t1.y;
	this.z = -t1.z;
    }

    /**
     * Negates the value of this tuple in place.
     */
    public final void negate() {
	this.x = -this.x;
	this.y = -this.y;
	this.z = -this.z;
    }

    /**
     * Sets the value of this tuple to the scalar multiplication
     * of tuple t1.
     * @param s the scalar value
     * @param t1 the source tuple
     */
    public final void scale(int s, Point3i t1) {
	this.x = s*t1.x;
	this.y = s*t1.y;
	this.z = s*t1.z;
    }

    /**
     * Sets the value of this tuple to the scalar multiplication
     * of the scale factor with this.
     * @param s the scalar value
     */
    public final void scale(int s) {
	this.x *= s;
	this.y *= s;
	this.z *= s;
    }

    /**
     * Sets the value of this tuple to the scalar multiplication
     * of tuple t1 plus tuple t2 (this = s*t1 + t2).
     * @param s the scalar value
     * @param t1 the tuple to be multipled
     * @param t2 the tuple to be added
     */
    public final void scaleAdd(int s, Point3i t1, Point3i t2) {
	this.x = s*t1.x + t2.x;
	this.y = s*t1.y + t2.y;
	this.z = s*t1.z + t2.z;
    }

    /**
     * Sets the value of this tuple to the scalar multiplication
     * of itself and then adds tuple t1 (this = s*this + t1).
     * @param s the scalar value
     * @param t1 the tuple to be added
     */
    public final void scaleAdd(int s, Point3i t1) {
        this.x = s*this.x + t1.x;
        this.y = s*this.y + t1.y;
        this.z = s*this.z + t1.z;
    }

    /**
     * Returns a string that contains the values of this Tuple3i.
     * The form is (x,y,z).
     * @return the String representation
     */
    public String toString() {
        return "(" + this.x + ", " + this.y + ", " + this.z + ')';
    }

    /**
     * Returns true if the Object t1 is of type Tuple3i and all of the
     * data members of t1 are equal to the corresponding data members in
     * this Tuple3i.
     * @param t1  the object with which the comparison is made
     */
    public boolean equals(Object t1) {
        try {
	    Point3i t2 = (Point3i) t1;
	    return(this.x == t2.x && this.y == t2.y && this.z == t2.z);
        }
        catch (NullPointerException | ClassCastException e2) {
	    return false;
	}
	}

    /**
     * Returns a hash code value based on the data values in this
     * object.  Two different Tuple3i objects with identical data values
     * (i.e., Tuple3i.equals returns true) will return the same hash
     * code value.  Two objects with different data members may return the
     * same hash value, although this is not likely.
     * @return the integer hash code value
     */
    public int hashCode() {
	long bits = 1L;
	bits = 31L * bits + x;
	bits = 31L * bits + y;
	bits = 31L * bits + z;
	return (int) (bits ^ (bits >> 32));
    }

    /**
     *  Clamps the tuple parameter to the range [low, high] and
     *  places the values into this tuple.
     *  @param min   the lowest value in the tuple after clamping
     *  @param max  the highest value in the tuple after clamping
     *  @param t   the source tuple, which will not be modified
     */
    public final void clamp(int min, int max, Point3i t) {
		x = t.x > max ? max : Math.max(t.x, min);

		y = t.y > max ? max : Math.max(t.y, min);

		z = t.z > max ? max : Math.max(t.z, min);
    }

    /**
     *  Clamps the minimum value of the tuple parameter to the min
     *  parameter and places the values into this tuple.
     *  @param min   the lowest value in the tuple after clamping
     *  @param t   the source tuple, which will not be modified
     */
    public final void clampMin(int min, Point3i t) {
		x = Math.max(t.x, min);

		y = Math.max(t.y, min);

		z = Math.max(t.z, min);
    }

    /**
     *  Clamps the maximum value of the tuple parameter to the max
     *  parameter and places the values into this tuple.
     *  @param max   the highest value in the tuple after clamping
     *  @param t   the source tuple, which will not be modified
     */
    public final void clampMax(int max, Point3i t) {
		x = Math.min(t.x, max);

		y = Math.min(t.y, max);

		z = Math.min(t.z, max);
    }

    /**
     *  Sets each component of the tuple parameter to its absolute
     *  value and places the modified values into this tuple.
     *  @param t   the source tuple, which will not be modified
     */
    public final void absolute(Point3i t) {
	x = Math.abs(t.x);
	y = Math.abs(t.y);
	z = Math.abs(t.z);
    }

    /**
     *  Clamps this tuple to the range [low, high].
     *  @param min  the lowest value in this tuple after clamping
     *  @param max  the highest value in this tuple after clamping
     */
    public final void clamp(int min, int max) {
	if( x > max ) {
	    x = max;
        } else if( x < min ) {
	    x = min;
        }

        if( y > max ) {
	    y = max;
        } else if( y < min ) {
	    y = min;
        }

        if( z > max ) {
	    z = max;
        } else if( z < min ) {
	    z = min;
        }
    }

    /**
     *  Clamps the minimum value of this tuple to the min parameter.
     *  @param min   the lowest value in this tuple after clamping
     */
    public final void clampMin(int min) {
	if (x < min)
	    x=min;

	if (y < min)
	    y = min;

	if (z < min)
	    z = min;
    }

    /**
     *  Clamps the maximum value of this tuple to the max parameter.
     *  @param max   the highest value in the tuple after clamping
     */
    public final void clampMax(int max) {
	if (x > max)
	    x = max;

	if (y > max)
	    y = max;

	if (z > max)
	    z = max;
    }

    /**
     *  Sets each component of this tuple to its absolute value.
     */
    public final void absolute() {
	x = Math.abs(x);
	y = Math.abs(y);
	z = Math.abs(z);
    }

    /**
     * Creates a new object of the same class as this object.
     *
     * @return a clone of this instance.
     * @exception OutOfMemoryError if there is not enough memory.
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
	 * Get the <i>x</i> coordinate.
	 *
	 * @return  the <i>x</i> coordinate.
	 *
	 * @since vecmath 1.5
	 */
	public final int getX() {
		return x;
	}

    /**
     * Set the <i>x</i> coordinate.
     *
     * @param x  value to <i>x</i> coordinate.
     *
     * @since vecmath 1.5
     */
    public final void setX(int x) {
        this.x = x;
    }

    /**
     * Get the <i>y</i> coordinate.
     *
     * @return  the <i>y</i> coordinate.
     *
     * @since vecmath 1.5
     */
    public final int getY() {
        return y;
    }

    /**
     * Set the <i>y</i> coordinate.
     *
     * @param y value to <i>y</i> coordinate.
     *
     * @since vecmath 1.5
     */
    public final void setY(int y) {
        this.y = y;
    }

    /**
     * Get the <i>z</i> coordinate.
     *
     * @return the <i>z</i> coordinate.
     * @since vecmath 1.5
     */
    public final int getZ() {
        return z;
    }

    /**
     * Set the <i>z</i> coordinate.
     *
     * @param z value to <i>z</i> coordinate.
     *
     * @since vecmath 1.5
     */
    public final void setZ(int z) {
        this.z = z;
    }
}
