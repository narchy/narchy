/*
 * Vargs.java
 * Copyright (C) 2003
 *
 * $Id: Vargs.java,v 1.1 2004-07-07 19:59:56 hzi Exp $
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
package jake2.util;

import java.util.Vector;

/**
 * Vargs is a helper class to encapsulate printf arguments.
 * 
 * @author cwei
 */
public class Vargs {

	
	private static final int SIZE = 5;

	private Vector v;

	public Vargs() {
		this(SIZE);
	}

	public Vargs(int initialSize) {
		if (v != null)
			v.clear(); 
		v = new Vector(initialSize);
	}

	public Vargs add(boolean value) {
		v.add(value);
		return this;
	}

	public Vargs add(byte value) {
		v.add(value);
		return this;
	}

	public Vargs add(char value) {
		v.add(value);
		return this;
	}

	public Vargs add(short value) {
		v.add(value);
		return this;
	}

	public Vargs add(int value) {
		v.add(value);
		return this;
	}

	public Vargs add(long value) {
		v.add(value);
		return this;
	}

	public Vargs add(float value) {
		v.add(value);
		return this;
	}

	public Vargs add(double value) {
		v.add(value);
		return this;
	}

	public Vargs add(String value) {
		v.add(value);
		return this;
	}

	public Vargs add(Object value) {
		v.add(value);
		return this;
	}

	public Vargs clear() {
		v.clear();
		return this;
	}

	public Vector toVector() {
		
		
		
		return (Vector) v.clone();
	}

	public Object[] toArray() {
		return v.toArray();
	}

	public int size() {
		return v.size();
	}
}
