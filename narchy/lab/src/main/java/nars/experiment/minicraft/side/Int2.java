/*
 * Copyright 2012 Jonathan Leahey
 *
 * This file is part of Minicraft
 *
 * Minicraft is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * Minicraft is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Minicraft. If not, see http:
 */

package nars.experiment.minicraft.side;

import java.io.Serializable;

/** TODO replace with something else */
public final class Int2 implements Serializable {

    public int x;
    public int y;

    public Int2(int x, int y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Int2 compare) {
            return x == compare.x && y == compare.y;
        }
        return false;
    }

    @Override
    public String toString() {
        return "{" + x + ", " + y + '}';
    }
}