/*
 * vidmode_t.java
 * Copyright (C) 2003
 *
 * $Id: vidmode_t.java,v 1.1 2004-07-07 19:58:52 hzi Exp $
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
package jake2.client;

/**
 * vidmode_t
 * 
 * @author cwei
 */
class vidmode_t {
    int width;
    int height;

    vidmode_t (String description, int width, int height, int mode) {
        this.width = width;
		this.height = height;
    }
}
