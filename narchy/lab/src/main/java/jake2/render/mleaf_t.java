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




package jake2.render;

public class mleaf_t extends mnode_t {

	
	/*
	public int contents; 
	public int visframe; 

	public float minmaxs[] = new float[6]; 

	public mnode_t parent;
	*/

	
	public int cluster;
	public int area;

	
	public int nummarksurfaces;
	
	
	private int markIndex;
	private msurface_t[] markSurfaces;
	
	public void setMarkSurface(int markIndex, msurface_t[] markSurfaces) {
		this.markIndex = markIndex;
		this.markSurfaces = markSurfaces;
	}

	public msurface_t getMarkSurface(int index) {
		assert (index >= 0 && index <= nummarksurfaces) : "mleaf: markSurface bug (index = " + index +"; num = " + nummarksurfaces + ')';
		
		return (index < nummarksurfaces) ? markSurfaces[markIndex + index] : null;
	}

}
