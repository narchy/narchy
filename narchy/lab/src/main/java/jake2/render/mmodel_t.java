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

public class mmodel_t
{
	public final float[] mins = { 0, 0, 0 };
	public final float[] maxs = { 0, 0, 0 };
	public final float[] origin = { 0, 0, 0 }; 
	public float radius;
	public int headnode;
	public static final int visleafs = 0;
	public int firstface;
    public int numfaces;
}
