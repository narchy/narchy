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

public class refdef_t {
	public int		x;
    public int y;
    public int width;
    public int height;
	public float		fov_x;
    public float fov_y;
    public final float[] vieworg = {0, 0, 0};
    public final float[] viewangles = {0, 0, 0};
    public final float[] blend = {0, 0, 0, 0};
	public float		time;				
	public int		rdflags;

    public byte[] areabits;

    public lightstyle_t[] lightstyles;

	public int		num_entities;
    public entity_t[] entities;

	public int		num_dlights;
    public dlight_t[] dlights;

	public int		num_particles;
	
}
