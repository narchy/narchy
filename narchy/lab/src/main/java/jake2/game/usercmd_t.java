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



package jake2.game;


public class usercmd_t implements Cloneable {
	public byte msec;
	public byte buttons;
	public final short[] angles= new short[3];
	public short forwardmove;
    public short sidemove;
    public short upmove;
	public byte impulse; 
	public byte lightlevel; 

	public void clear() {
		forwardmove= sidemove= upmove= msec= buttons= impulse= lightlevel= 0;
		angles[0] = angles[1] = angles[2] = 0;
	}

	public usercmd_t() {
	}

	public usercmd_t(usercmd_t from) {
		msec= from.msec;
		buttons= from.buttons;
		angles[0]= from.angles[0];
		angles[1]= from.angles[1];
		angles[2]= from.angles[2];
		forwardmove= from.forwardmove;
		sidemove= from.sidemove;
		upmove= from.upmove;
		impulse= from.impulse;
		lightlevel= from.lightlevel;
	}

	public usercmd_t set(usercmd_t from) {
		msec= from.msec;
		buttons= from.buttons;
		angles[0]= from.angles[0];
		angles[1]= from.angles[1];
		angles[2]= from.angles[2];
		forwardmove= from.forwardmove;
		sidemove= from.sidemove;
		upmove= from.upmove;
		impulse= from.impulse;
		lightlevel= from.lightlevel;

		return this;
	}
}