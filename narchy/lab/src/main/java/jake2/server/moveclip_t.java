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



package jake2.server;

import jake2.game.edict_t;
import jake2.game.trace_t;
import jake2.util.Math3D;

class moveclip_t
{
	final float [] boxmins={0,0,0};
	final float [] boxmaxs={0,0,0};
	float [] mins;
    float [] maxs;
	final float [] mins2={0,0,0};
	final float [] maxs2={0,0,0};	
	float [] start;
    float [] end;
	
	trace_t	trace	= new trace_t();
	edict_t	passedict;
	int contentmask;
	
	public void clear() {
		Math3D.VectorClear(boxmins);
		Math3D.VectorClear(boxmaxs);
		Math3D.VectorClear(mins);
		Math3D.VectorClear(maxs);
		Math3D.VectorClear(mins2);
		Math3D.VectorClear(maxs2);
		start = end = null;
		trace.clear();
		passedict = null;
		contentmask = 0;
	}
}
