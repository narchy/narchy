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

import jake2.game.player_state_t;

import java.util.Arrays;

public class frame_t {
		
	private static final int MAX_MAP_AREAS = 256;
		
	boolean		valid;			
	int				serverframe;
	public int servertime;		
	int				deltaframe;
	final byte[]			areabits = new byte [MAX_MAP_AREAS/8];		
	public final player_state_t playerstate = new player_state_t(); 
	public int				num_entities;
	public int				parse_entities;	
		
	public void set(frame_t from) {
		valid = from.valid;
		serverframe = from.serverframe;
		deltaframe = from.deltaframe;
		num_entities = from.num_entities;
		parse_entities = from.parse_entities;
		System.arraycopy(from.areabits, 0, areabits, 0, areabits.length);
		playerstate.set(from.playerstate);
	}
	
	public void reset()
	{
		valid = false;
		serverframe = servertime = deltaframe = 0;
		Arrays.fill(areabits, (byte)0);
		playerstate.clear();
		num_entities = parse_entities = 0;
	}	
}
