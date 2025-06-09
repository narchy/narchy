/*
 * SV.java
 * Copyright (C) 2003
 * 
 * $Id: SV.java,v 1.12 2006-01-21 21:53:32 salomo Exp $
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
package jake2.server;

import jake2.Defines;
import jake2.Globals;
import jake2.client.M;
import jake2.game.*;
import jake2.qcommon.Com;
import jake2.util.Lib;
import jake2.util.Math3D;

import java.util.stream.IntStream;

/**
 * SV
 */
public final class SV {

    
    private static edict_t[] SV_TestEntityPosition(edict_t ent) {
        int mask;

        if (ent.clipmask != 0)
            mask = ent.clipmask;
        else
            mask = Defines.MASK_SOLID;

        trace_t trace = game_import_t.trace(ent.s.origin, ent.mins, ent.maxs,
                ent.s.origin, ent, mask);

        if (trace.startsolid)
            return GameBase.g_edicts;

        return null;
    }

    
    private static void SV_CheckVelocity(edict_t ent) {


        for (int i = 0; i < 3; i++) {
            if (ent.velocity[i] > GameBase.sv_maxvelocity.value)
                ent.velocity[i] = GameBase.sv_maxvelocity.value;
            else if (ent.velocity[i] < -GameBase.sv_maxvelocity.value)
                ent.velocity[i] = -GameBase.sv_maxvelocity.value;
        }
    }

    /**
     * Runs thinking code for this frame if necessary.
     */
    private static boolean SV_RunThink(edict_t ent) {

        float thinktime = ent.nextthink;
        if (thinktime <= 0)
            return true;
        if (thinktime > GameBase.level.time + 0.001)
            return true;

        ent.nextthink = 0;

        if (ent.think == null)
            Com.Error(Defines.ERR_FATAL, "NULL ent.think");

        ent.think.think(ent);

        return false;
    }

    /**
     * Two entities have touched, so run their touch functions.
     */
    private static void SV_Impact(edict_t e1, trace_t trace) {

        edict_t e2 = trace.ent;

        if (e1.touch != null && e1.solid != Defines.SOLID_NOT)
            e1.touch.touch(e1, e2, trace.plane, trace.surface);

        if (e2.touch != null && e2.solid != Defines.SOLID_NOT)
            e2.touch.touch(e2, e1, GameBase.dummyplane, null);
    }
    
    /**
     * SV_FlyMove
     * 
     * The basic solid body movement clip that slides along multiple planes
     * Returns the clipflags if the velocity was modified (hit something solid)
     * 1 = floor 2 = wall / step 4 = dead stop
     */
    public static final int MAX_CLIP_PLANES = 5;

    private static int SV_FlyMove(edict_t ent, float time, int mask) {
        float[] original_velocity = { 0.0f, 0.0f, 0.0f };

        Math3D.VectorCopy(ent.velocity, original_velocity);
        float[] primal_velocity = {0.0f, 0.0f, 0.0f};
        Math3D.VectorCopy(ent.velocity, primal_velocity);

        ent.groundentity = null;
        float time_left = time;
        int numplanes = 0;
        int blocked = 0;
        int numbumps = 4;
        float[] end = {0.0f, 0.0f, 0.0f};
        float[] new_velocity = {0.0f, 0.0f, 0.0f};
        float[][] planes = new float[MAX_CLIP_PLANES][3];
        float[] dir = {0.0f, 0.0f, 0.0f};
        for (int bumpcount = 0; bumpcount < numbumps; bumpcount++) {
            int i;
            for (i = 0; i < 3; i++)
                end[i] = ent.s.origin[i] + time_left * ent.velocity[i];

            trace_t trace = game_import_t.trace(ent.s.origin, ent.mins, ent.maxs, end,
                    ent, mask);

            if (trace.allsolid) { 
                Math3D.VectorCopy(Globals.vec3_origin, ent.velocity);
                return 3;
            }

            if (trace.fraction > 0) { 
                Math3D.VectorCopy(trace.endpos, ent.s.origin);
                Math3D.VectorCopy(ent.velocity, original_velocity);
                numplanes = 0;
            }

            if (trace.fraction == 1)
                break;

            edict_t hit = trace.ent;

            if (trace.plane.normal[2] > 0.7) {
                blocked |= 1; 
                if (hit.solid == Defines.SOLID_BSP) {
                    ent.groundentity = hit;
                    ent.groundentity_linkcount = hit.linkcount;
                }
            }
            if (trace.plane.normal[2] == 0.0f) {
                blocked |= 2; 
            }

            
            
            
            SV_Impact(ent, trace);
            if (!ent.inuse)
                break; 

            time_left -= time_left * trace.fraction;

            
            if (numplanes >= MAX_CLIP_PLANES) { 
                                                         
                Math3D.VectorCopy(Globals.vec3_origin, ent.velocity);
                return 3;
            }

            Math3D.VectorCopy(trace.plane.normal, planes[numplanes]);
            numplanes++;

            
            
            
            for (i = 0; i < numplanes; i++) {
                GameBase.ClipVelocity(original_velocity, planes[i],
                        new_velocity, 1);

                int j;
                for (j = 0; j < numplanes; j++)
                    if ((j != i)
                            && !Math3D.VectorEquals(planes[i], planes[j])) {
                        if (Math3D.DotProduct(new_velocity, planes[j]) < 0)
                            break; 
                    }
                if (j == numplanes)
                    break;
            }

            if (i != numplanes) { 
                Math3D.VectorCopy(new_velocity, ent.velocity);
            } else { 
                if (numplanes != 2) {
                    
                    
                    Math3D.VectorCopy(Globals.vec3_origin, ent.velocity);
                    return 7;
                }
                Math3D.CrossProduct(planes[0], planes[1], dir);
                float d = Math3D.DotProduct(dir, ent.velocity);
                Math3D.VectorScale(dir, d, ent.velocity);
            }

            
            
            
            
            if (Math3D.DotProduct(ent.velocity, primal_velocity) <= 0) {
                Math3D.VectorCopy(Globals.vec3_origin, ent.velocity);
                return blocked;
            }
        }

        return blocked;
    }

    /**
     * SV_AddGravity.
     */
    private static void SV_AddGravity(edict_t ent) {
        ent.velocity[2] -= ent.gravity * GameBase.sv_gravity.value
                * Defines.FRAMETIME;
    }

    /**
     * Does not change the entities velocity at all
     */
    private static trace_t SV_PushEntity(edict_t ent, float[] push) {
        float[] start = { 0, 0, 0 };

        Math3D.VectorCopy(ent.s.origin, start);
        float[] end = {0, 0, 0};
        Math3D.VectorAdd(start, push, end);

        
        
        boolean retry;

        trace_t trace;
        do {
            int mask = ent.clipmask != 0 ? ent.clipmask : Defines.MASK_SOLID;

            trace = game_import_t
                    .trace(start, ent.mins, ent.maxs, end, ent, mask);

            Math3D.VectorCopy(trace.endpos, ent.s.origin);
            game_import_t.linkentity(ent);

            retry = false;
            if (trace.fraction != 1.0) {
                SV_Impact(ent, trace);

                
                if (!trace.ent.inuse && ent.inuse) {
                    
                    Math3D.VectorCopy(start, ent.s.origin);
                    game_import_t.linkentity(ent);
                    
                    retry = true;
                }
            }
        } while (retry);

        if (ent.inuse)
            GameBase.G_TouchTriggers(ent);

        return trace;
    }

    /**
     * Objects need to be moved back on a failed push, otherwise riders would
     * continue to slide.
     */
    private static boolean SV_Push(edict_t pusher, float[] move, float[] amove) {
        int i;


        for (i = 0; i < 3; i++) {
            float temp = move[i] * 8.0f;
            if (temp > 0.0)
                temp += 0.5;
            else
                temp -= 0.5;
            move[i] = 0.125f * (int) temp;
        }


        float[] maxs = {0, 0, 0};
        float[] mins = {0, 0, 0};
        for (i = 0; i < 3; i++) {
            mins[i] = pusher.absmin[i] + move[i];
            maxs[i] = pusher.absmax[i] + move[i];
        }


        float[] org = {0, 0, 0};
        Math3D.VectorSubtract(Globals.vec3_origin, amove, org);
        float[] up = {0, 0, 0};
        float[] right = {0, 0, 0};
        float[] forward = {0, 0, 0};
        Math3D.AngleVectors(org, forward, right, up);

        
        GameBase.pushed[GameBase.pushed_p].ent = pusher;
        Math3D.VectorCopy(pusher.s.origin,
                GameBase.pushed[GameBase.pushed_p].origin);
        Math3D.VectorCopy(pusher.s.angles,
                GameBase.pushed[GameBase.pushed_p].angles);

        if (pusher.client != null)
            GameBase.pushed[GameBase.pushed_p].deltayaw = pusher.client.ps.pmove.delta_angles[Defines.YAW];

        GameBase.pushed_p++;

        
        Math3D.VectorAdd(pusher.s.origin, move, pusher.s.origin);
        Math3D.VectorAdd(pusher.s.angles, amove, pusher.s.angles);
        game_import_t.linkentity(pusher);


        float[] move2 = {0, 0, 0};
        float[] org2 = {0, 0, 0};
        for (int e = 1; e < GameBase.num_edicts; e++) {
            edict_t check = GameBase.g_edicts[e];
            if (!check.inuse)
                continue;
            if (check.movetype == Defines.MOVETYPE_PUSH
                    || check.movetype == Defines.MOVETYPE_STOP
                    || check.movetype == Defines.MOVETYPE_NONE
                    || check.movetype == Defines.MOVETYPE_NOCLIP)
                continue;

            if (check.area.prev == null)
                continue; 

            
            
            if (check.groundentity != pusher) {
                
                if (check.absmin[0] >= maxs[0] || check.absmin[1] >= maxs[1]
                        || check.absmin[2] >= maxs[2]
                        || check.absmax[0] <= mins[0]
                        || check.absmax[1] <= mins[1]
                        || check.absmax[2] <= mins[2])
                    continue;

                
                if (SV_TestEntityPosition(check) == null)
                    continue;
            }

            if ((pusher.movetype == Defines.MOVETYPE_PUSH)
                    || (check.groundentity == pusher)) {
                
                GameBase.pushed[GameBase.pushed_p].ent = check;
                Math3D.VectorCopy(check.s.origin,
                        GameBase.pushed[GameBase.pushed_p].origin);
                Math3D.VectorCopy(check.s.angles,
                        GameBase.pushed[GameBase.pushed_p].angles);
                GameBase.pushed_p++;

                
                Math3D.VectorAdd(check.s.origin, move, check.s.origin);
                if (check.client != null) { 
                    check.client.ps.pmove.delta_angles[Defines.YAW] += amove[Defines.YAW];
                }

                
                Math3D.VectorSubtract(check.s.origin, pusher.s.origin, org);
                org2[0] = Math3D.DotProduct(org, forward);
                org2[1] = -Math3D.DotProduct(org, right);
                org2[2] = Math3D.DotProduct(org, up);
                Math3D.VectorSubtract(org2, org, move2);
                Math3D.VectorAdd(check.s.origin, move2, check.s.origin);

                
                if (check.groundentity != pusher)
                    check.groundentity = null;

                edict_t[] block = SV_TestEntityPosition(check);
                if (block == null) { 
                    game_import_t.linkentity(check);
                    
                    continue;
                }

                
                
                
                Math3D.VectorSubtract(check.s.origin, move, check.s.origin);
                block = SV_TestEntityPosition(check);

                if (block == null) {
                    GameBase.pushed_p--;
                    continue;
                }
            }

            
            GameBase.obstacle = check;

            
            
            
            for (int ip = GameBase.pushed_p - 1; ip >= 0; ip--) {
                pushed_t p = GameBase.pushed[ip];
                Math3D.VectorCopy(p.origin, p.ent.s.origin);
                Math3D.VectorCopy(p.angles, p.ent.s.angles);
                if (p.ent.client != null) {
                    p.ent.client.ps.pmove.delta_angles[Defines.YAW] = (short) p.deltayaw;
                }
                game_import_t.linkentity(p.ent);
            }
            return false;
        }

        
        
        for (int ip = GameBase.pushed_p - 1; ip >= 0; ip--)
            GameBase.G_TouchTriggers(GameBase.pushed[ip].ent);

        return true;
    }

    /**
     * 
     * Bmodel objects don't interact with each other, but push all box objects.
     */
    public static void SV_Physics_Pusher(edict_t ent) {


        if ((ent.flags & Defines.FL_TEAMSLAVE) != 0)
            return;

        
        
        
        
        GameBase.pushed_p = 0;
        edict_t part;
        float[] amove = {0, 0, 0};
        float[] move = {0, 0, 0};
        for (part = ent; part != null; part = part.teamchain) {
            if (part.velocity[0] != 0 || part.velocity[1] != 0
                    || part.velocity[2] != 0 || part.avelocity[0] != 0
                    || part.avelocity[1] != 0 || part.avelocity[2] != 0) { 
                                                                           
                                                                           
                Math3D.VectorScale(part.velocity, Defines.FRAMETIME, move);
                Math3D.VectorScale(part.avelocity, Defines.FRAMETIME, amove);

                if (!SV_Push(part, move, amove))
                    break; 
            }
        }
        if (GameBase.pushed_p > Defines.MAX_EDICTS)
            SV_GAME.PF_error(Defines.ERR_FATAL,
                    "pushed_p > &pushed[MAX_EDICTS], memory corrupted");

        if (part != null) {
            
            for (edict_t mv = ent; mv != null; mv = mv.teamchain) {
                if (mv.nextthink > 0)
                    mv.nextthink += Defines.FRAMETIME;
            }

            
            
            if (part.blocked != null)
                part.blocked.blocked(part, GameBase.obstacle);
        } else { 
            for (part = ent; part != null; part = part.teamchain) {
                SV_RunThink(part);
            }
        }
    }


    /**
     * Non moving objects can only think.
     */
    public static void SV_Physics_None(edict_t ent) {
        
        SV_RunThink(ent);
    }

    /**
     * A moving object that doesn't obey physics.
     */
    public static void SV_Physics_Noclip(edict_t ent) {
        
        if (!SV_RunThink(ent))
            return;

        Math3D.VectorMA(ent.s.angles, Defines.FRAMETIME, ent.avelocity,
                ent.s.angles);
        Math3D.VectorMA(ent.s.origin, Defines.FRAMETIME, ent.velocity,
                ent.s.origin);

        game_import_t.linkentity(ent);
    }

    /**
     * Toss, bounce, and fly movement. When onground, do nothing.
     */
    public static void SV_Physics_Toss(edict_t ent) {


        SV_RunThink(ent);

        
        if ((ent.flags & Defines.FL_TEAMSLAVE) != 0)
            return;

        if (ent.velocity[2] > 0)
            ent.groundentity = null;

        
        if (ent.groundentity != null)
            if (!ent.groundentity.inuse)
                ent.groundentity = null;

        
        if (ent.groundentity != null)
            return;

        float[] old_origin = {0, 0, 0};
        Math3D.VectorCopy(ent.s.origin, old_origin);

        SV_CheckVelocity(ent);

        
        if (ent.movetype != Defines.MOVETYPE_FLY
                && ent.movetype != Defines.MOVETYPE_FLYMISSILE)
            SV_AddGravity(ent);

        
        Math3D.VectorMA(ent.s.angles, Defines.FRAMETIME, ent.avelocity,
                ent.s.angles);


        float[] move = {0, 0, 0};
        Math3D.VectorScale(ent.velocity, Defines.FRAMETIME, move);
        trace_t trace = SV_PushEntity(ent, move);
        if (!ent.inuse)
            return;

        if (trace.fraction < 1) {
            float backoff;
            if (ent.movetype == Defines.MOVETYPE_BOUNCE)
                backoff = 1.5f;
            else
                backoff = 1;

            GameBase.ClipVelocity(ent.velocity, trace.plane.normal,
                    ent.velocity, backoff);

            
            if (trace.plane.normal[2] > 0.7) {
                if (ent.velocity[2] < 60
                        || ent.movetype != Defines.MOVETYPE_BOUNCE) {
                    ent.groundentity = trace.ent;
                    ent.groundentity_linkcount = trace.ent.linkcount;
                    Math3D.VectorCopy(Globals.vec3_origin, ent.velocity);
                    Math3D.VectorCopy(Globals.vec3_origin, ent.avelocity);
                }
            }

            
            
        }


        boolean wasinwater = (ent.watertype & Defines.MASK_WATER) != 0;
        ent.watertype = GameBase.gi.pointcontents.pointcontents(ent.s.origin);
        boolean isinwater = (ent.watertype & Defines.MASK_WATER) != 0;

        if (isinwater)
            ent.waterlevel = 1;
        else
            ent.waterlevel = 0;

        if (!wasinwater && isinwater)
            game_import_t.positioned_sound(old_origin, ent, Defines.CHAN_AUTO,
                    game_import_t.soundindex("misc/h2ohit1.wav"), 1, 1, 0);
        else if (wasinwater && !isinwater)
            game_import_t.positioned_sound(ent.s.origin, ent, Defines.CHAN_AUTO,
                    game_import_t.soundindex("misc/h2ohit1.wav"), 1, 1, 0);

        
        for (edict_t slave = ent.teamchain; slave != null; slave = slave.teamchain) {
            Math3D.VectorCopy(ent.s.origin, slave.s.origin);
            game_import_t.linkentity(slave);
        }
    }


    
    private static void SV_AddRotationalFriction(edict_t ent) {

        Math3D.VectorMA(ent.s.angles, Defines.FRAMETIME, ent.avelocity,
                ent.s.angles);
        float adjustment = Defines.FRAMETIME * Defines.sv_stopspeed
                * Defines.sv_friction;
        for (int n = 0; n < 3; n++) {
            if (ent.avelocity[n] > 0) {
                ent.avelocity[n] -= adjustment;
                if (ent.avelocity[n] < 0)
                    ent.avelocity[n] = 0;
            } else {
                ent.avelocity[n] += adjustment;
                if (ent.avelocity[n] > 0)
                    ent.avelocity[n] = 0;
            }
        }
    }
    
    /**
     * Monsters freefall when they don't have a ground entity, otherwise all
     * movement is done with discrete steps.
     * 
     * This is also used for objects that have become still on the ground, but
     * will fall if the floor is pulled out from under them. FIXME: is this
     * true?
     */

    public static void SV_Physics_Step(edict_t ent) {


        if (ent.groundentity == null)
            M.M_CheckGround(ent);

        edict_t groundentity = ent.groundentity;

        SV_CheckVelocity(ent);

        boolean wasonground = groundentity != null;

        if (IntStream.of(0, 1, 2).anyMatch(v -> ent.avelocity[v] != 0)) {
            SV_AddRotationalFriction(ent);
        }


        boolean hitsound = false;
        if (!wasonground)
            if (0 == (ent.flags & Defines.FL_FLY))
                if (!((ent.flags & Defines.FL_SWIM) != 0 && (ent.waterlevel > 2))) {
                    if (ent.velocity[2] < GameBase.sv_gravity.value * -0.1)
                        hitsound = true;
                    if (ent.waterlevel == 0)
                        SV_AddGravity(ent);
                }


        float friction;
        float control;
        float newspeed;
        float speed;
        if ((ent.flags & Defines.FL_FLY) != 0 && (ent.velocity[2] != 0)) {
            speed = Math.abs(ent.velocity[2]);
            control = speed < Defines.sv_stopspeed ? Defines.sv_stopspeed
                    : speed;
            friction = Defines.sv_friction / 3f;
            newspeed = speed - (Defines.FRAMETIME * control * friction);
            if (newspeed < 0)
                newspeed = 0;
            newspeed /= speed;
            ent.velocity[2] *= newspeed;
        }

        
        if ((ent.flags & Defines.FL_SWIM) != 0 && (ent.velocity[2] != 0)) {
            speed = Math.abs(ent.velocity[2]);
            control = speed < Defines.sv_stopspeed ? Defines.sv_stopspeed
                    : speed;
            newspeed = speed
                    - (Defines.FRAMETIME * control * Defines.sv_waterfriction * ent.waterlevel);
            if (newspeed < 0)
                newspeed = 0;
            newspeed /= speed;
            ent.velocity[2] *= newspeed;
        }

        boolean b = IntStream.of(2, 1, 0).anyMatch(i -> ent.velocity[i] != 0);
        if (b) {
            
            
            if ((wasonground)
                    || 0 != (ent.flags & (Defines.FL_SWIM | Defines.FL_FLY)))
                if (!(ent.health <= 0.0 && !M.M_CheckBottom(ent))) {
                    float[] vel = ent.velocity;
                    speed = (float) Math
                            .sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
                    if (speed != 0) {
                        friction = Defines.sv_friction;

                        control = speed < Defines.sv_stopspeed ? Defines.sv_stopspeed
                                : speed;
                        newspeed = speed - Defines.FRAMETIME * control
                                * friction;

                        if (newspeed < 0)
                            newspeed = 0;
                        newspeed /= speed;

                        vel[0] *= newspeed;
                        vel[1] *= newspeed;
                    }
                }

            int mask;
            if ((ent.svflags & Defines.SVF_MONSTER) != 0)
                mask = Defines.MASK_MONSTERSOLID;
            else
                mask = Defines.MASK_SOLID;

            SV_FlyMove(ent, Defines.FRAMETIME, mask);

            game_import_t.linkentity(ent);
            GameBase.G_TouchTriggers(ent);
            if (!ent.inuse)
                return;

            if (ent.groundentity != null)
                if (!wasonground)
                    if (hitsound)
                        game_import_t.sound(ent, 0,
                        		game_import_t.soundindex("world/land.wav"), 1, 1, 0);
        }

        
        SV_RunThink(ent);
    }

    /**
     * Called by monster program code. The move will be adjusted for slopes and
     * stairs, but if the move isn't possible, no move is done, false is
     * returned, and pr_global_struct.trace_normal is set to the normal of the
     * blocking wall.
     */
    
    
    
    public static boolean SV_movestep(edict_t ent, float[] move, boolean relink) {
        float[] oldorg = { 0, 0, 0 };


        Math3D.VectorCopy(ent.s.origin, oldorg);
        float[] neworg = {0, 0, 0};
        Math3D.VectorAdd(ent.s.origin, move, neworg);


        int contents;
        float[] test = {0, 0, 0};
        trace_t trace;
        if ((ent.flags & (Defines.FL_SWIM | Defines.FL_FLY)) != 0) {
            
            for (int i = 0; i < 2; i++) {
                Math3D.VectorAdd(ent.s.origin, move, neworg);
                if (i == 0 && ent.enemy != null) {
                    if (ent.goalentity == null)
                        ent.goalentity = ent.enemy;
                    float dz = ent.s.origin[2] - ent.goalentity.s.origin[2];
                    if (ent.goalentity.client != null) {
                        if (dz > 40)
                            neworg[2] -= 8;
                        if (!((ent.flags & Defines.FL_SWIM) != 0 && (ent.waterlevel < 2)))
                            if (dz < 30)
                                neworg[2] += 8;
                    } else {
                        if (dz > 8)
                            neworg[2] -= 8;
                        else if (dz > 0)
                            neworg[2] -= dz;
                        else if (dz < -8)
                            neworg[2] += 8;
                        else
                            neworg[2] += dz;
                    }
                }
                trace = game_import_t.trace(ent.s.origin, ent.mins, ent.maxs,
                        neworg, ent, Defines.MASK_MONSTERSOLID);

                
                if ((ent.flags & Defines.FL_FLY) != 0) {
                    if (ent.waterlevel == 0) {
                        test[0] = trace.endpos[0];
                        test[1] = trace.endpos[1];
                        test[2] = trace.endpos[2] + ent.mins[2] + 1;
                        contents = GameBase.gi.pointcontents.pointcontents(test);
                        if ((contents & Defines.MASK_WATER) != 0)
                            return false;
                    }
                }

                
                if ((ent.flags & Defines.FL_SWIM) != 0) {
                    if (ent.waterlevel < 2) {
                        test[0] = trace.endpos[0];
                        test[1] = trace.endpos[1];
                        test[2] = trace.endpos[2] + ent.mins[2] + 1;
                        contents = GameBase.gi.pointcontents.pointcontents(test);
                        if ((contents & Defines.MASK_WATER) == 0)
                            return false;
                    }
                }

                if (trace.fraction == 1) {
                    Math3D.VectorCopy(trace.endpos, ent.s.origin);
                    if (relink) {
                        game_import_t.linkentity(ent);
                        GameBase.G_TouchTriggers(ent);
                    }
                    return true;
                }

                if (ent.enemy == null)
                    break;
            }

            return false;
        }


        float stepsize;
        if ((ent.monsterinfo.aiflags & Defines.AI_NOSTEP) == 0)
            stepsize = GameBase.STEPSIZE;
        else
            stepsize = 1;

        neworg[2] += stepsize;
        float[] end = {0, 0, 0};
        Math3D.VectorCopy(neworg, end);
        end[2] -= stepsize * 2;

        trace = game_import_t.trace(neworg, ent.mins, ent.maxs, end, ent,
                Defines.MASK_MONSTERSOLID);

        if (trace.allsolid)
            return false;

        if (trace.startsolid) {
            neworg[2] -= stepsize;
            trace = game_import_t.trace(neworg, ent.mins, ent.maxs, end, ent,
                    Defines.MASK_MONSTERSOLID);
            if (trace.allsolid || trace.startsolid)
                return false;
        }

        
        if (ent.waterlevel == 0) {
            test[0] = trace.endpos[0];
            test[1] = trace.endpos[1];
            test[2] = trace.endpos[2] + ent.mins[2] + 1;
            contents = GameBase.gi.pointcontents.pointcontents(test);

            if ((contents & Defines.MASK_WATER) != 0)
                return false;
        }

        if (trace.fraction == 1) {
            
            if ((ent.flags & Defines.FL_PARTIALGROUND) != 0) {
                Math3D.VectorAdd(ent.s.origin, move, ent.s.origin);
                if (relink) {
                    game_import_t.linkentity(ent);
                    GameBase.G_TouchTriggers(ent);
                }
                ent.groundentity = null;
                return true;
            }

            return false; 
        }

        
        Math3D.VectorCopy(trace.endpos, ent.s.origin);

        if (!M.M_CheckBottom(ent)) {
            if ((ent.flags & Defines.FL_PARTIALGROUND) != 0) {
                
                
                if (relink) {
                    game_import_t.linkentity(ent);
                    GameBase.G_TouchTriggers(ent);
                }
                return true;
            }
            Math3D.VectorCopy(oldorg, ent.s.origin);
            return false;
        }

        if ((ent.flags & Defines.FL_PARTIALGROUND) != 0) {
            ent.flags &= ~Defines.FL_PARTIALGROUND;
        }
        ent.groundentity = trace.ent;
        ent.groundentity_linkcount = trace.ent.linkcount;

        
        if (relink) {
            game_import_t.linkentity(ent);
            GameBase.G_TouchTriggers(ent);
        }
        return true;
    }

    /** 
     * Turns to the movement direction, and walks the current distance if facing
     * it.
     */
    public static boolean SV_StepDirection(edict_t ent, float yaw, float dist) {

        ent.ideal_yaw = yaw;
        M.M_ChangeYaw(ent);

        yaw = (float) (yaw * Math.PI * 2 / 360);
        float[] move = {0, 0, 0};
        move[0] = (float) Math.cos(yaw) * dist;
        move[1] = (float) Math.sin(yaw) * dist;
        move[2] = 0;

        float[] oldorigin = {0, 0, 0};
        Math3D.VectorCopy(ent.s.origin, oldorigin);
        if (SV_movestep(ent, move, false)) {
            float delta = ent.s.angles[Defines.YAW] - ent.ideal_yaw;
            if (delta > 45 && delta < 315) { 
                                             
                Math3D.VectorCopy(oldorigin, ent.s.origin);
            }
            game_import_t.linkentity(ent);
            GameBase.G_TouchTriggers(ent);
            return true;
        }
        game_import_t.linkentity(ent);
        GameBase.G_TouchTriggers(ent);
        return false;
    }

    /**
     * SV_FixCheckBottom
     * 
     */
    private static void SV_FixCheckBottom(edict_t ent) {
        ent.flags |= Defines.FL_PARTIALGROUND;
    }

    public static void SV_NewChaseDir(edict_t actor, edict_t enemy, float dist) {


        if (enemy == null) {
            Com.DPrintf("SV_NewChaseDir without enemy!\n");
            return;
        }
        float olddir = Math3D.anglemod((int) (actor.ideal_yaw / 45) * 45);
        float turnaround = Math3D.anglemod(olddir - 180);

        float deltax = enemy.s.origin[0] - actor.s.origin[0];
        float deltay = enemy.s.origin[1] - actor.s.origin[1];
        float[] d = {0, 0, 0};
        if (deltax > 10)
            d[1] = 0;
        else if (deltax < -10)
            d[1] = 180;
        else
            d[1] = DI_NODIR;
        if (deltay < -10)
            d[2] = 270;
        else if (deltay > 10)
            d[2] = 90;
        else
            d[2] = DI_NODIR;


        float tdir;
        if (d[1] != DI_NODIR && d[2] != DI_NODIR) {
            if (d[1] == 0)
                tdir = d[2] == 90 ? 45 : 315;
            else
                tdir = d[2] == 90 ? 135 : 215;

            if (tdir != turnaround && SV_StepDirection(actor, tdir, dist))
                return;
        }

        
        if (((Lib.rand() & 3) & 1) != 0 || Math.abs(deltay) > Math.abs(deltax)) {
            tdir = d[1];
            d[1] = d[2];
            d[2] = tdir;
        }

        if (d[1] != DI_NODIR && d[1] != turnaround
                && SV_StepDirection(actor, d[1], dist))
            return;

        if (d[2] != DI_NODIR && d[2] != turnaround
                && SV_StepDirection(actor, d[2], dist))
            return;

        /* there is no direct path to the player, so pick another direction */

        if (olddir != DI_NODIR
                && SV_StepDirection(actor, olddir, dist))
            return;

        if ((Lib.rand() & 1) != 0) /* randomly determine direction of search */{
            for (tdir = 0; tdir <= 315; tdir += 45)
                if (tdir != turnaround && SV_StepDirection(actor, tdir, dist))
                    return;
        } else {
            for (tdir = 315; tdir >= 0; tdir -= 45)
                if (tdir != turnaround && SV_StepDirection(actor, tdir, dist))
                    return;
        }

        if (turnaround != DI_NODIR
                && SV_StepDirection(actor, turnaround, dist))
            return;

        actor.ideal_yaw = olddir; 

        
        

        if (!M.M_CheckBottom(actor))
            SV_FixCheckBottom(actor);
    }

    /**
     * SV_CloseEnough - returns true if distance between 2 ents is smaller than
     * given dist.  
     */
    public static boolean SV_CloseEnough(edict_t ent, edict_t goal, float dist) {

        for (int i = 0; i < 3; i++) {
            if (goal.absmin[i] > ent.absmax[i] + dist)
                return false;
            if (goal.absmax[i] < ent.absmin[i] - dist)
                return false;
        }
        return true;
    }

	private static final int DI_NODIR = -1;
}