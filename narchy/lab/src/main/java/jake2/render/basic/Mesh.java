/*
 * Mesh.java
 * Copyright (C) 2003
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
package jake2.render.basic;

import jake2.Defines;
import jake2.client.VID;
import jake2.client.entity_t;
import jake2.qcommon.qfiles;
import jake2.render.Anorms;
import jake2.render.image_t;
import jake2.util.Lib;
import jake2.util.Math3D;

import java.nio.FloatBuffer;

/**
 * Mesh
 * 
 * @author cwei
 */
abstract class Mesh extends Light {

    

    /*
     * =============================================================
     * 
     * ALIAS MODELS
     * 
     * =============================================================
     */

    static final int NUMVERTEXNORMALS = 162;

    private final float[][] r_avertexnormals = Anorms.VERTEXNORMALS;

    private final float[][] s_lerped = new float[qfiles.MAX_VERTS][4];

    private final float[] shadevector = { 0, 0, 0 };

    private final float[] shadelight = { 0, 0, 0 };

    
    private static final int SHADEDOT_QUANT = 16;

    private final float[][] r_avertexnormal_dots = Anorms.VERTEXNORMAL_DOTS;

    private float[] shadedots = r_avertexnormal_dots[0];

    private void GL_LerpVerts(int nverts, int[] ov, int[] v,
                              float[][] lerp, float[] move, float[] frontv, float[] backv) {

	int ovv, vv;

        
        if ((currententity.flags & (Defines.RF_SHELL_RED
                | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_BLUE
                | Defines.RF_SHELL_DOUBLE | Defines.RF_SHELL_HALF_DAM)) != 0) {
            for (int i = 0; i < nverts; i++ /* , v++, ov++, lerp+=4 */
            ) {
                vv = v[i];
                float[] normal = r_avertexnormals[(vv >>> 24) & 0xFF];
                ovv = ov[i];

                lerp[i][0] = move[0] + (ovv & 0xFF) * backv[0] + (vv & 0xFF)
                        * frontv[0] + normal[0] * Defines.POWERSUIT_SCALE;
                lerp[i][1] = move[1] + ((ovv >>> 8) & 0xFF) * backv[1] + ((vv >>> 8) & 0xFF)
                        * frontv[1] + normal[1] * Defines.POWERSUIT_SCALE;
                lerp[i][2] = move[2] + ((ovv >>> 16) & 0xFF) * backv[2] + ((vv >>> 16) & 0xFF)
                        * frontv[2] + normal[2] * Defines.POWERSUIT_SCALE;
            }
        } else {
            for (int i = 0; i < nverts; i++ /* , v++, ov++, lerp+=4 */
            ) {
                vv = v[i];
                ovv = ov[i];
                lerp[i][0] = move[0] + (ovv & 0xFF) * backv[0] + (vv & 0xFF)
                        * frontv[0];
                lerp[i][1] = move[1] + ((ovv >>> 8) & 0xFF) * backv[1] + ((vv >>> 8) & 0xFF)
                        * frontv[1];
                lerp[i][2] = move[2] + ((ovv >>> 16) & 0xFF) * backv[2] + ((vv >>> 16) & 0xFF)
                        * frontv[2];
            }
        }
    }

    private void GL_LerpVerts(int nverts, int[] ov, int[] v, float[] move,
                              float[] frontv, float[] backv) {
        
	int ovv, vv;
        FloatBuffer lerp = vertexArrayBuf;

        
        if ((currententity.flags & (Defines.RF_SHELL_RED
                | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_BLUE
                | Defines.RF_SHELL_DOUBLE | Defines.RF_SHELL_HALF_DAM)) != 0) {
            int j = 0;
            for (int i = 0; i < nverts; i++ /* , v++, ov++, lerp+=4 */
            ) {
                vv = v[i];
                float[] normal = r_avertexnormals[(v[i] >>> 24) & 0xFF];
                ovv = ov[i];

                lerp.put(j, move[0] + (ovv & 0xFF) * backv[0] + (vv & 0xFF) * frontv[0]
                        + normal[0] * Defines.POWERSUIT_SCALE);
                lerp.put(j + 1, move[1] + ((ovv >>> 8) & 0xFF) * backv[1] + ((vv >>> 8) & 0xFF) * frontv[1]
                        + normal[1] * Defines.POWERSUIT_SCALE);
                lerp.put(j + 2, move[2] + ((ovv >>> 16) & 0xFF) * backv[2] + ((vv >>> 16) & 0xFF) * frontv[2]
                        + normal[2] * Defines.POWERSUIT_SCALE);
                j += 3;
            }
        } else {
            int j = 0;
            for (int i = 0; i < nverts; i++ /* , v++, ov++, lerp+=4 */
            ) {
                ovv = ov[i];
                vv = v[i];

                lerp.put(j, move[0] + (ovv & 0xFF) * backv[0] + (vv & 0xFF) * frontv[0]);
                lerp.put(j + 1, move[1] + ((ovv >>> 8) & 0xFF) * backv[1] + ((vv >>> 8) & 0xFF) * frontv[1]);
                lerp.put(j + 2, move[2] + ((ovv >>> 16) & 0xFF) * backv[2] + ((vv >>> 16) & 0xFF) * frontv[2]);
                j += 3;
            }
        }
    }

    private final FloatBuffer colorArrayBuf = Lib.newFloatBuffer(qfiles.MAX_VERTS * 4);

    private final FloatBuffer vertexArrayBuf = Lib.newFloatBuffer(qfiles.MAX_VERTS * 3);

    boolean isFilled;

    float[] tmpVec = { 0, 0, 0 };

    /*
     * ============= GL_DrawAliasFrameLerp
     * 
     * interpolates between two frames and origins FIXME: batch lerp all
     * vertexes =============
     */
    private void GL_DrawAliasFrameLerp(qfiles.dmdl_t paliashdr, float backlerp) {

        float alpha;


        qfiles.daliasframe_t frame = paliashdr.aliasFrames[currententity.frame];

        int[] v = frame.verts;

        qfiles.daliasframe_t oldframe = paliashdr.aliasFrames[currententity.oldframe];

        int[] ov = oldframe.verts;

        int[] order = paliashdr.glCmds;

        if ((currententity.flags & Defines.RF_TRANSLUCENT) != 0)
            alpha = currententity.alpha;
        else
            alpha = 1.0f;

        
        if ((currententity.flags & (Defines.RF_SHELL_RED
                | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_BLUE
                | Defines.RF_SHELL_DOUBLE | Defines.RF_SHELL_HALF_DAM)) != 0)
            gl.glDisable(GL_TEXTURE_2D);


        float[] frontv = {0, 0, 0};
        Math3D.VectorSubtract(currententity.oldorigin, currententity.origin,
                frontv);
        float[][] vectors = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}

        };
        Math3D.AngleVectors(currententity.angles, vectors[0], vectors[1],
                vectors[2]);

        float[] move = {0, 0, 0};
        move[0] = Math3D.DotProduct(frontv, vectors[0]);
        move[1] = -Math3D.DotProduct(frontv, vectors[1]); 
        move[2] = Math3D.DotProduct(frontv, vectors[2]); 

        Math3D.VectorAdd(move, oldframe.translate, move);

        float frontlerp = 1.0f - backlerp;
        int i;
        float[] backv = {0, 0, 0};
        for (i = 0; i < 3; i++) {
            move[i] = backlerp * move[i] + frontlerp * frame.translate[i];
            frontv[i] = frontlerp * frame.scale[i];
            backv[i] = backlerp * oldframe.scale[i];
        }

        int index_xyz;
        int count;
        int orderIndex = 0;
        float l;
        if (gl_vertex_arrays.value != 0.0f) {
            GL_LerpVerts(paliashdr.num_xyz, ov, v, move, frontv, backv);

            gl.glEnableClientState(GL_VERTEX_ARRAY);
            gl.glVertexPointer(3, 0, vertexArrayBuf);

            
            if ((currententity.flags & (Defines.RF_SHELL_RED
                    | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_BLUE
                    | Defines.RF_SHELL_DOUBLE | Defines.RF_SHELL_HALF_DAM)) != 0) {
            	
                gl.glDisableClientState(GL_COLOR_ARRAY);
                gl.glColor4f(shadelight[0], shadelight[1], shadelight[2], alpha);
            } else {
                gl.glEnableClientState(GL_COLOR_ARRAY);
                gl.glColorPointer(4, 0, colorArrayBuf);

                
                
                
                FloatBuffer color = colorArrayBuf;
                int j = 0;
                for (i = 0; i < paliashdr.num_xyz; i++) {
                    
                    l = shadedots[(v[i] >>> 24) & 0xFF];
                    color.put(j++, l * shadelight[0]);
                    color.put(j++, l * shadelight[1]);
                    color.put(j++, l * shadelight[2]);
                    color.put(j++, alpha);
                }
            }

            if (qglLockArraysEXT)
                gl.glLockArraysEXT(0, paliashdr.num_xyz);

            while (true) {
                
                count = order[orderIndex++];
                if (count == 0)
                    break; 
                if (count < 0) {
                    count = -count;
                    gl.glBegin(GL_TRIANGLE_FAN);
                } else {
                    gl.glBegin(GL_TRIANGLE_STRIP);
                }

                
                if ((currententity.flags & (Defines.RF_SHELL_RED
                        | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_BLUE
                        | Defines.RF_SHELL_DOUBLE | Defines.RF_SHELL_HALF_DAM)) != 0) {
                    do {
                        index_xyz = order[orderIndex + 2];
                        orderIndex += 3;
                        /*
                         * vertexArrayBuf.position(4 * index_xyz);
                         * vertexArrayBuf.get(tmpVec); gl.glVertex3fv( tmpVec );
                         */
                        gl.glArrayElement(index_xyz);

                    } while (--count != 0);
                } else {
                    do {
                        
                        gl.glTexCoord2f(Float
                                .intBitsToFloat(order[orderIndex + 0]), Float
                                .intBitsToFloat(order[orderIndex + 1]));

                        index_xyz = order[orderIndex + 2];
                        orderIndex += 3;

                        
                        gl.glArrayElement(index_xyz);

                    } while (--count != 0);
                }
                gl.glEnd();
            }

            if (qglLockArraysEXT)
                gl.glUnlockArraysEXT();
        } else {
            GL_LerpVerts(paliashdr.num_xyz, ov, v, s_lerped, move,
                    frontv, backv);

            while (true) {
                
                count = order[orderIndex++];
                if (count == 0)
                    break; 
                if (count < 0) {
                    count = -count;
                    gl.glBegin(GL_TRIANGLE_FAN);
                } else {
                    gl.glBegin(GL_TRIANGLE_STRIP);
                }

                float[] tmp;
                if ((currententity.flags & (Defines.RF_SHELL_RED
                        | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_BLUE)) != 0) {
                    do {
                        index_xyz = order[orderIndex + 2];
                        orderIndex += 3;

                        gl.glColor4f(shadelight[0], shadelight[1],
                                shadelight[2], alpha);
                        tmp = s_lerped[index_xyz];
                        gl.glVertex3f(tmp[0], tmp[1], tmp[2]);

                    } while (--count != 0);
                } else {
                    do {
                        
                        
                        

                        gl.glTexCoord2f(Float
                                .intBitsToFloat(order[orderIndex + 0]), Float
                                .intBitsToFloat(order[orderIndex + 1]));
                        index_xyz = order[orderIndex + 2];
                        orderIndex += 3;

                        
                        l = shadedots[(v[index_xyz] >>> 24) & 0xFF];

                        gl.glColor4f(l * shadelight[0], l * shadelight[1], l
                                * shadelight[2], alpha);
                        tmp = s_lerped[index_xyz];
                        gl.glVertex3f(tmp[0], tmp[1], tmp[2]);
                    } while (--count != 0);
                }
                gl.glEnd();
            }
        }

        
        if ((currententity.flags & (Defines.RF_SHELL_RED
                | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_BLUE
                | Defines.RF_SHELL_DOUBLE | Defines.RF_SHELL_HALF_DAM)) != 0)
            gl.glEnable(GL_TEXTURE_2D);
    }

    /*
     * ============= GL_DrawAliasShadow =============
     */
    private void GL_DrawAliasShadow(qfiles.dmdl_t paliashdr, int posenum) {
        float[] point = { 0, 0, 0 };
        float lheight = currententity.origin[2] - lightspot[2];

        int[] order = paliashdr.glCmds;

        float height = -lheight + 1.0f;

        int orderIndex = 0;
        int index;

        while (true) {

            int count = order[orderIndex++];
            if (count == 0)
                break; 
            if (count < 0) {
                count = -count;
                gl.glBegin(GL_TRIANGLE_FAN);
            } else
                gl.glBegin(GL_TRIANGLE_STRIP);

            do {
                
                /*
                 * point[0] = verts[order[2]].v[0] * frame.scale[0] +
                 * frame.translate[0]; point[1] = verts[order[2]].v[1] *
                 * frame.scale[1] + frame.translate[1]; point[2] =
                 * verts[order[2]].v[2] * frame.scale[2] + frame.translate[2];
                 */

                if (gl_vertex_arrays.value != 0.0f) {
                    index = order[orderIndex + 2] * 3;
                    point[0] = vertexArrayBuf.get(index);
                    point[1] = vertexArrayBuf.get(index + 1);
                    point[2] = vertexArrayBuf.get(index + 2);
                } else {
                    Math3D.VectorCopy(s_lerped[order[orderIndex + 2]], point);
                }

                point[0] -= shadevector[0] * (point[2] + lheight);
                point[1] -= shadevector[1] * (point[2] + lheight);
                point[2] = height;
                gl.glVertex3f(point[0], point[1], point[2]);

                orderIndex += 3;

            } while (--count != 0);

            gl.glEnd();
        }
    }

    /*
     * * R_CullAliasModel
     */
    private boolean R_CullAliasModel(float[][] bbox, entity_t e) {

        qfiles.dmdl_t paliashdr = (qfiles.dmdl_t) currentmodel.extradata;

        if ((e.frame >= paliashdr.num_frames) || (e.frame < 0)) {
            VID.Printf(Defines.PRINT_ALL, "R_CullAliasModel "
                    + currentmodel.name + ": no such frame " + e.frame + '\n');
            e.frame = 0;
        }
        if ((e.oldframe >= paliashdr.num_frames) || (e.oldframe < 0)) {
            VID.Printf(Defines.PRINT_ALL, "R_CullAliasModel "
                    + currentmodel.name + ": no such oldframe " + e.oldframe
                    + '\n');
            e.oldframe = 0;
        }

        qfiles.daliasframe_t pframe = paliashdr.aliasFrames[e.frame];
        qfiles.daliasframe_t poldframe = paliashdr.aliasFrames[e.oldframe];

        /*
         * * compute axially aligned mins and maxs
         */
        float[] maxs = {0, 0, 0};
        float[] mins = {0, 0, 0};
        int i;
        if (pframe == poldframe) {
            for (i = 0; i < 3; i++) {
                mins[i] = pframe.translate[i];
                maxs[i] = mins[i] + pframe.scale[i] * 255;
            }
        } else {
            float[] oldmaxs = {0, 0, 0};
            float[] thismaxs = {0, 0, 0};
            float[] oldmins = {0, 0, 0};
            float[] thismins = {0, 0, 0};
            for (i = 0; i < 3; i++) {
                thismins[i] = pframe.translate[i];
                thismaxs[i] = thismins[i] + pframe.scale[i] * 255;

                oldmins[i] = poldframe.translate[i];
                oldmaxs[i] = oldmins[i] + poldframe.scale[i] * 255;

                mins[i] = Math.min(thismins[i], oldmins[i]);

                maxs[i] = Math.max(thismaxs[i], oldmaxs[i]);
            }
        }

        /*
         * * compute a full bounding box
         */
        for (i = 0; i < 8; i++) {
            float[] tmp = { 0, 0, 0 };

            if ((i & 1) != 0)
                tmp[0] = mins[0];
            else
                tmp[0] = maxs[0];

            if ((i & 2) != 0)
                tmp[1] = mins[1];
            else
                tmp[1] = maxs[1];

            if ((i & 4) != 0)
                tmp[2] = mins[2];
            else
                tmp[2] = maxs[2];

            Math3D.VectorCopy(tmp, bbox[i]);
        }

        /*
         * * rotate the bounding box
         */
        float[] angles = {0, 0, 0};
        Math3D.VectorCopy(e.angles, angles);
        angles[YAW] = -angles[YAW];
        float[][] vectors = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        Math3D.AngleVectors(angles, vectors[0], vectors[1], vectors[2]);

        for (i = 0; i < 8; i++) {
            float[] tmp = { 0, 0, 0 };

            Math3D.VectorCopy(bbox[i], tmp);

            bbox[i][0] = Math3D.DotProduct(vectors[0], tmp);
            bbox[i][1] = -Math3D.DotProduct(vectors[1], tmp);
            bbox[i][2] = Math3D.DotProduct(vectors[2], tmp);

            Math3D.VectorAdd(e.origin, bbox[i], bbox[i]);
        }

        int aggregatemask = ~0;

        for (int p = 0; p < 8; p++) {
            int mask = 0;

            for (int f = 0; f < 4; f++) {
                float dp = Math3D.DotProduct(frustum[f].normal, bbox[p]);

                if ((dp - frustum[f].dist) < 0) {
                    mask |= (1 << f);
                }
            }

            aggregatemask &= mask;
        }

        return aggregatemask != 0;

    }

    /*
     * ================= R_DrawAliasModel
     * 
     * =================
     */
    @Override
    void R_DrawAliasModel(entity_t e) {


        if ((e.flags & Defines.RF_WEAPONMODEL) == 0) {
            float[][] bbox = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
                    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            if (R_CullAliasModel(bbox, e))
                return;
        }

        if ((e.flags & Defines.RF_WEAPONMODEL) != 0) {
            if (r_lefthand.value == 2.0f)
                return;
        }

        qfiles.dmdl_t paliashdr = (qfiles.dmdl_t) currentmodel.extradata;


        int i;
        if ((currententity.flags & (Defines.RF_SHELL_HALF_DAM
                | Defines.RF_SHELL_GREEN | Defines.RF_SHELL_RED
                | Defines.RF_SHELL_BLUE | Defines.RF_SHELL_DOUBLE)) != 0) {
            Math3D.VectorClear(shadelight);
            if ((currententity.flags & Defines.RF_SHELL_HALF_DAM) != 0) {
                shadelight[0] = 0.56f;
                shadelight[1] = 0.59f;
                shadelight[2] = 0.45f;
            }
            if ((currententity.flags & Defines.RF_SHELL_DOUBLE) != 0) {
                shadelight[0] = 0.9f;
                shadelight[1] = 0.7f;
            }
            if ((currententity.flags & Defines.RF_SHELL_RED) != 0)
                shadelight[0] = 1.0f;
            if ((currententity.flags & Defines.RF_SHELL_GREEN) != 0)
                shadelight[1] = 1.0f;
            if ((currententity.flags & Defines.RF_SHELL_BLUE) != 0)
                shadelight[2] = 1.0f;
        }

        else if ((currententity.flags & Defines.RF_FULLBRIGHT) != 0) {
            for (i = 0; i < 3; i++)
                shadelight[i] = 1.0f;
        } else {
            R_LightPoint(currententity.origin, shadelight);

            
            
            if ((currententity.flags & Defines.RF_WEAPONMODEL) != 0) {
                
                
                if (shadelight[0] > shadelight[1]) {
                    if (shadelight[0] > shadelight[2])
                        r_lightlevel.value = 150 * shadelight[0];
                    else
                        r_lightlevel.value = 150 * shadelight[2];
                } else {
                    if (shadelight[1] > shadelight[2])
                        r_lightlevel.value = 150 * shadelight[1];
                    else
                        r_lightlevel.value = 150 * shadelight[2];
                }
            }

            if (gl_monolightmap.string.charAt(0) != '0') {
                float s = shadelight[0];

                if (s < shadelight[1])
                    s = shadelight[1];
                if (s < shadelight[2])
                    s = shadelight[2];

                shadelight[0] = s;
                shadelight[1] = s;
                shadelight[2] = s;
            }
        }

        if ((currententity.flags & Defines.RF_MINLIGHT) != 0) {
            for (i = 0; i < 3; i++)
                if (shadelight[i] > 0.1f)
                    break;
            if (i == 3) {
                shadelight[0] = 0.1f;
                shadelight[1] = 0.1f;
                shadelight[2] = 0.1f;
            }
        }

        if ((currententity.flags & Defines.RF_GLOW) != 0) {

            float scale = (float) (0.1f * Math.sin(r_newrefdef.time * 7));
            for (i = 0; i < 3; i++) {
                float min = shadelight[i] * 0.8f;
                shadelight[i] += scale;
                if (shadelight[i] < min)
                    shadelight[i] = min;
            }
        }

        
        
        if ((r_newrefdef.rdflags & Defines.RDF_IRGOGGLES) != 0
                && (currententity.flags & Defines.RF_IR_VISIBLE) != 0) {
            shadelight[0] = 1.0f;
            shadelight[1] = 0.0f;
            shadelight[2] = 0.0f;
        }
        
        

        shadedots = r_avertexnormal_dots[((int) (currententity.angles[1] * (SHADEDOT_QUANT / 360.0)))
                & (SHADEDOT_QUANT - 1)];

        float an = (float) (currententity.angles[1] / 180 * Math.PI);
        shadevector[0] = (float) Math.cos(-an);
        shadevector[1] = (float) Math.sin(-an);
        shadevector[2] = 1;
        Math3D.VectorNormalize(shadevector);

        
        
        

        c_alias_polys += paliashdr.num_tris;

        
        
        
        if ((currententity.flags & Defines.RF_DEPTHHACK) != 0)
            
            gl.glDepthRange(gldepthmin, gldepthmin + 0.3
                    * (gldepthmax - gldepthmin));

        if ((currententity.flags & Defines.RF_WEAPONMODEL) != 0
                && (r_lefthand.value == 1.0f)) {
            gl.glMatrixMode(GL_PROJECTION);
            gl.glPushMatrix();
            gl.glLoadIdentity();
            gl.glScalef(-1, 1, 1);
            MYgluPerspective(r_newrefdef.fov_y, (float) r_newrefdef.width
                    / r_newrefdef.height, 4, 4096);
            gl.glMatrixMode(GL_MODELVIEW);

            gl.glCullFace(GL_BACK);
        }

        gl.glPushMatrix();
        e.angles[PITCH] = -e.angles[PITCH]; 
        R_RotateForEntity(e);
        e.angles[PITCH] = -e.angles[PITCH];


        image_t skin;
        if (currententity.skin != null)
            skin = currententity.skin; 
        else {
            if (currententity.skinnum >= qfiles.MAX_MD2SKINS)
                skin = currentmodel.skins[0];
            else {
                skin = currentmodel.skins[currententity.skinnum];
                if (skin == null)
                    skin = currentmodel.skins[0];
            }
        }
        if (skin == null)
            skin = r_notexture; 
        GL_Bind(skin.texnum);

        

        gl.glShadeModel(GL_SMOOTH);

        GL_TexEnv(GL_MODULATE);
        if ((currententity.flags & Defines.RF_TRANSLUCENT) != 0) {
            gl.glEnable(GL_BLEND);
        }

        if ((currententity.frame >= paliashdr.num_frames)
                || (currententity.frame < 0)) {
            VID.Printf(Defines.PRINT_ALL, "R_DrawAliasModel "
                    + currentmodel.name + ": no such frame "
                    + currententity.frame + '\n');
            currententity.frame = 0;
            currententity.oldframe = 0;
        }

        if ((currententity.oldframe >= paliashdr.num_frames)
                || (currententity.oldframe < 0)) {
            VID.Printf(Defines.PRINT_ALL, "R_DrawAliasModel "
                    + currentmodel.name + ": no such oldframe "
                    + currententity.oldframe + '\n');
            currententity.frame = 0;
            currententity.oldframe = 0;
        }

        if (r_lerpmodels.value == 0.0f)
            currententity.backlerp = 0;
        GL_DrawAliasFrameLerp(paliashdr, currententity.backlerp);

        GL_TexEnv(GL_REPLACE);
        gl.glShadeModel(GL_FLAT);

        gl.glPopMatrix();

        if ((currententity.flags & Defines.RF_WEAPONMODEL) != 0
                && (r_lefthand.value == 1.0F)) {
            gl.glMatrixMode(GL_PROJECTION);
            gl.glPopMatrix();
            gl.glMatrixMode(GL_MODELVIEW);
            gl.glCullFace(GL_FRONT);
        }

        if ((currententity.flags & Defines.RF_TRANSLUCENT) != 0) {
            gl.glDisable(GL_BLEND);
        }

        if ((currententity.flags & Defines.RF_DEPTHHACK) != 0)
            gl.glDepthRange(gldepthmin, gldepthmax);

        if (gl_shadows.value != 0.0f
                && (currententity.flags & (Defines.RF_TRANSLUCENT | Defines.RF_WEAPONMODEL)) == 0) {
            gl.glPushMatrix();
            R_RotateForEntity(e);
            gl.glDisable(GL_TEXTURE_2D);
            gl.glEnable(GL_BLEND);
            gl.glColor4f(0, 0, 0, 0.5f);
            GL_DrawAliasShadow(paliashdr, currententity.frame);
            gl.glEnable(GL_TEXTURE_2D);
            gl.glDisable(GL_BLEND);
            gl.glPopMatrix();
        }
        gl.glColor4f(1, 1, 1, 1);
    }

}