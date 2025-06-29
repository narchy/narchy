/*
 * Warp.java
 * Copyright (C) 2003
 *
 * $Id: Warp.java,v 1.2 2006-11-21 00:50:46 cawe Exp $
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
package jake2.render.fast;

import jake2.Defines;
import jake2.Globals;
import jake2.qcommon.Com;
import jake2.render.glpoly_t;
import jake2.render.image_t;
import jake2.render.msurface_t;
import jake2.util.Math3D;
import jake2.util.Vec3Cache;

/**
 * Warp
 *  
 * @author cwei
 */
public abstract class Warp extends Model {
	
	public static final float[] SIN = {
		0f, 0.19633f, 0.392541f, 0.588517f, 0.784137f, 0.979285f, 1.17384f, 1.3677f,
		 1.56072f, 1.75281f, 1.94384f, 2.1337f, 2.32228f, 2.50945f, 2.69512f, 2.87916f,
		 3.06147f, 3.24193f, 3.42044f, 3.59689f, 3.77117f, 3.94319f, 4.11282f, 4.27998f,
		 4.44456f, 4.60647f, 4.76559f, 4.92185f, 5.07515f, 5.22538f, 5.37247f, 5.51632f,
		 5.65685f, 5.79398f, 5.92761f, 6.05767f, 6.18408f, 6.30677f, 6.42566f, 6.54068f,
		 6.65176f, 6.75883f, 6.86183f, 6.9607f, 7.05537f, 7.14579f, 7.23191f, 7.31368f,
		 7.39104f, 7.46394f, 7.53235f, 7.59623f, 7.65552f, 7.71021f, 7.76025f, 7.80562f,
		 7.84628f, 7.88222f, 7.91341f, 7.93984f, 7.96148f, 7.97832f, 7.99036f, 7.99759f,
		 8f, 7.99759f, 7.99036f, 7.97832f, 7.96148f, 7.93984f, 7.91341f, 7.88222f,
		 7.84628f, 7.80562f, 7.76025f, 7.71021f, 7.65552f, 7.59623f, 7.53235f, 7.46394f,
		 7.39104f, 7.31368f, 7.23191f, 7.14579f, 7.05537f, 6.9607f, 6.86183f, 6.75883f,
		 6.65176f, 6.54068f, 6.42566f, 6.30677f, 6.18408f, 6.05767f, 5.92761f, 5.79398f,
		 5.65685f, 5.51632f, 5.37247f, 5.22538f, 5.07515f, 4.92185f, 4.76559f, 4.60647f,
		 4.44456f, 4.27998f, 4.11282f, 3.94319f, 3.77117f, 3.59689f, 3.42044f, 3.24193f,
		 3.06147f, 2.87916f, 2.69512f, 2.50945f, 2.32228f, 2.1337f, 1.94384f, 1.75281f,
		 1.56072f, 1.3677f, 1.17384f, 0.979285f, 0.784137f, 0.588517f, 0.392541f, 0.19633f,
		 9.79717e-16f, -0.19633f, -0.392541f, -0.588517f, -0.784137f, -0.979285f, -1.17384f, -1.3677f,
		 -1.56072f, -1.75281f, -1.94384f, -2.1337f, -2.32228f, -2.50945f, -2.69512f, -2.87916f,
		 -3.06147f, -3.24193f, -3.42044f, -3.59689f, -3.77117f, -3.94319f, -4.11282f, -4.27998f,
		 -4.44456f, -4.60647f, -4.76559f, -4.92185f, -5.07515f, -5.22538f, -5.37247f, -5.51632f,
		 -5.65685f, -5.79398f, -5.92761f, -6.05767f, -6.18408f, -6.30677f, -6.42566f, -6.54068f,
		 -6.65176f, -6.75883f, -6.86183f, -6.9607f, -7.05537f, -7.14579f, -7.23191f, -7.31368f,
		 -7.39104f, -7.46394f, -7.53235f, -7.59623f, -7.65552f, -7.71021f, -7.76025f, -7.80562f,
		 -7.84628f, -7.88222f, -7.91341f, -7.93984f, -7.96148f, -7.97832f, -7.99036f, -7.99759f,
		 -8f, -7.99759f, -7.99036f, -7.97832f, -7.96148f, -7.93984f, -7.91341f, -7.88222f,
		 -7.84628f, -7.80562f, -7.76025f, -7.71021f, -7.65552f, -7.59623f, -7.53235f, -7.46394f,
		 -7.39104f, -7.31368f, -7.23191f, -7.14579f, -7.05537f, -6.9607f, -6.86183f, -6.75883f,
		 -6.65176f, -6.54068f, -6.42566f, -6.30677f, -6.18408f, -6.05767f, -5.92761f, -5.79398f,
		 -5.65685f, -5.51632f, -5.37247f, -5.22538f, -5.07515f, -4.92185f, -4.76559f, -4.60647f,
		 -4.44456f, -4.27998f, -4.11282f, -3.94319f, -3.77117f, -3.59689f, -3.42044f, -3.24193f,
		 -3.06147f, -2.87916f, -2.69512f, -2.50945f, -2.32228f, -2.1337f, -1.94384f, -1.75281f,
		 -1.56072f, -1.3677f, -1.17384f, -0.979285f, -0.784137f, -0.588517f, -0.392541f, -0.19633f
	};

	private float	skyrotate;
	private final float[] skyaxis = {0, 0, 0};
	private final image_t[] sky_images = new image_t[6];

	private msurface_t	warpface;

	private static final int SUBDIVIDE_SIZE = 64;

	/**
	 * BoundPoly
	 * @param numverts
	 * @param verts
	 * @param mins
	 * @param maxs
	 */
	private static void BoundPoly(int numverts, float[][] verts, float[] mins, float[] maxs) {
		mins[0] = mins[1] = mins[2] = 9999;
		maxs[0] = maxs[1] = maxs[2] = -9999;

		for (int i=0 ; i<numverts ; i++) {
			float[] v = verts[i];
			for (int j = 0; j<3 ; j++) {
				float vj = v[j];
				if (vj < mins[j])
					mins[j] = vj;
				if (vj > maxs[j])
					maxs[j] = vj;
			}
		}
	}

	/**
	 * SubdividePolygon
	 * @param numverts
	 * @param verts
	 */
    private void SubdividePolygon(int numverts, float[][] verts)
	{

		if (numverts > 60)
			Com.Error(Defines.ERR_DROP, "numverts = " + numverts);

		float[] mins = Vec3Cache.get();
		float[] maxs = Vec3Cache.get();

		BoundPoly(numverts, verts, mins, maxs);

		float[] dist = new float[64];
		float[][] back = new float[64][3];
		float[][] front = new float[64][3];
		int i;
		for (i=0 ; i<3 ; i++)
		{
			float m = (mins[i] + maxs[i]) * 0.5f;
			m = SUBDIVIDE_SIZE * (float)Math.floor(m / SUBDIVIDE_SIZE + 0.5f);
			if (maxs[i] - m < 8)
				continue;
			if (m - mins[i] < 8)
				continue;


			int j;
			for (j=0 ; j<numverts ; j++) {
				dist[j] = verts[j][i] - m;
			}

			
			dist[j] = dist[0];

			Math3D.VectorCopy(verts[0], verts[numverts]);

			int b;
			int f = b = 0;
			for (j=0 ; j<numverts ; j++)
			{
				float[] v = verts[j];
				if (dist[j] >= 0)
				{
					Math3D.VectorCopy(v, front[f]);
					f++;
				}
				if (dist[j] <= 0)
				{
					Math3D.VectorCopy(v, back[b]);
					b++;
				}
				if (dist[j] == 0 || dist[j+1] == 0) continue;
				
				if ( (dist[j] > 0) != (dist[j+1] > 0) )
				{

					float frac = dist[j] / (dist[j] - dist[j + 1]);
					for (int k = 0; k<3 ; k++)
						front[f][k] = back[b][k] = v[k] + frac*(verts[j+1][k] - v[k]);
						
					f++;
					b++;
				}
			}

			SubdividePolygon(f, front);
			SubdividePolygon(b, back);
			
			Vec3Cache.release(2); 
			return;
		}
		
		Vec3Cache.release(2); 
		
		
		
		
		

		
		glpoly_t poly = Polygon.create(numverts + 2);

		poly.next = warpface.polys;
		warpface.polys = poly;
		
		float[] total = Vec3Cache.get();
		Math3D.VectorClear(total);
		float total_s = 0;
		float total_t = 0;
		for (i = 0; i < numverts; i++) {
            poly.x(i + 1, verts[i][0]);
            poly.y(i + 1, verts[i][1]);
            poly.z(i + 1, verts[i][2]);
			float s = Math3D.DotProduct(verts[i], warpface.texinfo.vecs[0]);
			float t = Math3D.DotProduct(verts[i], warpface.texinfo.vecs[1]);

			total_s += s;
            total_t += t;
            Math3D.VectorAdd(total, verts[i], total);

            poly.s1(i + 1, s);
            poly.t1(i + 1, t);
        }
        
        float scale = 1.0f / numverts; 
        poly.x(0, total[0] * scale);
        poly.y(0, total[1] * scale);
        poly.z(0, total[2] * scale);
        poly.s1(0, total_s * scale);
        poly.t1(0, total_t * scale);

        poly.x(i + 1, poly.x(1));
        poly.y(i + 1, poly.y(1));
        poly.z(i + 1, poly.z(1));
        poly.s1(i + 1, poly.s1(1));
        poly.t1(i + 1, poly.t1(1));
        poly.s2(i + 1, poly.s2(1));
        poly.t2(i + 1, poly.t2(1));
        
        Vec3Cache.release(); 
	}

	private final float[][] tmpVerts = new float[64][3];
	/**
	 * GL_SubdivideSurface
	 * Breaks a polygon up along axial 64 unit
	 * boundaries so that turbulent and sky warps
	 * can be done reasonably.
	 */
    @Override
    void GL_SubdivideSurface(msurface_t fa) {
        float[][] verts = tmpVerts;
		warpface = fa;
        
        
        
        int numverts = 0;
        for (int i = 0; i < fa.numedges; i++) {
            int lindex = loadmodel.surfedges[fa.firstedge + i];

			float[] vec;
			if (lindex > 0)
                vec = loadmodel.vertexes[loadmodel.edges[lindex].v[0]].position;
            else
                vec = loadmodel.vertexes[loadmodel.edges[-lindex].v[1]].position;
            Math3D.VectorCopy(vec, verts[numverts]);
            numverts++;
        }
        SubdividePolygon(numverts, verts);
    }


	private static final float TURBSCALE = (float)(256.0f / (2 * Math.PI));

	/**
	 * EmitWaterPolys
	 * Does a water warp on the pre-fragmented glpoly_t chain
	 */
    @Override
    void EmitWaterPolys(msurface_t fa)
	{
		float rdt = r_newrefdef.time;

		float scroll;
		if ((fa.texinfo.flags & Defines.SURF_FLOWING) != 0)
			scroll = -64 * ( (r_newrefdef.time*0.5f) - (int)(r_newrefdef.time*0.5f) );
		else
			scroll = 0;

		for (glpoly_t bp = fa.polys; bp != null; bp = bp.next) {

			gl.glBegin(GL_TRIANGLE_FAN);
            for (int i = 0; i < bp.numverts; i++) {
				float os = bp.s1(i);
				float ot = bp.t1(i);

				float s = os
						+ Warp.SIN[(int) ((ot * 0.125f + r_newrefdef.time) * TURBSCALE) & 255];
				s += scroll;
                s *= (1.0f / 64);

				float t = ot
						+ Warp.SIN[(int) ((os * 0.125f + rdt) * TURBSCALE) & 255];
				t *= (1.0f / 64);

                gl.glTexCoord2f(s, t);
                gl.glVertex3f(bp.x(i), bp.y(i), bp.z(i));
            }
            gl.glEnd();
        }
	}



	private final float[][] skyclip = {
		{ 1,  1, 0},
		{ 1, -1, 0},
		{ 0, -1, 1},
		{ 0,  1, 1},
		{ 1,  0, 1},
		{-1,  0, 1} 
	};

	private int c_sky;

	
	private final int[][]	st_to_vec =
	{
		{3,-1,2},
		{-3,1,2},

		{1,3,2},
		{-1,-3,2},

		{-2,-1,3},		
		{2,-1,-3}		

	};

	private final int[][]	vec_to_st =
	{
		{-2,3,1},
		{2,3,-1},

		{1,3,2},
		{-1,3,-2},

		{-2,-1,3},
		{-2,1,-3}

	};

	private final float[][] skymins = new float[2][6];
	private final float[][] skymaxs = new float[2][6];
	private float	sky_min;
    private float sky_max;

	
	private final float[] v = {0, 0, 0};
	private final float[] av = {0, 0, 0};
	/**
	 * DrawSkyPolygon
	 * @param nump
	 * @param vecs
	 */
    private void DrawSkyPolygon(int nump, float[][] vecs)
	{
		c_sky++;
		
		Math3D.VectorCopy(Globals.vec3_origin, v);
		int i;
		for (i=0; i<nump ; i++)
		{
			Math3D.VectorAdd(vecs[i], v, v);
		}
		av[0] = Math.abs(v[0]);
		av[1] = Math.abs(v[1]);
		av[2] = Math.abs(v[2]);
		int axis;
		if (av[0] > av[1] && av[0] > av[2])
		{
			if (v[0] < 0)
				axis = 1;
			else
				axis = 0;
		}
		else if (av[1] > av[2] && av[1] > av[0])
		{
			if (v[1] < 0)
				axis = 3;
			else
				axis = 2;
		}
		else
		{
			if (v[2] < 0)
				axis = 5;
			else
				axis = 4;
		}


		for (i=0 ; i<nump ; i++)
		{
			int j = vec_to_st[axis][2];
			float dv;
			if (j > 0)
				dv = vecs[i][j - 1];
			else
				dv = -vecs[i][-j - 1];
			if (dv < 0.001f)
				continue;	
			j = vec_to_st[axis][0];
			float s;
			if (j < 0)
				s = -vecs[i][-j -1] / dv;
			else
				s = vecs[i][j-1] / dv;
			j = vec_to_st[axis][1];
			float t;
			if (j < 0)
				t = -vecs[i][-j -1] / dv;
			else
				t = vecs[i][j-1] / dv;

			if (s < skymins[0][axis])
				skymins[0][axis] = s;
			if (t < skymins[1][axis])
				skymins[1][axis] = t;
			if (s > skymaxs[0][axis])
				skymaxs[0][axis] = s;
			if (t > skymaxs[1][axis])
				skymaxs[1][axis] = t;
		}
	}

	private static final float ON_EPSILON = 0.1f;
	private static final int MAX_CLIP_VERTS = 64;
	
	private static final int SIDE_BACK = 1;
	private static final int SIDE_FRONT = 0;
	private static final int SIDE_ON = 2;
	
	private final float[] dists = new float[MAX_CLIP_VERTS];
	private final int[] sides = new int[MAX_CLIP_VERTS];
	private final float[][][][] newv = new float[6][2][MAX_CLIP_VERTS][3];

	/**
	 * ClipSkyPolygon
	 * @param nump
	 * @param vecs
	 * @param stage
	 */
    private void ClipSkyPolygon(int nump, float[][] vecs, int stage)
	{
		if (nump > MAX_CLIP_VERTS-2)
			Com.Error(Defines.ERR_DROP, "ClipSkyPolygon: MAX_CLIP_VERTS");
		if (stage == 6)
		{	
			DrawSkyPolygon(nump, vecs);
			return;
		}

		boolean front = false;
		boolean back = false;
		float[] norm = skyclip[stage];

		int i;
		float d;
		for (i=0 ; i<nump ; i++)
		{
			d = Math3D.DotProduct(vecs[i], norm);
			if (d > ON_EPSILON)
			{
				front = true;
				sides[i] = SIDE_FRONT;
			}
			else if (d < -ON_EPSILON)
			{
				back = true;
				sides[i] = SIDE_BACK;
			}
			else
				sides[i] = SIDE_ON;
			dists[i] = d;
		}

		if (!front || !back)
		{	
			ClipSkyPolygon (nump, vecs, stage+1);
			return;
		}

		
		sides[i] = sides[0];
		dists[i] = dists[0];
		Math3D.VectorCopy(vecs[0], vecs[i]);

		int newc0 = 0; 	int  newc1 = 0;
		for (i=0; i<nump ; i++)
		{
			float[] v = vecs[i];
            switch (sides[i]) {
                case SIDE_FRONT -> {
                    Math3D.VectorCopy(v, newv[stage][0][newc0]);
                    newc0++;
                }
                case SIDE_BACK -> {
                    Math3D.VectorCopy(v, newv[stage][1][newc1]);
                    newc1++;
                }
                case SIDE_ON -> {
                    Math3D.VectorCopy(v, newv[stage][0][newc0]);
                    newc0++;
                    Math3D.VectorCopy(v, newv[stage][1][newc1]);
                    newc1++;
                }
            }

			if (sides[i] == SIDE_ON || sides[i+1] == SIDE_ON || sides[i+1] == sides[i])
				continue;

			d = dists[i] / (dists[i] - dists[i+1]);
			for (int j = 0; j<3 ; j++)
			{
				float e = v[j] + d * (vecs[i + 1][j] - v[j]);
				newv[stage][0][newc0][j] = e;
				newv[stage][1][newc1][j] = e;
			}
			newc0++;
			newc1++;
		}

		
		ClipSkyPolygon(newc0, newv[stage][0], stage+1);
		ClipSkyPolygon(newc1, newv[stage][1], stage+1);
	}

	private final float[][] verts = new float[MAX_CLIP_VERTS][3];

	/**
	 * R_AddSkySurface
	 */
    @Override
    void R_AddSkySurface(msurface_t fa)
	{
	    
        for (glpoly_t p = fa.polys; p != null; p = p.next) {
            for (int i = 0; i < p.numverts; i++) {
                verts[i][0] = p.x(i) - r_origin[0];
                verts[i][1] = p.y(i) - r_origin[1];
                verts[i][2] = p.z(i) - r_origin[2];
            }
            ClipSkyPolygon(p.numverts, verts, 0);
        }
 	}

	/**
	 * R_ClearSkyBox
	 */
    @Override
    void R_ClearSkyBox()
	{
		float[] skymins0 = skymins[0];
		float[] skymins1 = skymins[1];
		float[] skymaxs0 = skymaxs[0];
		float[] skymaxs1 = skymaxs[1];
		
		for (int i=0 ; i<6 ; i++)
		{
			skymins0[i] = skymins1[i] = 9999;
			skymaxs0[i] = skymaxs1[i] = -9999;
		}
	}
	
	
	private final float[] v1 = {0, 0, 0};
	private final float[] b = {0, 0, 0};
	/**
	 * MakeSkyVec
	 * @param s
	 * @param t
	 * @param axis
	 */
    private void MakeSkyVec(float s, float t, int axis)
	{
		b[0] = s*2300;
		b[1] = t*2300;
		b[2] = 2300;

		for (int j = 0; j<3 ; j++)
		{
			int k = st_to_vec[axis][j];
			if (k < 0)
				v1[j] = -b[-k - 1];
			else
				v1[j] = b[k - 1];
		}

		
		s = (s + 1) * 0.5f;
		t = (t + 1) * 0.5f;

		if (s < sky_min)
			s = sky_min;
		else if (s > sky_max)
			s = sky_max;
		if (t < sky_min)
			t = sky_min;
		else if (t > sky_max)
			t = sky_max;

		t = 1.0f - t;
		gl.glTexCoord2f (s, t);
		gl.glVertex3f(v1[0], v1[1], v1[2]);
	}

	private final int[] skytexorder = {0,2,1,3,4,5};

	/**
	 * R_DrawSkyBox
	 */
    @Override
    void R_DrawSkyBox()
	{
		int i;

		if (skyrotate != 0)
		{	
			for (i=0 ; i<6 ; i++)
				if (skymins[0][i] < skymaxs[0][i]
				&& skymins[1][i] < skymaxs[1][i])
					break;
			if (i == 6)
				return;		
		}

		gl.glPushMatrix ();
		gl.glTranslatef (r_origin[0], r_origin[1], r_origin[2]);
		gl.glRotatef (r_newrefdef.time * skyrotate, skyaxis[0], skyaxis[1], skyaxis[2]);

		for (i=0 ; i<6 ; i++)
		{
			if (skyrotate != 0)
			{	
				skymins[0][i] = -1;
				skymins[1][i] = -1;
				skymaxs[0][i] = 1;
				skymaxs[1][i] = 1;
			}

			if (skymins[0][i] >= skymaxs[0][i]
			|| skymins[1][i] >= skymaxs[1][i])
				continue;

			GL_Bind(sky_images[skytexorder[i]].texnum);

			gl.glBegin(GL_QUADS);
			MakeSkyVec(skymins[0][i], skymins[1][i], i);
			MakeSkyVec(skymins[0][i], skymaxs[1][i], i);
			MakeSkyVec(skymaxs[0][i], skymaxs[1][i], i);
			MakeSkyVec(skymaxs[0][i], skymins[1][i], i);
			gl.glEnd ();
		}
		gl.glPopMatrix ();
	}

	
	private final String[] suf = {"rt", "bk", "lf", "ft", "up", "dn"};
	
	/**
	 * R_SetSky
	 * @param name
	 * @param rotate
	 * @param axis
	 */
	@Override
    public void R_SetSky(String name, float rotate, float[] axis)
	{
		assert (axis.length == 3) : "vec3_t bug";

		skyrotate = rotate;
		Math3D.VectorCopy(axis, skyaxis);

		for (int i=0 ; i<6 ; i++)
		{
			
			if (gl_skymip.value != 0 || skyrotate != 0)
				gl_picmip.value++;

			String pathname;
			if ( qglColorTableEXT && gl_ext_palettedtexture.value != 0) {
				
				pathname = "env/" + name + suf[i] + ".pcx";
			} else {
				
				pathname = "env/" + name + suf[i] + ".tga";
			}

			sky_images[i] = GL_FindImage(pathname, it_sky);

			if (sky_images[i] == null)
				sky_images[i] = r_notexture;

			if (gl_skymip.value != 0 || skyrotate != 0)
			{	
				gl_picmip.value--;
				sky_min = 1.0f / 256;
				sky_max = 255.0f / 256;
			}
			else	
			{
				sky_min = 1.0f / 512;
				sky_max = 511.0f / 512;
			}
		}
	}
}