/*
 * Surf.java
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
import jake2.client.dlight_t;
import jake2.client.entity_t;
import jake2.client.lightstyle_t;
import jake2.game.cplane_t;
import jake2.qcommon.Com;
import jake2.render.*;
import jake2.util.Lib;
import jake2.util.Math3D;

import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Surf
 *  
 * @author cwei
 */
abstract class Surf extends Draw {

	
	private final float[] modelorg = {0, 0, 0};

	private msurface_t	r_alpha_surfaces;

	static final int DYNAMIC_LIGHT_WIDTH = 128;
	static final int DYNAMIC_LIGHT_HEIGHT = 128;

	static final int LIGHTMAP_BYTES = 4;

	private static final int BLOCK_WIDTH = 128;
	private static final int BLOCK_HEIGHT = 128;

	private static final int MAX_LIGHTMAPS = 128;

	private int c_visible_lightmaps;

	private static final int GL_LIGHTMAP_FORMAT = GL_RGBA;

	static class gllightmapstate_t 
	{
		int internal_format;
		int current_lightmap_texture;

		final msurface_t[] lightmap_surfaces = new msurface_t[MAX_LIGHTMAPS];
		final int[] allocated = new int[BLOCK_WIDTH];

		
		
		final IntBuffer lightmap_buffer = Lib.newIntBuffer(BLOCK_WIDTH * BLOCK_HEIGHT, ByteOrder.LITTLE_ENDIAN);
		
		
		gllightmapstate_t() {
			for (int i = 0; i < MAX_LIGHTMAPS; i++)
				lightmap_surfaces[i] = new msurface_t();
		}
		
		void clearLightmapSurfaces() {
			for (int i = 0; i < MAX_LIGHTMAPS; i++)
				
				lightmap_surfaces[i] = new msurface_t();
		}
		
	} 

	private final gllightmapstate_t gl_lms = new gllightmapstate_t();










	
	abstract byte[] Mod_ClusterPVS(int cluster, model_t model);
	
	abstract void R_DrawSkyBox();
	abstract void R_AddSkySurface(msurface_t surface);
	abstract void R_ClearSkyBox();
	abstract void EmitWaterPolys(msurface_t fa);
	
	abstract void R_MarkLights (dlight_t light, int bit, mnode_t node);
	abstract void R_SetCacheState( msurface_t surf );
	abstract void R_BuildLightMap(msurface_t surf, IntBuffer dest, int stride);

	/*
	=============================================================

		BRUSH MODELS

	=============================================================
	*/

	/*
	===============
	R_TextureAnimation

	Returns the proper texture for a given time and base texture
	===============
	*/
	private image_t R_TextureAnimation(mtexinfo_t tex)
	{

        if (tex.next == null)
			return tex.image;

        int c = currententity.frame % tex.numframes;
		while (c != 0)
		{
			tex = tex.next;
			c--;
		}

		return tex.image;
	}

	/*
	================
	DrawGLPoly
	================
	*/
	private void DrawGLPoly(glpoly_t p)
	{
		gl.glBegin(GL_POLYGON);
		for (int i=0 ; i<p.numverts ; i++)
		{
			gl.glTexCoord2f(p.s1(i), p.t1(i));
			gl.glVertex3f(p.x(i), p.y(i), p.z(i));
		}
		gl.glEnd();
	}

	
	
	/*
	================
	DrawGLFlowingPoly -- version of DrawGLPoly that handles scrolling texture
	================
	*/
	private void DrawGLFlowingPoly(msurface_t fa)
	{
		float scroll = -64 * ( (r_newrefdef.time / 40.0f) - (int)(r_newrefdef.time / 40.0f) );
		if(scroll == 0.0f)
			scroll = -64.0f;

		gl.glBegin (GL_POLYGON);
		glpoly_t p = fa.polys;
		for (int i=0 ; i<p.numverts ; i++)
		{
			gl.glTexCoord2f(p.s1(i) + scroll, p.t1(i));
			gl.glVertex3f(p.x(i), p.y(i), p.z(i));
		}
		gl.glEnd ();
	}
	
	

	/*
	** R_DrawTriangleOutlines
	*/
	private void R_DrawTriangleOutlines()
	{
		if (gl_showtris.value == 0)
			return;

		gl.glDisable (GL_TEXTURE_2D);
		gl.glDisable (GL_DEPTH_TEST);
		gl.glColor4f (1,1,1,1);

		for (int i=0 ; i<MAX_LIGHTMAPS ; i++)
		{

            for (msurface_t surf = gl_lms.lightmap_surfaces[i]; surf != null; surf = surf.lightmapchain )
			{
				glpoly_t p = surf.polys;
				for ( ; p != null ; p=p.chain)
				{
					for (int j=2 ; j<p.numverts ; j++ )
					{
						gl.glBegin (GL_LINE_STRIP);
						gl.glVertex3f(p.x(0), p.y(0), p.z(0));
						gl.glVertex3f(p.x(j-1), p.y(j-1), p.z(j-1));
						gl.glVertex3f(p.x(j), p.y(j), p.z(j));
						gl.glVertex3f(p.x(0), p.y(0), p.z(0));
						gl.glEnd ();
					}
				}
			}
		}
		gl.glEnable (GL_DEPTH_TEST);
		gl.glEnable (GL_TEXTURE_2D);
	}

	/*
	** DrawGLPolyChain
	*/
	private void DrawGLPolyChain(glpoly_t p, float soffset, float toffset)
	{
		if ( soffset == 0 && toffset == 0 )
		{
			for ( ; p != null; p = p.chain )
			{
				gl.glBegin(GL_POLYGON);
				for (int j=0 ; j<p.numverts ; j++)
				{
					gl.glTexCoord2f (p.s2(j), p.t2(j));
					gl.glVertex3f(p.x(j), p.y(j), p.z(j));
				}
				gl.glEnd();
			}
		}
		else
		{
			for ( ; p != null; p = p.chain )
			{
				gl.glBegin(GL_POLYGON);
				for (int j=0 ; j<p.numverts ; j++)
				{
					gl.glTexCoord2f (p.s2(j) - soffset, p.t2(j) - toffset);
					gl.glVertex3f(p.x(j), p.y(j), p.z(j));
				}
				gl.glEnd();
			}
		}
	}

	/*
	** R_BlendLightMaps
	**
	** This routine takes all the given light mapped surfaces in the world and
	** blends them into the framebuffer.
	*/
	private void R_BlendLightmaps()
	{


        if (r_fullbright.value != 0)
			return;
		if (r_worldmodel.lightdata == null)
			return;

		
		gl.glDepthMask( false );

		/*
		** set the appropriate blending mode unless we're only looking at the
		** lightmaps.
		*/
		if (gl_lightmap.value == 0)
		{
			gl.glEnable(GL_BLEND);

			if ( gl_saturatelighting.value != 0)
			{
				gl.glBlendFunc( GL_ONE, GL_ONE );
			}
			else
			{
				char format = gl_monolightmap.string.toUpperCase().charAt(0);
				if ( format != '0' )
				{
					switch (format) {
						case 'I', 'L' -> gl.glBlendFunc(GL_ZERO, GL_SRC_COLOR);
						default -> gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					}
				}
				else
				{
					gl.glBlendFunc(GL_ZERO, GL_SRC_COLOR );
				}
			}
		}

		if ( currentmodel == r_worldmodel )
			c_visible_lightmaps = 0;

		/*
		** render static lightmaps first
		*/
        msurface_t surf;
        for (int i = 1; i < MAX_LIGHTMAPS; i++ )
		{
			if ( gl_lms.lightmap_surfaces[i] != null )
			{
				if (currentmodel == r_worldmodel)
					c_visible_lightmaps++;
					
				GL_Bind( gl_state.lightmap_textures + i);

				for ( surf = gl_lms.lightmap_surfaces[i]; surf != null; surf = surf.lightmapchain )
				{
					if ( surf.polys != null )
						DrawGLPolyChain( surf.polys, 0, 0 );
				}
			}
		}

		
		

		/*
		** render dynamic lightmaps
		*/
		if ( gl_dynamic.value != 0 )
		{
			LM_InitBlock();

			GL_Bind( gl_state.lightmap_textures+0 );

			if (currentmodel == r_worldmodel)
				c_visible_lightmaps++;

            msurface_t newdrawsurf = gl_lms.lightmap_surfaces[0];

            for ( surf = gl_lms.lightmap_surfaces[0]; surf != null; surf = surf.lightmapchain )
			{
                IntBuffer base;

                int smax = (surf.extents[0] >> 4) + 1;
                int tmax = (surf.extents[1] >> 4) + 1;
				
				pos_t lightPos = new pos_t(surf.dlight_s, surf.dlight_t);

				if ( LM_AllocBlock( smax, tmax, lightPos) )
				{
					
					surf.dlight_s = lightPos.x;
					surf.dlight_t = lightPos.y;

					base = gl_lms.lightmap_buffer;
					base.position(surf.dlight_t * BLOCK_WIDTH + surf.dlight_s );

					R_BuildLightMap (surf, base.slice(), BLOCK_WIDTH);
				}
				else
				{


                    LM_UploadBlock( true );


                    msurface_t drawsurf;
                    for (drawsurf = newdrawsurf; drawsurf != surf; drawsurf = drawsurf.lightmapchain )
					{
						if ( drawsurf.polys != null )
							DrawGLPolyChain( drawsurf.polys, 
											  ( drawsurf.light_s - drawsurf.dlight_s ) * ( 1.0f / 128.0f ), 
											( drawsurf.light_t - drawsurf.dlight_t ) * ( 1.0f / 128.0f ) );
					}

					newdrawsurf = drawsurf;

					
					LM_InitBlock();

					
					if ( !LM_AllocBlock( smax, tmax, lightPos) )
					{
						Com.Error( Defines.ERR_FATAL, "Consecutive calls to LM_AllocBlock(" + smax + ',' + tmax + ") failed (dynamic)\n");
					}
					
					
					surf.dlight_s = lightPos.x;
					surf.dlight_t = lightPos.y;

					base = gl_lms.lightmap_buffer;
					base.position(surf.dlight_t * BLOCK_WIDTH + surf.dlight_s );

					R_BuildLightMap (surf, base.slice(), BLOCK_WIDTH);
				}
			}

			/*
			** draw remainder of dynamic lightmaps that haven't been uploaded yet
			*/
			if ( newdrawsurf != null )
				LM_UploadBlock( true );

			for ( surf = newdrawsurf; surf != null; surf = surf.lightmapchain )
			{
				if ( surf.polys != null )
					DrawGLPolyChain( surf.polys, ( surf.light_s - surf.dlight_s ) * ( 1.0f / 128.0f ), ( surf.light_t - surf.dlight_t ) * ( 1.0f / 128.0f ) );
			}
		}

		/*
		** restore state
		*/
		gl.glDisable(GL_BLEND);
		gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		gl.glDepthMask( true );
	}
	
	private final IntBuffer temp2 = Lib.newIntBuffer(34 * 34, ByteOrder.LITTLE_ENDIAN);

	/*
	================
	R_RenderBrushPoly
	================
	*/
	private void R_RenderBrushPoly(msurface_t fa)
	{

        c_brush_polys++;

        image_t image = R_TextureAnimation(fa.texinfo);

		if ((fa.flags & Defines.SURF_DRAWTURB) != 0)
		{	
			GL_Bind( image.texnum );

			
			GL_TexEnv( GL_MODULATE );
			gl.glColor4f( gl_state.inverse_intensity, 
						gl_state.inverse_intensity,
						gl_state.inverse_intensity,
						1.0F );
			EmitWaterPolys (fa);
			GL_TexEnv( GL_REPLACE );

			return;
		}
		else
		{
			GL_Bind( image.texnum );
			GL_TexEnv( GL_REPLACE );
		}

		
		
		if((fa.texinfo.flags & Defines.SURF_FLOWING) != 0)
			DrawGLFlowingPoly(fa);
		else
			DrawGLPoly (fa.polys);
		
		

		
		boolean gotoDynamic = false;
		/*
		** check for lightmap modification
		*/
        int maps;
        for (maps = 0; maps < Defines.MAXLIGHTMAPS && fa.styles[maps] != (byte)255; maps++ )
		{
			if ( r_newrefdef.lightstyles[fa.styles[maps] & 0xFF].white != fa.cached_light[maps] ) {
				gotoDynamic = true;
				break;
			}
		}
		
		
		if (maps == 4) maps--;


        boolean is_dynamic = false;
        if ( gotoDynamic || ( fa.dlightframe == r_framecount ) )
		{
			
			if ( gl_dynamic.value != 0 )
			{
				if (( fa.texinfo.flags & (Defines.SURF_SKY | Defines.SURF_TRANS33 | Defines.SURF_TRANS66 | Defines.SURF_WARP ) ) == 0)
				{
					is_dynamic = true;
				}
			}
		}

		if ( is_dynamic )
		{
			if ( ( (fa.styles[maps] & 0xFF) >= 32 || fa.styles[maps] == 0 ) && ( fa.dlightframe != r_framecount ) )
			{

                int smax = (fa.extents[0] >> 4) + 1;
                int tmax = (fa.extents[1] >> 4) + 1;

				R_BuildLightMap( fa, temp2, smax);
				R_SetCacheState( fa );

				GL_Bind( gl_state.lightmap_textures + fa.lightmaptexturenum );

				gl.glTexSubImage2D( GL_TEXTURE_2D, 0,
								  fa.light_s, fa.light_t, 
								  smax, tmax, 
								  GL_LIGHTMAP_FORMAT, 
								  GL_UNSIGNED_BYTE, temp2 );

				fa.lightmapchain = gl_lms.lightmap_surfaces[fa.lightmaptexturenum];
				gl_lms.lightmap_surfaces[fa.lightmaptexturenum] = fa;
			}
			else
			{
				fa.lightmapchain = gl_lms.lightmap_surfaces[0];
				gl_lms.lightmap_surfaces[0] = fa;
			}
		}
		else
		{
			fa.lightmapchain = gl_lms.lightmap_surfaces[fa.lightmaptexturenum];
			gl_lms.lightmap_surfaces[fa.lightmaptexturenum] = fa;
		}
	}


	/*
	================
	R_DrawAlphaSurfaces

	Draw water surfaces and windows.
	The BSP tree is waled front to back, so unwinding the chain
	of alpha_surfaces will draw back to front, giving proper ordering.
	================
	*/
    @Override
    void R_DrawAlphaSurfaces()
	{


        r_world_matrix.clear();
		gl.glLoadMatrix(r_world_matrix);

		gl.glEnable (GL_BLEND);
		GL_TexEnv(GL_MODULATE );


        float intens = gl_state.inverse_intensity;

		for (msurface_t s = r_alpha_surfaces; s != null ; s=s.texturechain)
		{
			GL_Bind(s.texinfo.image.texnum);
			c_brush_polys++;
			if ((s.texinfo.flags & Defines.SURF_TRANS33) != 0)
				gl.glColor4f (intens, intens, intens, 0.33f);
			else if ((s.texinfo.flags & Defines.SURF_TRANS66) != 0)
				gl.glColor4f (intens, intens, intens, 0.66f);
			else
				gl.glColor4f (intens,intens,intens,1);
			if ((s.flags & Defines.SURF_DRAWTURB) != 0)
				EmitWaterPolys(s);
			else if((s.texinfo.flags & Defines.SURF_FLOWING) != 0)			
				DrawGLFlowingPoly (s);							
			else
				DrawGLPoly (s.polys);
		}

		GL_TexEnv( GL_REPLACE );
		gl.glColor4f (1,1,1,1);
		gl.glDisable (GL_BLEND);

		r_alpha_surfaces = null;
	}

	/*
	================
	DrawTextureChains
	================
	*/
	private void DrawTextureChains()
	{

		int c_visible_textures = 0;


        image_t image;
        msurface_t s;
        int i;
        if ( !qglSelectTextureSGIS && !qglActiveTextureARB )
		{
			for (i = 0; i < numgltextures ; i++)
			{
				image = gltextures[i];
				if (image.registration_sequence == 0)
					continue;
				s = image.texturechain;
				if (s == null)
					continue;
				c_visible_textures++;

				for ( ; s != null ; s=s.texturechain)
					R_RenderBrushPoly(s);

				image.texturechain = null;
			}
		}
		else
		{
			for (i = 0; i < numgltextures ; i++)
			{
				image = gltextures[i];

				if (image.registration_sequence == 0)
					continue;
				if (image.texturechain == null)
					continue;
				c_visible_textures++;

				for ( s = image.texturechain; s != null ; s=s.texturechain)
				{
					if ( ( s.flags & Defines.SURF_DRAWTURB) == 0 )
						R_RenderBrushPoly(s);
				}
			}

			GL_EnableMultitexture( false );
			for (i = 0; i < numgltextures ; i++)
			{
				image = gltextures[i];

				if (image.registration_sequence == 0)
					continue;
				s = image.texturechain;
				if (s == null)
					continue;

				for ( ; s != null ; s=s.texturechain)
				{
					if ( (s.flags & Defines.SURF_DRAWTURB) != 0 )
						R_RenderBrushPoly(s);
				}

				image.texturechain = null;
			}
		}

		GL_TexEnv( GL_REPLACE );
	}

	
	private final IntBuffer temp = Lib.newIntBuffer(128 * 128, ByteOrder.LITTLE_ENDIAN);

	private void GL_RenderLightmappedPoly(msurface_t surf)
	{
		int nv = surf.polys.numverts;
		int map;
		image_t image = R_TextureAnimation( surf.texinfo );
        int lmtex = surf.lightmaptexturenum;


        boolean gotoDynamic = false;

		for ( map = 0; map < Defines.MAXLIGHTMAPS && (surf.styles[map] != (byte)255); map++ )
		{
			if ( r_newrefdef.lightstyles[surf.styles[map] & 0xFF].white != surf.cached_light[map] ) {
				gotoDynamic = true;
				break;
			}
		}

		
		if (map == 4) map--;


        boolean is_dynamic = false;
        if ( gotoDynamic || ( surf.dlightframe == r_framecount ) )
		{
			
			if ( gl_dynamic.value != 0 )
			{
				if ( (surf.texinfo.flags & (Defines.SURF_SKY | Defines.SURF_TRANS33 | Defines.SURF_TRANS66 | Defines.SURF_WARP )) == 0 )
				{
					is_dynamic = true;
				}
			}
		}

        glpoly_t p;
        if ( is_dynamic )
		{
			
			int smax, tmax;

			if ( ( (surf.styles[map] & 0xFF) >= 32 || surf.styles[map] == 0 ) && ( surf.dlightframe != r_framecount ) )
			{
				smax = (surf.extents[0]>>4)+1;
				tmax = (surf.extents[1]>>4)+1;

				R_BuildLightMap( surf, temp, smax);
				R_SetCacheState( surf );

				GL_MBind( GL_TEXTURE1, gl_state.lightmap_textures + surf.lightmaptexturenum );

				lmtex = surf.lightmaptexturenum;

			}
			else
			{
				smax = (surf.extents[0]>>4)+1;
				tmax = (surf.extents[1]>>4)+1;

				R_BuildLightMap( surf, temp, smax);

				GL_MBind( GL_TEXTURE1, gl_state.lightmap_textures + 0 );

				lmtex = 0;

			}
			gl.glTexSubImage2D( GL_TEXTURE_2D, 0,
							  surf.light_s, surf.light_t,
							  smax, tmax,
							  GL_LIGHTMAP_FORMAT,
							  GL_UNSIGNED_BYTE, temp );

			c_brush_polys++;

			GL_MBind( GL_TEXTURE0, image.texnum );
			GL_MBind( GL_TEXTURE1, gl_state.lightmap_textures + lmtex );

			
			
			if ((surf.texinfo.flags & Defines.SURF_FLOWING) != 0)
			{

                float scroll = -64 * ((r_newrefdef.time / 40.0f) - (int) (r_newrefdef.time / 40.0f));
				if(scroll == 0.0f)
					scroll = -64.0f;

				for ( p = surf.polys; p != null; p = p.chain )
				{
					gl.glBegin (GL_POLYGON);
					for (int i=0 ; i< nv; i++)
					{
						gl.glMultiTexCoord2f(GL_TEXTURE0, p.s1(i) + scroll, p.t1(i));
						gl.glMultiTexCoord2f(GL_TEXTURE1, p.s2(i), p.t2(i));
						
						
						gl.glVertex3f(p.x(i), p.y(i), p.z(i));
					}
					gl.glEnd ();
				}
			}
			else
			{
				for ( p = surf.polys; p != null; p = p.chain )
				{
					gl.glBegin (GL_POLYGON);
					for (int i=0 ; i< nv; i++)
					{
						gl.glMultiTexCoord2f(GL_TEXTURE0, p.s1(i), p.t1(i));
						gl.glMultiTexCoord2f(GL_TEXTURE1, p.s2(i), p.t2(i));
						
						
						gl.glVertex3f(p.x(i), p.y(i), p.z(i));
					}
					gl.glEnd ();
				}
			}
			
			
		}
		else
		{
			c_brush_polys++;

			GL_MBind( GL_TEXTURE0, image.texnum );
			GL_MBind( GL_TEXTURE1, gl_state.lightmap_textures + lmtex);
			
			
			
			if ((surf.texinfo.flags & Defines.SURF_FLOWING) != 0)
			{

                float scroll = -64 * ((r_newrefdef.time / 40.0f) - (int) (r_newrefdef.time / 40.0f));
				if(scroll == 0.0)
					scroll = -64.0f;

				for ( p = surf.polys; p != null; p = p.chain )
				{
					gl.glBegin(GL_POLYGON);
					for (int i=0 ; i< nv; i++)
					{
						gl.glMultiTexCoord2f(GL_TEXTURE0, p.s1(i) + scroll, p.t1(i));
						gl.glMultiTexCoord2f(GL_TEXTURE1, p.s2(i), p.t2(i));
						
						
						gl.glVertex3f(p.x(i), p.y(i), p.z(i));
					}
					gl.glEnd();
				}
			}
			else
			{
			
			
				for ( p = surf.polys; p != null; p = p.chain )
				{
					gl.glBegin (GL_POLYGON);
					for (int i=0 ; i< nv; i++)
					{
						gl.glMultiTexCoord2f(GL_TEXTURE0, p.s1(i), p.t1(i));
						gl.glMultiTexCoord2f(GL_TEXTURE1, p.s2(i), p.t2(i));
						
						
						gl.glVertex3f(p.x(i), p.y(i), p.z(i));
					}
					gl.glEnd ();
				}
			
			
			}
			
			
		}
	}

	/*
	=================
	R_DrawInlineBModel
	=================
	*/
	private void R_DrawInlineBModel()
	{


        if ( gl_flashblend.value == 0 )
		{
			for (int k = 0; k<r_newrefdef.num_dlights ; k++)
			{
                dlight_t lt = r_newrefdef.dlights[k];
                R_MarkLights(lt, 1<<k, currentmodel.nodes[currentmodel.firstnode]);
			}
		}

		
		int psurfp = currentmodel.firstmodelsurface;
        msurface_t[] surfaces = currentmodel.surfaces;
		

		if ( (currententity.flags & Defines.RF_TRANSLUCENT) != 0 )
		{
			gl.glEnable (GL_BLEND);
			gl.glColor4f (1,1,1,0.25f);
			GL_TexEnv( GL_MODULATE );
		}

		
		
		
		for (int i = 0; i<currentmodel.nummodelsurfaces ; i++)
		{
            msurface_t psurf = surfaces[psurfp++];

            cplane_t pplane = psurf.plane;

            float dot = Math3D.DotProduct(modelorg, pplane.normal) - pplane.dist;


            if (((psurf.flags & Defines.SURF_PLANEBACK) != 0 && (dot < -BACKFACE_EPSILON)) ||
				((psurf.flags & Defines.SURF_PLANEBACK) == 0 && (dot > BACKFACE_EPSILON)))
			{
				if ((psurf.texinfo.flags & (Defines.SURF_TRANS33 | Defines.SURF_TRANS66)) != 0 )
				{	
					psurf.texturechain = r_alpha_surfaces;
					r_alpha_surfaces = psurf;
				}
				else if ( qglMTexCoord2fSGIS && ( psurf.flags & Defines.SURF_DRAWTURB ) == 0 )
				{
					GL_RenderLightmappedPoly( psurf );
				}
				else
				{
					GL_EnableMultitexture( false );
					R_RenderBrushPoly( psurf );
					GL_EnableMultitexture( true );
				}
			}
		}

		if ( (currententity.flags & Defines.RF_TRANSLUCENT) == 0 )
		{
			if ( !qglMTexCoord2fSGIS )
				R_BlendLightmaps();
		}
		else
		{
			gl.glDisable (GL_BLEND);
			gl.glColor4f (1,1,1,1);
			GL_TexEnv( GL_REPLACE );
		}
	}

	/*
	=================
	R_DrawBrushModel
	=================
	*/
    @Override
    void R_DrawBrushModel(entity_t e)
	{

        if (currentmodel.nummodelsurfaces == 0)
			return;

		currententity = e;
		gl_state.currenttextures[0] = gl_state.currenttextures[1] = -1;

		boolean b = IntStream.of(0, 1, 2).anyMatch(v -> e.angles[v] != 0);
        boolean rotated;
        float[] maxs = {0, 0, 0};
        float[] mins = {0, 0, 0};
        if (b)
		{
			rotated = true;
			for (int i = 0; i<3 ; i++)
			{
				mins[i] = e.origin[i] - currentmodel.radius;
				maxs[i] = e.origin[i] + currentmodel.radius;
				
			}
		}
		else
		{
			rotated = false;
			Math3D.VectorAdd(e.origin, currentmodel.mins, mins);
			Math3D.VectorAdd(e.origin, currentmodel.maxs, maxs);
			
		}

		if (R_CullBox(mins, maxs)) {
			
			return;
		}

		gl.glColor3f (1,1,1);
		
		
		gl_lms.clearLightmapSurfaces();
		
		Math3D.VectorSubtract (r_newrefdef.vieworg, e.origin, modelorg);
		if (rotated)
		{
			float[] temp = {0, 0, 0};

            Math3D.VectorCopy (modelorg, temp);
            float[] up = {0, 0, 0};
            float[] right = {0, 0, 0};
            float[] forward = {0, 0, 0};
            Math3D.AngleVectors (e.angles, forward, right, up);
			modelorg[0] = Math3D.DotProduct (temp, forward);
			modelorg[1] = -Math3D.DotProduct (temp, right);
			modelorg[2] = Math3D.DotProduct (temp, up);
		}

		gl.glPushMatrix();
		
		e.angles[0] = -e.angles[0];	
		e.angles[2] = -e.angles[2];	
		R_RotateForEntity(e);
		e.angles[0] = -e.angles[0];	
		e.angles[2] = -e.angles[2];	

		GL_EnableMultitexture( true );
		GL_SelectTexture(GL_TEXTURE0);
		GL_TexEnv( GL_REPLACE );
		GL_SelectTexture(GL_TEXTURE1);
		GL_TexEnv( GL_MODULATE );

		R_DrawInlineBModel();
		GL_EnableMultitexture( false );

		gl.glPopMatrix();
	}

	/*
	=============================================================

		WORLD MODEL

	=============================================================
	*/

	/*
	================
	R_RecursiveWorldNode
	================
	*/
	private void R_RecursiveWorldNode(mnode_t node)
	{

        if (node.contents == Defines.CONTENTS_SOLID)
			return;		
		
		if (node.visframe != r_visframecount)
			return;
			
		if (R_CullBox(node.mins, node.maxs))
			return;


        int c;
        if (node.contents != -1)
		{
            mleaf_t pleaf = (mleaf_t) node;


            if (r_newrefdef.areabits != null)
			{
				if ( ((r_newrefdef.areabits[pleaf.area >> 3] & 0xFF) & (1 << (pleaf.area & 7)) ) == 0 )
					return;		
			}

			int markp = 0;

            msurface_t mark = pleaf.getMarkSurface(markp);
            c = pleaf.nummarksurfaces;

			if (c != 0)
			{
				do
				{
					mark.visframe = r_framecount;
					mark = pleaf.getMarkSurface(++markp); 
				} while (--c != 0);
			}

			return;
		}


        cplane_t plane = node.plane;

        float dot = switch (plane.type) {
            case Defines.PLANE_X -> modelorg[0] - plane.dist;
            case Defines.PLANE_Y -> modelorg[1] - plane.dist;
            case Defines.PLANE_Z -> modelorg[2] - plane.dist;
            default -> Math3D.DotProduct(modelorg, plane.normal) - plane.dist;
        };

        int sidebit;
        int side;
        if (dot >= 0.0f)
		{
			side = 0;
			sidebit = 0;
		}
		else
		{
			side = 1;
			sidebit = Defines.SURF_PLANEBACK;
		}

		
		R_RecursiveWorldNode(node.children[side]);

		
		
		for ( c = 0; c < node.numsurfaces; c++)
		{
            msurface_t surf = r_worldmodel.surfaces[node.firstsurface + c];
            if (surf.visframe != r_framecount)
				continue;

			if ( (surf.flags & Defines.SURF_PLANEBACK) != sidebit )
				continue;		

			if ((surf.texinfo.flags & Defines.SURF_SKY) != 0)
			{	
				R_AddSkySurface(surf);
			}
			else if ((surf.texinfo.flags & (Defines.SURF_TRANS33 | Defines.SURF_TRANS66)) != 0)
			{	
				surf.texturechain = r_alpha_surfaces;
				r_alpha_surfaces = surf;
			}
			else
			{
				if ( qglMTexCoord2fSGIS && ( surf.flags & Defines.SURF_DRAWTURB) == 0 )
				{
					GL_RenderLightmappedPoly( surf );
				}
				else
				{


                    image_t image = R_TextureAnimation(surf.texinfo);
                    surf.texturechain = image.texturechain;
					image.texturechain = surf;
				}
			}
		}

		
		R_RecursiveWorldNode(node.children[1 - side]);
	}


	/*
	=============
	R_DrawWorld
	=============
	*/
    @Override
    void R_DrawWorld()
	{
		entity_t	ent = new entity_t();
		
		ent.frame = (int)(r_newrefdef.time*2);
		currententity = ent;
		
		if (r_drawworld.value == 0)
			return;

		if ( (r_newrefdef.rdflags & Defines.RDF_NOWORLDMODEL) != 0 )
			return;

		currentmodel = r_worldmodel;

		Math3D.VectorCopy(r_newrefdef.vieworg, modelorg);

		gl_state.currenttextures[0] = gl_state.currenttextures[1] = -1;

		gl.glColor3f (1,1,1);
		
		gl_lms.clearLightmapSurfaces();
		
		R_ClearSkyBox();

		if ( qglMTexCoord2fSGIS )
		{
			GL_EnableMultitexture( true );

			GL_SelectTexture( GL_TEXTURE0);
			GL_TexEnv( GL_REPLACE );
			GL_SelectTexture( GL_TEXTURE1);

			if ( gl_lightmap.value != 0)
				GL_TexEnv( GL_REPLACE );
			else 
				GL_TexEnv( GL_MODULATE );

			R_RecursiveWorldNode(r_worldmodel.nodes[0]); 

			GL_EnableMultitexture( false );
		}
		else
		{
			R_RecursiveWorldNode(r_worldmodel.nodes[0]); 
		}

		/*
		** theoretically nothing should happen in the next two functions
		** if multitexture is enabled
		*/
		DrawTextureChains();
		R_BlendLightmaps();
	
		R_DrawSkyBox();

		R_DrawTriangleOutlines();
	}

	private final byte[] fatvis = new byte[Defines.MAX_MAP_LEAFS / 8];

	/*
	===============
	R_MarkLeaves

	Mark the leaves and nodes that are in the PVS for the current
	cluster
	===============
	*/
    @Override
    void R_MarkLeaves()
	{


        Arrays.fill(fatvis, (byte)0);

        if (r_oldviewcluster == r_viewcluster && r_oldviewcluster2 == r_viewcluster2 && r_novis.value == 0 && r_viewcluster != -1)
			return;

		
		
		if (gl_lockpvs.value != 0)
			return;

		r_visframecount++;
		r_oldviewcluster = r_viewcluster;
		r_oldviewcluster2 = r_viewcluster2;

        int i;
        if (r_novis.value != 0 || r_viewcluster == -1 || r_worldmodel.vis == null)
		{
			
			for (i=0 ; i<r_worldmodel.numleafs ; i++)
				r_worldmodel.leafs[i].visframe = r_visframecount;
			for (i=0 ; i<r_worldmodel.numnodes ; i++)
				r_worldmodel.nodes[i].visframe = r_visframecount;
			return;
		}

        byte[] vis = Mod_ClusterPVS(r_viewcluster, r_worldmodel);
		
		if (r_viewcluster2 != r_viewcluster)
		{
			
			System.arraycopy(vis, 0, fatvis, 0, (r_worldmodel.numleafs+7) / 8);
			vis = Mod_ClusterPVS(r_viewcluster2, r_worldmodel);
            int c = (r_worldmodel.numleafs + 31) / 32;
            int k = 0;
			for (i=0 ; i<c ; i++) {
				fatvis[k] |= vis[k++];
				fatvis[k] |= vis[k++];
				fatvis[k] |= vis[k++];
				fatvis[k] |= vis[k++];
			}

			vis = fatvis;
		}
	
		for ( i=0; i < r_worldmodel.numleafs; i++)
		{
            mleaf_t leaf = r_worldmodel.leafs[i];
            int cluster = leaf.cluster;
            if (cluster == -1)
				continue;
			if (((vis[cluster>>3] & 0xFF) & (1 << (cluster & 7))) != 0)
			{
                mnode_t node = leaf;
                do
				{
					if (node.visframe == r_visframecount)
						break;
					node.visframe = r_visframecount;
					node = node.parent;
				} while (node != null);
			}
		}
	}



	/*
	=============================================================================

	  LIGHTMAP ALLOCATION

	=============================================================================
	*/

	private void LM_InitBlock()
	{
		Arrays.fill(gl_lms.allocated, 0);
	}

	private void LM_UploadBlock(boolean dynamic)
	{
		int texture;

        if ( dynamic )
		{
			texture = 0;
		}
		else
		{
			texture = gl_lms.current_lightmap_texture;
		}

		GL_Bind( gl_state.lightmap_textures + texture );
		gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        gl_lms.lightmap_buffer.rewind();
		if ( dynamic )
		{

            int height = 0;
            for (int i = 0; i < BLOCK_WIDTH; i++ )
			{
				if ( gl_lms.allocated[i] > height )
					height = gl_lms.allocated[i];
			}

			gl.glTexSubImage2D( GL_TEXTURE_2D, 
							  0,
							  0, 0,
							  BLOCK_WIDTH, height,
							  GL_LIGHTMAP_FORMAT,
							  GL_UNSIGNED_BYTE,
							  gl_lms.lightmap_buffer );
		}
		else
		{
			gl.glTexImage2D( GL_TEXTURE_2D, 
						   0, 
						   gl_lms.internal_format,
						   BLOCK_WIDTH, BLOCK_HEIGHT, 
						   0, 
						   GL_LIGHTMAP_FORMAT, 
						   GL_UNSIGNED_BYTE, 
						   gl_lms.lightmap_buffer );
			if ( ++gl_lms.current_lightmap_texture == MAX_LIGHTMAPS )
				Com.Error( Defines.ERR_DROP, "LM_UploadBlock() - MAX_LIGHTMAPS exceeded\n" );
				
				
			

		}
	}

	
	private boolean LM_AllocBlock(int w, int h, pos_t pos)
	{
		int x = pos.x; 
		int i;

        int best = BLOCK_HEIGHT;

		for (i=0 ; i<BLOCK_WIDTH-w ; i++)
		{
            int best2 = 0;

            int j;
            for (j=0 ; j<w ; j++)
			{
				if (gl_lms.allocated[i+j] >= best)
					break;
				if (gl_lms.allocated[i+j] > best2)
					best2 = gl_lms.allocated[i+j];
			}
			if (j == w)
			{	
				pos.x = x = i;
				pos.y = best = best2;
			}
		}

		if (best + h > BLOCK_HEIGHT)
			return false;

		for (i=0 ; i<w ; i++)
			gl_lms.allocated[x + i] = best + h;

		return true;
	}

	/*
	================
	GL_BuildPolygonFromSurface
	================
	*/
	void GL_BuildPolygonFromSurface(msurface_t fa)
	{
        float[] total = {0, 0, 0};


        medge_t[] pedges = currentmodel.edges;
        int lnumverts = fa.numedges;
		Math3D.VectorClear(total);


        glpoly_t poly = Polygon.create(lnumverts);

		poly.next = fa.polys;
		poly.flags = fa.flags;
		fa.polys = poly;

		for (int i=0 ; i<lnumverts ; i++)
		{
            int lindex = currentmodel.surfedges[fa.firstedge + i];

            float[] vec;
            medge_t r_pedge;
            if (lindex > 0)
			{
				r_pedge = pedges[lindex];
				vec = currentmodel.vertexes[r_pedge.v[0]].position;
			}
			else
			{
				r_pedge = pedges[-lindex];
				vec = currentmodel.vertexes[r_pedge.v[1]].position;
			}
            float s = Math3D.DotProduct(vec, fa.texinfo.vecs[0]) + fa.texinfo.vecs[0][3];
            s /= fa.texinfo.image.width;

            float t = Math3D.DotProduct(vec, fa.texinfo.vecs[1]) + fa.texinfo.vecs[1][3];
            t /= fa.texinfo.image.height;

			Math3D.VectorAdd (total, vec, total);
			
			poly.x(i, vec[0]);
			poly.y(i, vec[1]);
			poly.z(i, vec[2]);
			
			
			
			poly.s1(i, s);
			poly.t1(i, t);

			
			
			
			s = Math3D.DotProduct (vec, fa.texinfo.vecs[0]) + fa.texinfo.vecs[0][3];
			s -= fa.texturemins[0];
			s += fa.light_s*16;
			s += 8;
			s /= BLOCK_WIDTH*16; 

			t = Math3D.DotProduct (vec, fa.texinfo.vecs[1]) + fa.texinfo.vecs[1][3];
			t -= fa.texturemins[1];
			t += fa.light_t*16;
			t += 8;
			t /= BLOCK_HEIGHT*16; 

			
			
			poly.s2(i, s);
			poly.t2(i, t);
		}
	}

	/*
	========================
	GL_CreateSurfaceLightmap
	========================
	*/
	void GL_CreateSurfaceLightmap(msurface_t surf)
	{

        if ( (surf.flags & (Defines.SURF_DRAWSKY | Defines.SURF_DRAWTURB)) != 0)
			return;

        int smax = (surf.extents[0] >> 4) + 1;
        int tmax = (surf.extents[1] >> 4) + 1;
		
		pos_t lightPos = new pos_t(surf.light_s, surf.light_t);

		if ( !LM_AllocBlock( smax, tmax, lightPos ) )
		{
			LM_UploadBlock( false );
			LM_InitBlock();
			lightPos = new pos_t(surf.light_s, surf.light_t);
			if ( !LM_AllocBlock( smax, tmax, lightPos ) )
			{
				Com.Error( Defines.ERR_FATAL, "Consecutive calls to LM_AllocBlock(" + smax + ',' + tmax +") failed\n");
			}
		}
		
		
		surf.light_s = lightPos.x;
		surf.light_t = lightPos.y;

		surf.lightmaptexturenum = gl_lms.current_lightmap_texture;
		
		int basep = (surf.light_t * BLOCK_WIDTH + surf.light_s);
        IntBuffer base = gl_lms.lightmap_buffer;
		base.position(basep);

		R_SetCacheState( surf );
		R_BuildLightMap(surf, base.slice(), BLOCK_WIDTH);
	}

	private lightstyle_t[] lightstyles;
    private final IntBuffer dummy = Lib.newIntBuffer(128*128);
	/*
	==================
	GL_BeginBuildingLightmaps

	==================
	*/
	void GL_BeginBuildingLightmaps(model_t m)
	{
		
		int i;
		
		if ( lightstyles == null ) {
			lightstyles = new lightstyle_t[Defines.MAX_LIGHTSTYLES];
			for (i = 0; i < lightstyles.length; i++)
			{
				lightstyles[i] = new lightstyle_t();				
			}
		}

		
		Arrays.fill(gl_lms.allocated, 0);

		r_framecount = 1;		

		GL_EnableMultitexture( true );
		GL_SelectTexture( GL_TEXTURE1);

		/*
		** setup the base lightstyles so the lightmaps won't have to be regenerated
		** the first time they're seen
		*/
		for (i=0 ; i < Defines.MAX_LIGHTSTYLES ; i++)
		{
			lightstyles[i].rgb[0] = 1;
			lightstyles[i].rgb[1] = 1;
			lightstyles[i].rgb[2] = 1;
			lightstyles[i].white = 3;
		}
		r_newrefdef.lightstyles = lightstyles;

		if (gl_state.lightmap_textures == 0)
		{
			gl_state.lightmap_textures = TEXNUM_LIGHTMAPS;
		}

		gl_lms.current_lightmap_texture = 1;

		/*
		** if mono lightmaps are enabled and we want to use alpha
		** blending (a,1-a) then we're likely running on a 3DLabs
		** Permedia2.  In a perfect world we'd use a GL_ALPHA lightmap
		** in order to conserve space and maximize bandwidth, however 
		** this isn't a perfect world.
		**
		** So we have to use alpha lightmaps, but stored in GL_RGBA format,
		** which means we only get 1/16th the color resolution we should when
		** using alpha lightmaps.  If we find another board that supports
		** only alpha lightmaps but that can at least support the GL_ALPHA
		** format then we should change this code to use real alpha maps.
		*/
		
		char format = gl_monolightmap.string.toUpperCase().charAt(0);

		/*
		 ** try to do hacked colored lighting with a blended texture
		 */
        gl_lms.internal_format = switch (format) {
            case 'A', 'C' -> gl_tex_alpha_format;
            case 'I' -> GL_INTENSITY8;
            case 'L' -> GL_LUMINANCE8;
            default -> gl_tex_solid_format;
        };

		/*
		** initialize the dynamic lightmap texture
		*/
		GL_Bind( gl_state.lightmap_textures + 0 );
		gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		gl.glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		gl.glTexImage2D( GL_TEXTURE_2D, 
					   0, 
					   gl_lms.internal_format,
					   BLOCK_WIDTH, BLOCK_HEIGHT, 
					   0, 
					   GL_LIGHTMAP_FORMAT, 
					   GL_UNSIGNED_BYTE, 
					   dummy );
	}

	/*
	=======================
	GL_EndBuildingLightmaps
	=======================
	*/
	void GL_EndBuildingLightmaps()
	{
		LM_UploadBlock( false );
		GL_EnableMultitexture( false );
	}
	
	
	
	




















}
