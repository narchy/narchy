/*
 * Copyright (C) 1997-2001 Id Software, Inc.
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.
 * 
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place - Suite 330, Boston, MA 02111-1307, USA.
 *  
 */


package jake2.qcommon;

import jake2.Defines;
import jake2.Globals;
import jake2.game.*;
import jake2.util.Lib;
import jake2.util.Math3D;
import jake2.util.Vargs;
import jake2.util.Vec3Cache;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

public class CM {

    static class cnode_t {
        cplane_t plane; 

        final int[] children = { 0, 0 }; 
    }

    static class cbrushside_t {
        cplane_t plane; 

        mapsurface_t surface; 
    }

    static class cleaf_t {
        int contents;

        int cluster;

        int area;

        
        short firstleafbrush;

        
        short numleafbrushes;
    }

    static class cbrush_t {
        int contents;

        int numsides;

        int firstbrushside;

        int checkcount; 
    }

    static class carea_t {
        int numareaportals;

        int firstareaportal;

        int floodnum; 

        int floodvalid;
    }

    private static int checkcount;

    private static String map_name = "";

    private static int numbrushsides;

    private static final cbrushside_t[] map_brushsides = new cbrushside_t[Defines.MAX_MAP_BRUSHSIDES];
    static {
        for (int n = 0; n < Defines.MAX_MAP_BRUSHSIDES; n++)
            map_brushsides[n] = new cbrushside_t();
    }

    public static int numtexinfo;

    public static final mapsurface_t[] map_surfaces = new mapsurface_t[Defines.MAX_MAP_TEXINFO];
    static {
        for (int n = 0; n < Defines.MAX_MAP_TEXINFO; n++)
            map_surfaces[n] = new mapsurface_t();
    }

    private static int numplanes;

    /** Extra for box hull ( +6) */
    private static final cplane_t[] map_planes = new cplane_t[Defines.MAX_MAP_PLANES + 6];

    static {
        for (int n = 0; n < Defines.MAX_MAP_PLANES + 6; n++)
            map_planes[n] = new cplane_t();
    }

    private static int numnodes;

    /** Extra for box hull ( +6) */
    private static final cnode_t[] map_nodes = new cnode_t[Defines.MAX_MAP_NODES + 6];

    static {
        for (int n = 0; n < Defines.MAX_MAP_NODES + 6; n++)
            map_nodes[n] = new cnode_t();
    }

    private static int numleafs = 1;

    private static final cleaf_t[] map_leafs = new cleaf_t[Defines.MAX_MAP_LEAFS];
    static {
        for (int n = 0; n < Defines.MAX_MAP_LEAFS; n++)
            map_leafs[n] = new cleaf_t();
    }

    private static int emptyleaf;

    private static int numleafbrushes;

    private static final int[] map_leafbrushes = new int[Defines.MAX_MAP_LEAFBRUSHES];

    private static int numcmodels;

    private static final cmodel_t[] map_cmodels = new cmodel_t[Defines.MAX_MAP_MODELS];
    static {
        for (int n = 0; n < Defines.MAX_MAP_MODELS; n++)
            map_cmodels[n] = new cmodel_t();

    }

    private static int numbrushes;

    private static final cbrush_t[] map_brushes = new cbrush_t[Defines.MAX_MAP_BRUSHES];
    static {
        for (int n = 0; n < Defines.MAX_MAP_BRUSHES; n++)
            map_brushes[n] = new cbrush_t();

    }

    private static int numvisibility;

    private static final byte[] map_visibility = new byte[Defines.MAX_MAP_VISIBILITY];

    /** Main visibility data. */
    private static qfiles.dvis_t map_vis = new qfiles.dvis_t(ByteBuffer
            .wrap(map_visibility));

    private static int numentitychars;

    private static String map_entitystring;

    private static int numareas = 1;

    private static final carea_t[] map_areas = new carea_t[Defines.MAX_MAP_AREAS];
    static {
        for (int n = 0; n < Defines.MAX_MAP_AREAS; n++)
            map_areas[n] = new carea_t();

    }

    private static int numareaportals;

    private static final qfiles.dareaportal_t[] map_areaportals = new qfiles.dareaportal_t[Defines.MAX_MAP_AREAPORTALS];

    static {
        for (int n = 0; n < Defines.MAX_MAP_AREAPORTALS; n++)
            map_areaportals[n] = new qfiles.dareaportal_t();

    }

    private static int numclusters = 1;

    private static final mapsurface_t nullsurface = new mapsurface_t();

    private static int floodvalid;

    private static final boolean[] portalopen = new boolean[Defines.MAX_MAP_AREAPORTALS];

    private static cvar_t map_noareas;

    private static byte[] cmod_base;

    public static int checksum;

    private static int last_checksum;

    /** 
     * Loads in the map and all submodels.
     */
    public static cmodel_t CM_LoadMap(String name, boolean clientload,
                                      int[] checksum) {
        Com.DPrintf("CM_LoadMap(" + name + ")...\n");

        map_noareas = Cvar.Get("map_noareas", "0", 0);

        if (map_name.equals(name)
                && (clientload || 0 == Cvar.VariableValue("flushmap"))) {

            checksum[0] = last_checksum;

            if (!clientload) {
                Arrays.fill(portalopen, false);
                FloodAreaConnections();
            }
            return map_cmodels[0]; 
        }

        
        numnodes = 0;
        numleafs = 0;
        numcmodels = 0;
        numvisibility = 0;
        numentitychars = 0;
        map_entitystring = "";
        map_name = "";

        if (name == null || name.isEmpty()) {
            numleafs = 1;
            numclusters = 1;
            numareas = 1;
            checksum[0] = 0;
            return map_cmodels[0];
            
        }


        byte[] buf = FS.LoadFile(name);

        if (buf == null)
            Com.Error(Defines.ERR_DROP, "Couldn't load " + name);

        int length = buf.length;

        ByteBuffer bbuf = ByteBuffer.wrap(buf);

        last_checksum = MD4.Com_BlockChecksum(buf, length);
        checksum[0] = last_checksum;

        qfiles.dheader_t header = new qfiles.dheader_t(bbuf.slice());

        if (header.version != Defines.BSPVERSION)
            Com.Error(Defines.ERR_DROP, "CMod_LoadBrushModel: " + name
                    + " has wrong version number (" + header.version
                    + " should be " + Defines.BSPVERSION + ')');

        cmod_base = buf;

        
        CMod_LoadSurfaces(header.lumps[Defines.LUMP_TEXINFO]); 
        CMod_LoadLeafs(header.lumps[Defines.LUMP_LEAFS]);
        CMod_LoadLeafBrushes(header.lumps[Defines.LUMP_LEAFBRUSHES]);
        CMod_LoadPlanes(header.lumps[Defines.LUMP_PLANES]);
        CMod_LoadBrushes(header.lumps[Defines.LUMP_BRUSHES]);
        CMod_LoadBrushSides(header.lumps[Defines.LUMP_BRUSHSIDES]);
        CMod_LoadSubmodels(header.lumps[Defines.LUMP_MODELS]);

        CMod_LoadNodes(header.lumps[Defines.LUMP_NODES]);
        CMod_LoadAreas(header.lumps[Defines.LUMP_AREAS]);    
        CMod_LoadAreaPortals(header.lumps[Defines.LUMP_AREAPORTALS]);       
        CMod_LoadVisibility(header.lumps[Defines.LUMP_VISIBILITY]);
        CMod_LoadEntityString(header.lumps[Defines.LUMP_ENTITIES]);
        
        FS.FreeFile(buf);

        CM_InitBoxHull();

        Arrays.fill(portalopen, false);

        FloodAreaConnections();

        map_name = name;

        return map_cmodels[0];
    }

    /** Loads Submodels. */
    private static void CMod_LoadSubmodels(lump_t l) {
        Com.DPrintf("CMod_LoadSubmodels()\n");

        if ((l.filelen % qfiles.dmodel_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "CMod_LoadBmodel: funny lump size");

        int count = l.filelen / qfiles.dmodel_t.SIZE;

        if (count < 1)
            Com.Error(Defines.ERR_DROP, "Map with no models");
        if (count > Defines.MAX_MAP_MODELS)
            Com.Error(Defines.ERR_DROP, "Map has too many models");

        Com.DPrintf(" numcmodels=" + count + '\n');
        numcmodels = count;

        if (debugloadmap) {
            Com.DPrintf("submodles(headnode, <origin>, <mins>, <maxs>)\n");
        }
        for (int i = 0; i < count; i++) {
            qfiles.dmodel_t in = new qfiles.dmodel_t(ByteBuffer.wrap(cmod_base, i
                    * qfiles.dmodel_t.SIZE + l.fileofs, qfiles.dmodel_t.SIZE));
            cmodel_t out = map_cmodels[i];

            for (int j = 0; j < 3; j++) {
                out.mins[j] = in.mins[j] - 1;
                out.maxs[j] = in.maxs[j] + 1;
                out.origin[j] = in.origin[j];
            }
            out.headnode = in.headnode;
            if (debugloadmap) {
                Com.DPrintf(
                	"|%6i|%8.2f|%8.2f|%8.2f|  %8.2f|%8.2f|%8.2f|   %8.2f|%8.2f|%8.2f|\n",
                        new Vargs().add(out.headnode)
                        .add(out.origin[0]).add(out.origin[1]).add(out.origin[2])
                        .add(out.mins[0]).add(out.mins[1]).add(out.mins[2])
                        .add(out.maxs[0]).add(out.maxs[1]).add(out.maxs[2]));
            }
        }
    }

    private static final boolean debugloadmap = false;

    /** Loads surfaces. */
    private static void CMod_LoadSurfaces(lump_t l) {
        Com.DPrintf("CMod_LoadSurfaces()\n");
        texinfo_t in;
        mapsurface_t out;

        if ((l.filelen % texinfo_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");

        int count = l.filelen / texinfo_t.SIZE;
        if (count < 1)
            Com.Error(Defines.ERR_DROP, "Map with no surfaces");
        if (count > Defines.MAX_MAP_TEXINFO)
            Com.Error(Defines.ERR_DROP, "Map has too many surfaces");

        numtexinfo = count;
        Com.DPrintf(" numtexinfo=" + count + '\n');
        if (debugloadmap)
            Com.DPrintf("surfaces:\n");

        //for (int i = 0; i < count; i++)
        IntStream.range(0, count).parallel().forEach(i->
            CMod_LoadSurfaces(l, i)
        );
    }

    private static void CMod_LoadSurfaces(lump_t l, int i) {
        mapsurface_t out = map_surfaces[i] = new mapsurface_t();
        texinfo_t in = new texinfo_t(cmod_base, l.fileofs + i * texinfo_t.SIZE,
                texinfo_t.SIZE);

        out.c.name = in.texture;
        out.rname = in.texture;
        out.c.flags = in.flags;
        out.c.value = in.value;

        if (debugloadmap) {
            Com.DPrintf("|%20s|%20s|%6i|%6i|\n", new Vargs()
                    .add(out.c.name).add(out.rname).add(out.c.value).add(
                            out.c.flags));
        }
    }

    /** Loads nodes. */
    private static void CMod_LoadNodes(lump_t l) {
        Com.DPrintf("CMod_LoadNodes()\n");
        qfiles.dnode_t in;
        int child;

        int j;

        if ((l.filelen % qfiles.dnode_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size:"
                    + l.fileofs + ',' + qfiles.dnode_t.SIZE);
        int count = l.filelen / qfiles.dnode_t.SIZE;

        if (count < 1)
            Com.Error(Defines.ERR_DROP, "Map has no nodes");
        if (count > Defines.MAX_MAP_NODES)
            Com.Error(Defines.ERR_DROP, "Map has too many nodes");

        numnodes = count;
        Com.DPrintf(" numnodes=" + count + '\n');

        if (debugloadmap) {
            Com.DPrintf("nodes(planenum, child[0], child[1])\n");
        }

        //for (int i = 0; i < count; i++)
        IntStream.range(0, count).parallel().forEach(i->
            loadNode(l, i)
        );
    }

    private static void loadNode(lump_t l, int i) {
        qfiles.dnode_t in = new qfiles.dnode_t(ByteBuffer.wrap(cmod_base,
                qfiles.dnode_t.SIZE * i + l.fileofs, qfiles.dnode_t.SIZE));

        cnode_t out = map_nodes[i];

        out.plane = map_planes[in.planenum];
        System.arraycopy(in.children, 0, out.children, 0, 2);
        if (debugloadmap) {
            Com.DPrintf("|%6i| %6i| %6i|\n", new Vargs().add(in.planenum)
                    .add(out.children[0]).add(out.children[1]));
        }
    }

    /** Loads brushes.*/
    private static void CMod_LoadBrushes(lump_t l) {
        Com.DPrintf("CMod_LoadBrushes()\n");
        qfiles.dbrush_t in;

        if ((l.filelen % qfiles.dbrush_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");

        int count = l.filelen / qfiles.dbrush_t.SIZE;

        if (count > Defines.MAX_MAP_BRUSHES)
            Com.Error(Defines.ERR_DROP, "Map has too many brushes");

        numbrushes = count;
        Com.DPrintf(" numbrushes=" + count + '\n');
        if (debugloadmap) {
            Com.DPrintf("brushes:(firstbrushside, numsides, contents)\n");
        }


        //for (int i = 0; i < count; i++)
        IntStream.range(0, count).parallel().forEach(i->
            loadBrush(l, i)
        );
    }

    private static void loadBrush(lump_t l, int i) {
        qfiles.dbrush_t in = new qfiles.dbrush_t(ByteBuffer.wrap(cmod_base, i
                * qfiles.dbrush_t.SIZE + l.fileofs, qfiles.dbrush_t.SIZE));
        cbrush_t out = map_brushes[i];
        out.firstbrushside = in.firstside;
        out.numsides = in.numsides;
        out.contents = in.contents;

        if (debugloadmap) {
            Com.DPrintf("| %6i| %6i| %8X|\n", new Vargs().add(
                out.firstbrushside).add(out.numsides).add(
                    out.contents));
        }
    }

    /** Loads leafs.   */
    private static void CMod_LoadLeafs(lump_t l) {
        Com.DPrintf("CMod_LoadLeafs()\n");

        if ((l.filelen % qfiles.dleaf_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");

        int count = l.filelen / qfiles.dleaf_t.SIZE;

        if (count < 1)
            Com.Error(Defines.ERR_DROP, "Map with no leafs");

        
        if (count > Defines.MAX_MAP_PLANES)
            Com.Error(Defines.ERR_DROP, "Map has too many planes");

        Com.DPrintf(" numleafes=" + count + '\n');

        numleafs = count;
        numclusters = 0;
        if (debugloadmap)
            Com.DPrintf("cleaf-list:(contents, cluster, area, firstleafbrush, numleafbrushes)\n");
        int i;
        for (i = 0; i < count; i++) {
            qfiles.dleaf_t in = new qfiles.dleaf_t(cmod_base, i * qfiles.dleaf_t.SIZE
                    + l.fileofs, qfiles.dleaf_t.SIZE);

            cleaf_t out = map_leafs[i];

            out.contents = in.contents;
            out.cluster = in.cluster;
            out.area = in.area;
            out.firstleafbrush = (short) in.firstleafbrush;
            out.numleafbrushes = (short) in.numleafbrushes;

            if (out.cluster >= numclusters)
                numclusters = out.cluster + 1;

            if (debugloadmap) {
                Com.DPrintf("|%8x|%6i|%6i|%6i|\n", new Vargs()
                        .add(out.contents).add(out.cluster).add(out.area).add(
                                out.firstleafbrush).add(out.numleafbrushes));
            }

        }

        Com.DPrintf(" numclusters=" + numclusters + '\n');

        if (map_leafs[0].contents != Defines.CONTENTS_SOLID)
            Com.Error(Defines.ERR_DROP, "Map leaf 0 is not CONTENTS_SOLID");

        int solidleaf = 0;
        emptyleaf = -1;

        for (i = 1; i < numleafs; i++) {
            if (map_leafs[i].contents == 0) {
                emptyleaf = i;
                break;
            }
        }

        if (emptyleaf == -1)
            Com.Error(Defines.ERR_DROP, "Map does not have an empty leaf");
    }

    /** Loads planes. */
    private static void CMod_LoadPlanes(lump_t l) {
        Com.DPrintf("CMod_LoadPlanes()\n");

        if ((l.filelen % qfiles.dplane_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");

        int count = l.filelen / qfiles.dplane_t.SIZE;

        if (count < 1)
            Com.Error(Defines.ERR_DROP, "Map with no planes");

        
        if (count > Defines.MAX_MAP_PLANES)
            Com.Error(Defines.ERR_DROP, "Map has too many planes");

        Com.DPrintf(" numplanes=" + count + '\n');

        numplanes = count;
        if (debugloadmap) {
            Com
                    .DPrintf("cplanes(normal[0],normal[1],normal[2], dist, type, signbits)\n");
        }

        for (int i = 0; i < count; i++) {
            qfiles.dplane_t in = new qfiles.dplane_t(ByteBuffer.wrap(cmod_base, i
                    * qfiles.dplane_t.SIZE + l.fileofs, qfiles.dplane_t.SIZE));

            cplane_t out = map_planes[i];

            int bits = 0;
            for (int j = 0; j < 3; j++) {
                out.normal[j] = in.normal[j];

                if (out.normal[j] < 0)
                    bits |= 1 << j;
            }

            out.dist = in.dist;
            out.type = (byte) in.type;
            out.signbits = (byte) bits;

            if (debugloadmap) {
                Com.DPrintf("|%6.2f|%6.2f|%6.2f| %10.2f|%3i| %1i|\n",
                        new Vargs().add(out.normal[0]).add(out.normal[1]).add(
                                out.normal[2]).add(out.dist).add(out.type).add(
                                out.signbits));
            }
        }
    }

    /** Loads leaf brushes. */
    private static void CMod_LoadLeafBrushes(lump_t l) {
        Com.DPrintf("CMod_LoadLeafBrushes()\n");

        if ((l.filelen % 2) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");

        int count = l.filelen / 2;

        Com.DPrintf(" numbrushes=" + count + '\n');

        if (count < 1)
            Com.Error(Defines.ERR_DROP, "Map with no planes");

        
        if (count > Defines.MAX_MAP_LEAFBRUSHES)
            Com.Error(Defines.ERR_DROP, "Map has too many leafbrushes");

        int[] out = map_leafbrushes;
        numleafbrushes = count;

        ByteBuffer bb = ByteBuffer.wrap(cmod_base, l.fileofs, count * 2).order(
                ByteOrder.LITTLE_ENDIAN);

        if (debugloadmap) {
            Com.DPrintf("map_brushes:\n");
        }

        for (int i = 0; i < count; i++) {
            out[i] = bb.getShort();
            if (debugloadmap) {
                Com.DPrintf("|%6i|%6i|\n", new Vargs().add(i).add(out[i]));
            }
        }
    }

    /** Loads brush sides. */
    private static void CMod_LoadBrushSides(lump_t l) {
        Com.DPrintf("CMod_LoadBrushSides()\n");

        if ((l.filelen % qfiles.dbrushside_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");
        int count = l.filelen / qfiles.dbrushside_t.SIZE;

        
        if (count > Defines.MAX_MAP_BRUSHSIDES)
            Com.Error(Defines.ERR_DROP, "Map has too many planes");

        numbrushsides = count;

        Com.DPrintf(" numbrushsides=" + count + '\n');

        if (debugloadmap) {
            Com.DPrintf("brushside(planenum, surfacenum):\n");
        }
        for (int i = 0; i < count; i++) {

            qfiles.dbrushside_t in = new qfiles.dbrushside_t(ByteBuffer.wrap(cmod_base, i
                            * qfiles.dbrushside_t.SIZE + l.fileofs,
                    qfiles.dbrushside_t.SIZE));

            cbrushside_t out = map_brushsides[i];

            int num = in.planenum;

            out.plane = map_planes[num];

            int j = in.texinfo;

            if (j >= numtexinfo)
                Com.Error(Defines.ERR_DROP, "Bad brushside texinfo");

            
            if (j == -1)
                out.surface = new mapsurface_t(); 
            else
                out.surface = map_surfaces[j];

            if (debugloadmap) {
                Com.DPrintf("| %6i| %6i|\n", new Vargs().add(num).add(j));
            }
        }
    }

    /** Loads areas. */
    private static void CMod_LoadAreas(lump_t l) {
        Com.DPrintf("CMod_LoadAreas()\n");

        if ((l.filelen % qfiles.darea_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");

        int count = l.filelen / qfiles.darea_t.SIZE;

        if (count > Defines.MAX_MAP_AREAS)
            Com.Error(Defines.ERR_DROP, "Map has too many areas");

        Com.DPrintf(" numareas=" + count + '\n');
        numareas = count;

        if (debugloadmap) {
            Com.DPrintf("areas(numportals, firstportal)\n");
        }

        for (int i = 0; i < count; i++) {

            qfiles.darea_t in = new qfiles.darea_t(ByteBuffer.wrap(cmod_base, i
                    * qfiles.darea_t.SIZE + l.fileofs, qfiles.darea_t.SIZE));
            carea_t out = map_areas[i];

            out.numareaportals = in.numareaportals;
            out.firstareaportal = in.firstareaportal;
            out.floodvalid = 0;
            out.floodnum = 0;
            if (debugloadmap) {
                Com.DPrintf("| %6i| %6i|\n", new Vargs()
                        .add(out.numareaportals).add(out.firstareaportal));
            }
        }
    }

    /** Loads area portals. */
    private static void CMod_LoadAreaPortals(lump_t l) {
        Com.DPrintf("CMod_LoadAreaPortals()\n");

        if ((l.filelen % qfiles.dareaportal_t.SIZE) != 0)
            Com.Error(Defines.ERR_DROP, "MOD_LoadBmodel: funny lump size");
        int count = l.filelen / qfiles.dareaportal_t.SIZE;

        if (count > Defines.MAX_MAP_AREAS)
            Com.Error(Defines.ERR_DROP, "Map has too many areas");

        numareaportals = count;
        Com.DPrintf(" numareaportals=" + count + '\n');
        if (debugloadmap) {
            Com.DPrintf("areaportals(portalnum, otherarea)\n");
        }
        for (int i = 0; i < count; i++) {
            qfiles.dareaportal_t in = new qfiles.dareaportal_t(ByteBuffer.wrap(cmod_base, i
                            * qfiles.dareaportal_t.SIZE + l.fileofs,
                    qfiles.dareaportal_t.SIZE));

            qfiles.dareaportal_t out = map_areaportals[i];

            out.portalnum = in.portalnum;
            out.otherarea = in.otherarea;

            if (debugloadmap) {
                Com.DPrintf("|%6i|%6i|\n", new Vargs().add(out.portalnum).add(
                        out.otherarea));
            }
        }
    }

    /** Loads visibility data. */
    private static void CMod_LoadVisibility(lump_t l) {
        Com.DPrintf("CMod_LoadVisibility()\n");

        numvisibility = l.filelen;

        Com.DPrintf(" numvisibility=" + numvisibility + '\n');

        if (l.filelen > Defines.MAX_MAP_VISIBILITY)
            Com.Error(Defines.ERR_DROP, "Map has too large visibility lump");

        System.arraycopy(cmod_base, l.fileofs, map_visibility, 0, l.filelen);

        ByteBuffer bb = ByteBuffer.wrap(map_visibility, 0, l.filelen);
        bb.order(ByteOrder.LITTLE_ENDIAN);

        map_vis = new qfiles.dvis_t(bb);

    }

    /** Loads entity strings. */
    private static void CMod_LoadEntityString(lump_t l) {
        Com.DPrintf("CMod_LoadEntityString()\n");

        numentitychars = l.filelen;

        if (l.filelen > Defines.MAX_MAP_ENTSTRING)
            Com.Error(Defines.ERR_DROP, "Map has too large entity lump");

        int x = 0;
        for (; x < l.filelen && cmod_base[x + l.fileofs] != 0; x++);

        map_entitystring = new String(cmod_base, l.fileofs, x).trim();
        Com.dprintln("entitystring=" + map_entitystring.length() + 
                " bytes, [" + map_entitystring.substring(0, Math.min (
                        map_entitystring.length(), 15)) + "...]" );
    }

    /** Returns the model with a given id "*" + <number> */
    public static cmodel_t InlineModel(String name) {
        if (name == null || name.charAt(0) != '*')
            Com.Error(Defines.ERR_DROP, "CM_InlineModel: bad name");

        int num = Lib.atoi(name.substring(1));

        if (num < 1 || num >= numcmodels)
            Com.Error(Defines.ERR_DROP, "CM_InlineModel: bad number");

        return map_cmodels[num];
    }

    public static int CM_NumClusters() {
        return numclusters;
    }

    public static int CM_NumInlineModels() {
        return numcmodels;
    }

    public static String CM_EntityString() {
        return map_entitystring;
    }

    public static int CM_LeafContents(int leafnum) {
        if (leafnum < 0 || leafnum >= numleafs)
            Com.Error(Defines.ERR_DROP, "CM_LeafContents: bad number");
        return map_leafs[leafnum].contents;
    }

    public static int CM_LeafCluster(int leafnum) {
        if (leafnum < 0 || leafnum >= numleafs)
            Com.Error(Defines.ERR_DROP, "CM_LeafCluster: bad number");
        return map_leafs[leafnum].cluster;
    }

    public static int CM_LeafArea(int leafnum) {
        if (leafnum < 0 || leafnum >= numleafs)
            Com.Error(Defines.ERR_DROP, "CM_LeafArea: bad number");
        return map_leafs[leafnum].area;
    }

    private static cplane_t[] box_planes;

    private static int box_headnode;

    /** Set up the planes and nodes so that the six floats of a bounding box can
     * just be stored out and get a proper clipping hull structure.
     */
    private static void CM_InitBoxHull() {

        box_headnode = numnodes; 

        box_planes = new cplane_t[] { map_planes[numplanes],
                map_planes[numplanes + 1], map_planes[numplanes + 2],
                map_planes[numplanes + 3], map_planes[numplanes + 4],
                map_planes[numplanes + 5], map_planes[numplanes + 6],
                map_planes[numplanes + 7], map_planes[numplanes + 8],
                map_planes[numplanes + 9], map_planes[numplanes + 10],
                map_planes[numplanes + 11], map_planes[numplanes + 12] };

        if (numnodes + 6 > Defines.MAX_MAP_NODES
                || numbrushes + 1 > Defines.MAX_MAP_BRUSHES
                || numleafbrushes + 1 > Defines.MAX_MAP_LEAFBRUSHES
                || numbrushsides + 6 > Defines.MAX_MAP_BRUSHSIDES
                || numplanes + 12 > Defines.MAX_MAP_PLANES)
            Com.Error(Defines.ERR_DROP, "Not enough room for box tree");

        cbrush_t box_brush = map_brushes[numbrushes];
        box_brush.numsides = 6;
        box_brush.firstbrushside = numbrushsides;
        box_brush.contents = Defines.CONTENTS_MONSTER;

        cleaf_t box_leaf = map_leafs[numleafs];
        box_leaf.contents = Defines.CONTENTS_MONSTER;
        box_leaf.firstleafbrush = (short) numleafbrushes;
        box_leaf.numleafbrushes = 1;

        map_leafbrushes[numleafbrushes] = numbrushes;

        for (int i = 0; i < 6; i++) {
            int side = i & 1;


            cbrushside_t s = map_brushsides[numbrushsides + i];
            s.plane = map_planes[(numplanes + i * 2 + side)];
            s.surface = nullsurface;


            cnode_t c = map_nodes[box_headnode + i];
            c.plane = map_planes[(numplanes + i * 2)];
            c.children[side] = -1 - emptyleaf;
            if (i != 5)
                c.children[side ^ 1] = box_headnode + i + 1;
            else
                c.children[side ^ 1] = -1 - numleafs;


            cplane_t p = box_planes[i * 2];
            p.type = (byte) (i >> 1);
            p.signbits = 0;
            Math3D.VectorClear(p.normal);
            p.normal[i >> 1] = 1;

            p = box_planes[i * 2 + 1];
            p.type = (byte) (3 + (i >> 1));
            p.signbits = 0;
            Math3D.VectorClear(p.normal);
            p.normal[i >> 1] = -1;
        }
    }

    /** To keep everything totally uniform, bounding boxes are turned into small
     * BSP trees instead of being compared directly. */
    public static int HeadnodeForBox(float[] mins, float[] maxs) {
        box_planes[0].dist = maxs[0];
        box_planes[1].dist = -maxs[0];
        box_planes[2].dist = mins[0];
        box_planes[3].dist = -mins[0];
        box_planes[4].dist = maxs[1];
        box_planes[5].dist = -maxs[1];
        box_planes[6].dist = mins[1];
        box_planes[7].dist = -mins[1];
        box_planes[8].dist = maxs[2];
        box_planes[9].dist = -maxs[2];
        box_planes[10].dist = mins[2];
        box_planes[11].dist = -mins[2];

        return box_headnode;
    }

    /** Recursively searches the leaf number that contains the 3d point. */
    private static int CM_PointLeafnum_r(float[] p, int num) {

        while (num >= 0) {
            cnode_t node = map_nodes[num];
            cplane_t plane = node.plane;

            float d;
            if (plane.type < 3)
                d = p[plane.type] - plane.dist;
            else
                d = Math3D.DotProduct(plane.normal, p) - plane.dist;
            if (d < 0)
                num = node.children[1];
            else
                num = node.children[0];
        }

        Globals.c_pointcontents++; 

        return -1 - num;
    }

    /** Searches the leaf number that contains the 3d point. */
    public static int CM_PointLeafnum(float[] p) {
    	
        if (numplanes == 0)
            return 0; 
        return CM_PointLeafnum_r(p, 0);
    }


    private static int leaf_count;
    private static int leaf_maxcount;

    private static int[] leaf_list;

    private static float[] leaf_mins;
    private static float[] leaf_maxs;

    private static int leaf_topnode;

    /** Recursively fills in a list of all the leafs touched. */    
    private static void CM_BoxLeafnums_r(int nodenum) {

        while (true) {
            if (nodenum < 0) {
                if (leaf_count >= leaf_maxcount) {
                    Com.DPrintf("CM_BoxLeafnums_r: overflow\n");
                    return;
                }
                leaf_list[leaf_count++] = -1 - nodenum;
                return;
            }

            cnode_t node = map_nodes[nodenum];
            cplane_t plane = node.plane;

            int s = Math3D.BoxOnPlaneSide(leaf_mins, leaf_maxs, plane);

            switch (s) {
                case 1 -> nodenum = node.children[0];
                case 2 -> nodenum = node.children[1];
                default -> {
                    if (leaf_topnode == -1)
                        leaf_topnode = nodenum;
                    CM_BoxLeafnums_r(node.children[0]);
                    nodenum = node.children[1];
                }
            }
        }
    }

    /** Fills in a list of all the leafs touched and starts with the head node. */
    private static int CM_BoxLeafnums_headnode(float[] mins, float[] maxs,
                                               int[] list, int listsize, int headnode, int[] topnode) {
        leaf_list = list;
        leaf_count = 0;
        leaf_maxcount = listsize;
        leaf_mins = mins;
        leaf_maxs = maxs;

        leaf_topnode = -1;

        CM_BoxLeafnums_r(headnode);

        if (topnode != null)
            topnode[0] = leaf_topnode;

        return leaf_count;
    }

    /** Fills in a list of all the leafs touched. */
    public static int CM_BoxLeafnums(float[] mins, float[] maxs, int[] list,
                                     int listsize, int[] topnode) {
        return CM_BoxLeafnums_headnode(mins, maxs, list, listsize,
                map_cmodels[0].headnode, topnode);
    }

    /** Returns a tag that describes the content of the point. */
    public static int PointContents(float[] p, int headnode) {

        if (numnodes == 0) 
            return 0;

        int l = CM_PointLeafnum_r(p, headnode);

        return map_leafs[l].contents;
    }

    /*
     * ================== CM_TransformedPointContents
     * 
     * Handles offseting and rotation of the end points for moving and rotating
     * entities ==================
     */
    public static int TransformedPointContents(float[] p, int headnode,
            float[] origin, float[] angles) {
        float[] p_l = { 0, 0, 0 };


        Math3D.VectorSubtract(p, origin, p_l);

        
        if (headnode != box_headnode) {
            boolean b = IntStream.of(0, 1, 2).anyMatch(i -> angles[i] != 0);
            if (b) {
                float[] up = {0, 0, 0};
                float[] right = {0, 0, 0};
                float[] forward = {0, 0, 0};
                Math3D.AngleVectors(angles, forward, right, up);

                float[] temp = {0, 0, 0};
                Math3D.VectorCopy(p_l, temp);
                p_l[0] = Math3D.DotProduct(temp, forward);
                p_l[1] = -Math3D.DotProduct(temp, right);
                p_l[2] = Math3D.DotProduct(temp, up);
            }
        }

        int l = CM_PointLeafnum_r(p_l, headnode);

        return map_leafs[l].contents;
    }

    /*
     * ===============================================================================
     * 
     * BOX TRACING
     * 
     * ===============================================================================
     */

    
    private static final float DIST_EPSILON = 0.03125f;

    private static final float[] trace_start = { 0, 0, 0 };
    private static final float[] trace_end = { 0, 0, 0 };

    private static final float[] trace_mins = { 0, 0, 0 };
    private static final float[] trace_maxs = { 0, 0, 0 };

    private static final float[] trace_extents = { 0, 0, 0 };

    private static trace_t trace_trace = new trace_t();

    private static int trace_contents;

    private static boolean trace_ispoint; 

    /*
     * ================ CM_ClipBoxToBrush ================
     */
    private static void CM_ClipBoxToBrush(float[] mins, float[] maxs,
                                          float[] p1, float[] p2, trace_t trace, cbrush_t brush) {

        if (brush.numsides == 0)
            return;

        Globals.c_brush_traces++;

        boolean getout = false;
        boolean startout = false;
        cbrushside_t leadside = null;

        cplane_t clipplane = null;
        float leavefrac = 1;
        float enterfrac = -1;
        float[] ofs = {0, 0, 0};
        for (int i = 0; i < brush.numsides; i++) {
            cbrushside_t side = map_brushsides[brush.firstbrushside + i];
            cplane_t plane = side.plane;


            float dist;
            if (!trace_ispoint) {

                

                
                for (int j = 0; j < 3; j++) {
                    if (plane.normal[j] < 0)
                        ofs[j] = maxs[j];
                    else
                        ofs[j] = mins[j];
                }
                dist = Math3D.DotProduct(ofs, plane.normal);
                dist = plane.dist - dist;
            } else { 
                dist = plane.dist;
            }

            float d1 = Math3D.DotProduct(p1, plane.normal) - dist;
            float d2 = Math3D.DotProduct(p2, plane.normal) - dist;

            if (d2 > 0)
                getout = true; 
            if (d1 > 0)
                startout = true;

            
            if (d1 > 0 && d2 >= d1)
                return;

            if (d1 <= 0 && d2 <= 0)
                continue;


            float f;
            if (d1 > d2) {
                f = (d1 - DIST_EPSILON) / (d1 - d2);
                if (f > enterfrac) {
                    enterfrac = f;
                    clipplane = plane;
                    leadside = side;
                }
            } else { 
                f = (d1 + DIST_EPSILON) / (d1 - d2);
                if (f < leavefrac)
                    leavefrac = f;
            }
        }

        if (!startout) { 
            trace.startsolid = true;
            if (!getout)
                trace.allsolid = true;
            return;
        }
        if (enterfrac < leavefrac) {
            if (enterfrac > -1 && enterfrac < trace.fraction) {
                if (enterfrac < 0)
                    enterfrac = 0;
                trace.fraction = enterfrac;
                
                trace.plane.set(clipplane);
                trace.surface = leadside.surface.c;
                trace.contents = brush.contents;
            }
        }
    }

    /*
     * ================ CM_TestBoxInBrush ================
     */
    private static void CM_TestBoxInBrush(float[] mins, float[] maxs,
                                          float[] p1, trace_t trace, cbrush_t brush) {

        if (brush.numsides == 0)
            return;

        float[] ofs = {0, 0, 0};
        for (int i = 0; i < brush.numsides; i++) {
            cbrushside_t side = map_brushsides[brush.firstbrushside + i];
            cplane_t plane = side.plane;


            for (int j = 0; j < 3; j++) {
                if (plane.normal[j] < 0)
                    ofs[j] = maxs[j];
                else
                    ofs[j] = mins[j];
            }
            float dist = Math3D.DotProduct(ofs, plane.normal);
            dist = plane.dist - dist;

            float d1 = Math3D.DotProduct(p1, plane.normal) - dist;


            if (d1 > 0)
                return;

        }

        
        trace.startsolid = trace.allsolid = true;
        trace.fraction = 0;
        trace.contents = brush.contents;
    }

    /**
     * CM_TraceToLeaf.
     */
    private static void CM_TraceToLeaf(int leafnum) {

        cleaf_t leaf = map_leafs[leafnum];
        if (0 == (leaf.contents & trace_contents))
            return;

        
        for (int k = 0; k < leaf.numleafbrushes; k++) {

            int brushnum = map_leafbrushes[leaf.firstleafbrush + k];
            cbrush_t b = map_brushes[brushnum];
            if (b.checkcount == checkcount)
                continue; 
            b.checkcount = checkcount;

            if (0 == (b.contents & trace_contents))
                continue;
            CM_ClipBoxToBrush(trace_mins, trace_maxs, trace_start, trace_end,
                    trace_trace, b);
            if (0 == trace_trace.fraction)
                return;
        }

    }

    /*
     * ================ CM_TestInLeaf ================
     */
    private static void CM_TestInLeaf(int leafnum) {

        cleaf_t leaf = map_leafs[leafnum];
        if (0 == (leaf.contents & trace_contents))
            return;
        
        for (int k = 0; k < leaf.numleafbrushes; k++) {
            int brushnum = map_leafbrushes[leaf.firstleafbrush + k];
            cbrush_t b = map_brushes[brushnum];
            if (b.checkcount == checkcount)
                continue; 
            b.checkcount = checkcount;

            if (0 == (b.contents & trace_contents))
                continue;
            CM_TestBoxInBrush(trace_mins, trace_maxs, trace_start, trace_trace,
                    b);
            if (0 == trace_trace.fraction)
                return;
        }

    }

    /*
     * ================== CM_RecursiveHullCheck ==================
     */
    private static void CM_RecursiveHullCheck(int num, float p1f, float p2f,
                                              float[] p1, float[] p2) {

        if (trace_trace.fraction <= p1f)
            return; 

        
        if (num < 0) {
            CM_TraceToLeaf(-1 - num);
            return;
        }


        cnode_t node = map_nodes[num];
        cplane_t plane = node.plane;

        float offset;
        float t2;
        float t1;
        if (plane.type < 3) {
            t1 = p1[plane.type] - plane.dist;
            t2 = p2[plane.type] - plane.dist;
            offset = trace_extents[plane.type];
        } else {
            t1 = Math3D.DotProduct(plane.normal, p1) - plane.dist;
            t2 = Math3D.DotProduct(plane.normal, p2) - plane.dist;
            if (trace_ispoint)
                offset = 0;
            else
                offset = Math.abs(trace_extents[0] * plane.normal[0])
                        + Math.abs(trace_extents[1] * plane.normal[1])
                        + Math.abs(trace_extents[2] * plane.normal[2]);
        }

        
        if (t1 >= offset && t2 >= offset) {
            CM_RecursiveHullCheck(node.children[0], p1f, p2f, p1, p2);
            return;
        }
        if (t1 < -offset && t2 < -offset) {
            CM_RecursiveHullCheck(node.children[1], p1f, p2f, p1, p2);
            return;
        }


        int side;
        float idist;
        float frac2;
        float frac;
        if (t1 < t2) {
            idist = 1.0f / (t1 - t2);
            side = 1;
            frac2 = (t1 + offset + DIST_EPSILON) * idist;
            frac = (t1 - offset + DIST_EPSILON) * idist;
        } else if (t1 > t2) {
            idist = 1.0f / (t1 - t2);
            side = 0;
            frac2 = (t1 - offset - DIST_EPSILON) * idist;
            frac = (t1 + offset + DIST_EPSILON) * idist;
        } else {
            side = 0;
            frac = 1;
            frac2 = 0;
        }

        
        if (frac < 0)
            frac = 0;
        if (frac > 1)
            frac = 1;

        float midf = p1f + (p2f - p1f) * frac;
        float[] mid = Vec3Cache.get();

        int i;
        for (i = 0; i < 3; i++)
            mid[i] = p1[i] + frac * (p2[i] - p1[i]);

        CM_RecursiveHullCheck(node.children[side], p1f, midf, p1, mid);

        
        if (frac2 < 0)
            frac2 = 0;
        if (frac2 > 1)
            frac2 = 1;

        midf = p1f + (p2f - p1f) * frac2;
        for (i = 0; i < 3; i++)
            mid[i] = p1[i] + frac2 * (p2[i] - p1[i]);

        CM_RecursiveHullCheck(node.children[side ^ 1], midf, p2f, mid, p2);
        Vec3Cache.release();
    }

    

    /*
     * ================== CM_BoxTrace ==================
     */
    public static trace_t BoxTrace(float[] start, float[] end, float[] mins,
            float[] maxs, int headnode, int brushmask) {

        
        checkcount++;

        
        Globals.c_traces++;

        
        
        trace_trace = new trace_t();

        trace_trace.fraction = 1;
        trace_trace.surface = nullsurface.c;

        if (numnodes == 0) {
            
            return trace_trace;
        }

        trace_contents = brushmask;
        Math3D.VectorCopy(start, trace_start);
        Math3D.VectorCopy(end, trace_end);
        Math3D.VectorCopy(mins, trace_mins);
        Math3D.VectorCopy(maxs, trace_maxs);

        
        
        
        if (start[0] == end[0] && start[1] == end[1] && start[2] == end[2]) {

            float[] c1 = { 0, 0, 0 };

            Math3D.VectorAdd(start, mins, c1);
            float[] c2 = {0, 0, 0};
            Math3D.VectorAdd(start, maxs, c2);

            int i;
            for (i = 0; i < 3; i++) {
                c1[i] -= 1;
                c2[i] += 1;
            }

            int topnode = 0;
            int[] tn = {topnode};

            int[] leafs = new int[1024];
            int numleafs = CM_BoxLeafnums_headnode(c1, c2, leafs, 1024, headnode,
                    tn);
            for (i = 0; i < numleafs; i++) {
                CM_TestInLeaf(leafs[i]);
                if (trace_trace.allsolid)
                    break;
            }
            Math3D.VectorCopy(start, trace_trace.endpos);
            return trace_trace;
        }

        
        
        
        if (mins[0] == 0 && mins[1] == 0 && mins[2] == 0 && maxs[0] == 0
                && maxs[1] == 0 && maxs[2] == 0) {
            trace_ispoint = true;
            Math3D.VectorClear(trace_extents);
        } else {
            trace_ispoint = false;
            trace_extents[0] = Math.max(-mins[0], maxs[0]);
            trace_extents[1] = Math.max(-mins[1], maxs[1]);
            trace_extents[2] = Math.max(-mins[2], maxs[2]);
        }

        
        
        
        CM_RecursiveHullCheck(headnode, 0, 1, start, end);

        if (trace_trace.fraction == 1) {
            Math3D.VectorCopy(end, trace_trace.endpos);
        } else {
            for (int i = 0; i < 3; i++)
                trace_trace.endpos[i] = start[i] + trace_trace.fraction
                        * (end[i] - start[i]);
        }
        return trace_trace;
    }

    /**
     * CM_TransformedBoxTrace handles offseting and rotation of the end points for moving and rotating
     * entities.
     */
    public static trace_t TransformedBoxTrace(float[] start, float[] end,
            float[] mins, float[] maxs, int headnode, int brushmask,
            float[] origin, float[] angles) {
        float[] start_l = { 0, 0, 0 };


        Math3D.VectorSubtract(start, origin, start_l);
        float[] end_l = {0, 0, 0};
        Math3D.VectorSubtract(end, origin, end_l);


        boolean rotated = headnode != box_headnode
                && (IntStream.of(0, 1, 2).anyMatch(i -> angles[i] != 0));

        float[] temp = {0, 0, 0};
        float[] up = {0, 0, 0};
        float[] right = {0, 0, 0};
        float[] forward = {0, 0, 0};
        if (rotated) {
            Math3D.AngleVectors(angles, forward, right, up);

            Math3D.VectorCopy(start_l, temp);
            start_l[0] = Math3D.DotProduct(temp, forward);
            start_l[1] = -Math3D.DotProduct(temp, right);
            start_l[2] = Math3D.DotProduct(temp, up);

            Math3D.VectorCopy(end_l, temp);
            end_l[0] = Math3D.DotProduct(temp, forward);
            end_l[1] = -Math3D.DotProduct(temp, right);
            end_l[2] = Math3D.DotProduct(temp, up);
        }


        trace_t trace = BoxTrace(start_l, end_l, mins, maxs, headnode, brushmask);

        if (rotated && trace.fraction != 1.0) {

            float[] a = {0, 0, 0};
            Math3D.VectorNegate(angles, a);
            Math3D.AngleVectors(a, forward, right, up);

            Math3D.VectorCopy(trace.plane.normal, temp);
            trace.plane.normal[0] = Math3D.DotProduct(temp, forward);
            trace.plane.normal[1] = -Math3D.DotProduct(temp, right);
            trace.plane.normal[2] = Math3D.DotProduct(temp, up);
        }

        trace.endpos[0] = start[0] + trace.fraction * (end[0] - start[0]);
        trace.endpos[1] = start[1] + trace.fraction * (end[1] - start[1]);
        trace.endpos[2] = start[2] + trace.fraction * (end[2] - start[2]);

        return trace;
    }

    /*
     * ===============================================================================
     * PVS / PHS
     * ===============================================================================
     */

    /*
     * =================== CM_DecompressVis ===================
     */
    private static void CM_DecompressVis(byte[] in, int offset, byte[] out) {

        int row = (numclusters + 7) >> 3;
        int outp = 0;

        if (in == null || numvisibility == 0) { 
                                                
            while (row != 0) {
                out[outp++] = (byte) 0xFF;
                row--;
            }
            return;
        }

        int inp = offset;
        do {
            if (in[inp] != 0) {
                out[outp++] = in[inp++];
                continue;
            }

            int c = in[inp + 1] & 0xFF;
            inp += 2;
            if (outp + c > row) {
                c = row - (outp);
                Com.DPrintf("warning: Vis decompression overrun\n");
            }
            while (c != 0) {
                out[outp++] = 0;
                c--;
            }
        } while (outp < row);
    }

    private static final byte[] pvsrow = new byte[Defines.MAX_MAP_LEAFS / 8];

    private static final byte[] phsrow = new byte[Defines.MAX_MAP_LEAFS / 8];

    public static byte[] CM_ClusterPVS(int cluster) {
        if (cluster == -1)
            Arrays.fill(pvsrow, 0, (numclusters + 7) >> 3, (byte) 0);
        else
            CM_DecompressVis(map_visibility,
                    map_vis.bitofs[cluster][Defines.DVIS_PVS], pvsrow);
        return pvsrow;
    }

    public static byte[] CM_ClusterPHS(int cluster) {
        if (cluster == -1)
            Arrays.fill(phsrow, 0, (numclusters + 7) >> 3, (byte) 0);
        else
            CM_DecompressVis(map_visibility,
                    map_vis.bitofs[cluster][Defines.DVIS_PHS], phsrow);
        return phsrow;
    }

    /*
     * ===============================================================================
     * AREAPORTALS
     * ===============================================================================
     */

    private static void FloodArea_r(carea_t area, int floodnum) {

        if (area.floodvalid == floodvalid) {
            if (area.floodnum == floodnum)
                return;
            Com.Error(Defines.ERR_DROP, "FloodArea_r: reflooded");
        }

        area.floodnum = floodnum;
        area.floodvalid = floodvalid;

        for (int i = 0; i < area.numareaportals; i++) {
            qfiles.dareaportal_t p = map_areaportals[area.firstareaportal + i];
            if (portalopen[p.portalnum])
                FloodArea_r(map_areas[p.otherarea], floodnum);
        }
    }

    /**
     * FloodAreaConnections.
     */
    private static void FloodAreaConnections() {
        Com.DPrintf("FloodAreaConnections...\n");


        floodvalid++;
        int floodnum = 0;

        
        for (int i = 1; i < numareas; i++) {

            carea_t area = map_areas[i];

            if (area.floodvalid == floodvalid)
                continue; 
            floodnum++;
            FloodArea_r(area, floodnum);
        }
    }

    /**
     * CM_SetAreaPortalState.
     */
    public static void CM_SetAreaPortalState(int portalnum, boolean open) {
        if (portalnum > numareaportals)
            Com.Error(Defines.ERR_DROP, "areaportal > numareaportals");

        portalopen[portalnum] = open;
        FloodAreaConnections();
    }

    /**
     * CM_AreasConnected returns true, if two areas are connected.
     */

    public static boolean CM_AreasConnected(int area1, int area2) {
        if (map_noareas.value != 0)
            return true;

        if (area1 > numareas || area2 > numareas)
            Com.Error(Defines.ERR_DROP, "area > numareas");

        return map_areas[area1].floodnum == map_areas[area2].floodnum;

    }

    /**
     * CM_WriteAreaBits writes a length byte followed by a bit vector of all the areas that area
     * in the same flood as the area parameter
     * 
     * This is used by the client refreshes to cull visibility.
     */
    public static int CM_WriteAreaBits(byte[] buffer, int area) {

        int bytes = (numareas + 7) >> 3;

        if (map_noareas.value != 0) { 
            
            Arrays.fill(buffer, 0, bytes, (byte) 255);
        } else {
            Arrays.fill(buffer, 0, bytes, (byte) 0);
            int floodnum = map_areas[area].floodnum;
            for (int i = 0; i < numareas; i++) {
                if (map_areas[i].floodnum == floodnum || area == 0)
                    buffer[i >> 3] |= 1 << (i & 7);
            }
        }

        return bytes;
    }

    /**
     * CM_WritePortalState writes the portal state to a savegame file.
     */

    public static void CM_WritePortalState(RandomAccessFile os) {

        try {
            for (boolean aPortalopen : portalopen)
                os.writeInt(aPortalopen ? 1 : 0);
        } catch (Exception e) {
            Com.Printf("ERROR:" + e);
            e.printStackTrace();
        }
    }

    /**
     * CM_ReadPortalState reads the portal state from a savegame file and recalculates the area
     * connections.
     */
    public static void CM_ReadPortalState(RandomAccessFile f) {

        
        int len = portalopen.length * 4;

        byte[] buf = new byte[len];

        FS.Read(buf, len, f);

        ByteBuffer bb = ByteBuffer.wrap(buf);
        IntBuffer ib = bb.asIntBuffer();

        for (int n = 0; n < portalopen.length; n++)
            portalopen[n] = ib.get() != 0;

        FloodAreaConnections();
    }

    /**
     * CM_HeadnodeVisible returns true if any leaf under headnode has a cluster that is potentially
     * visible.
     */
    public static boolean CM_HeadnodeVisible(int nodenum, byte[] visbits) {
        while (true) {
            if (nodenum < 0) {
                int leafnum = -1 - nodenum;
                int cluster = map_leafs[leafnum].cluster;
                if (cluster == -1) return false;

                return 0 != (visbits[cluster >>> 3] & (1 << (cluster & 7)));

            }

            cnode_t node = map_nodes[nodenum];
            if (CM_HeadnodeVisible(node.children[0], visbits)) return true;

            nodenum = node.children[1];
        }
    }
}