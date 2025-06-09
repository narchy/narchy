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





package jake2.util;

import jake2.game.*;
import jake2.qcommon.Com;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * RandomAccessFile, but handles readString/WriteString specially and offers
 * other helper functions
 */
public class QuakeFile extends RandomAccessFile {

    /** Standard Constructor. */
    public QuakeFile(String filename, String mode) throws FileNotFoundException {
        super(filename, mode);
    }

    /** Writes a Vector to a RandomAccessFile. */
    public void writeVector(float[] v) throws IOException {
        writeFloat(v[0]);
        writeFloat(v[1]);
        writeFloat(v[2]);
    }

    /** Writes a Vector to a RandomAccessFile. */
    public final float[] readVector() throws IOException {
        return new float[] { readFloat(), readFloat(), readFloat() };
    }

    /** Reads a length specified string from a file. */
    public String readString() throws IOException {
        int len = readInt();

        return switch (len) {
            case -1 -> null;
            case 0 -> "";
            default -> {
                byte[] bb = new byte[len];
                super.read(bb, 0, len);
                yield new String(bb, 0, len);
            }
        };
    }

    /** Writes a length specified string to a file. */
    public void writeString(String s) throws IOException {
        if (s == null) {
            writeInt(-1);
            return;
        }

        int l = s.length();
        writeInt(l);
        if (l != 0)
            writeBytes(s);
    }

    /** Writes the edict reference. */
    public void writeEdictRef(edict_t ent) throws IOException {
        if (ent == null)
            writeInt(-1);
        else {
            writeInt(ent.s.number);
        }
    }

    /**
     * Reads an edict index from a file and returns the edict.
     */

    public edict_t readEdictRef() throws IOException {
        int i = readInt();

        
        if (i < 0)
            return null;

        if (i > GameBase.g_edicts.length) {
            Com.DPrintf("jake2: illegal edict num:" + i + '\n');
            return null;
        }

        
        return GameBase.g_edicts[i];
    }

    /** Writes the Adapter-ID to the file. */
    public void writeAdapter(SuperAdapter a) throws IOException {
        writeInt(3988);
        if (a == null)
            writeString(null);
        else {
            String str = a.getID();
            if (a == null) {
                Com.DPrintf("writeAdapter: invalid Adapter id for " + a + '\n');
            }
            writeString(str);
        }
    }

    /** Reads the adapter id and returns the adapter. */
    public SuperAdapter readAdapter() throws IOException {
        if (readInt() != 3988)
            Com.DPrintf("wrong read position: readadapter 3988 \n");

        String id = readString();

        if (id == null) {
            
            return null;
        }

        return SuperAdapter.getFromID(id);
    }

    /** Writes an item reference. */
    public void writeItem(gitem_t item) throws IOException {
        if (item == null)
            writeInt(-1);
        else
            writeInt(item.index);
    }

    /** Reads the item index and returns the game item. */
    public gitem_t readItem() throws IOException {
        int ndx = readInt();
        if (ndx == -1)
            return null;
        else
            return GameItemList.itemlist[ndx];
    }

}