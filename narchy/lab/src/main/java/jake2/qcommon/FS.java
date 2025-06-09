/*
 * FS.java
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
package jake2.qcommon;

import jake2.Defines;
import jake2.Globals;
import jake2.game.Cmd;
import jake2.game.cvar_t;
import jake2.sys.Sys;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * FS
 * 
 * @author cwei
 */
public final class FS extends Globals {
    private static final Pattern SLASHES = Pattern.compile("\\\\");

    /*
     * ==================================================
     * 
     * QUAKE FILESYSTEM
     * 
     * ==================================================
     */

    static class packfile_t {
        static final int SIZE = 64;

        static final int NAME_SIZE = 56;

        String name; 

        int filepos;
        int filelen;

        public String toString() {
            return name + " [ length: " + filelen + " pos: " + filepos + " ]";
        }
    }

    static class pack_t {
        String filename;

        RandomAccessFile handle;
        
        ByteBuffer backbuffer;

        int numfiles;

        HashMap files; 
    }

    private static String fs_gamedir;

    private static String fs_userdir;

    private static cvar_t fs_basedir;

    private static cvar_t fs_cddir;

    public static cvar_t fs_gamedirvar;

    static class filelink_t {
        String from;

        int fromlength;

        String to;
    }

    
    private static final Deque<filelink_t> fs_links = new ArrayDeque();

    static class searchpath_t {
        String filename;

        pack_t pack; 

        searchpath_t next;
    }

    private static searchpath_t fs_searchpaths;

    
    private static searchpath_t fs_base_searchpaths;

    /*
     * All of Quake's data access is through a hierchal file system, but the
     * contents of the file system can be transparently merged from several
     * sources.
     * 
     * The "base directory" is the path to the directory holding the quake.exe
     * and all game directories. The sys_* files pass this to host_init in
     * quakeparms_t->basedir. This can be overridden with the "-basedir" command
     * line parm to allow code debugging in a different directory. The base
     * directory is only used during filesystem initialization.
     * 
     * The "game directory" is the first tree on the search path and directory
     * that all generated files (savegames, screenshots, demos, config files)
     * will be saved to. This can be overridden with the "-game" command line
     * parameter. The game directory can never be changed while quake is
     * executing. This is a precacution against having a malicious server
     * instruct clients to write files over areas they shouldn't.
     *  
     */

    /*
     * CreatePath
     * 
     * Creates any directories needed to store the given filename.
     */
    public static void CreatePath(String path) {
        int index = path.lastIndexOf('/');
        
        if (index > 0) {
            File f = new File(path.substring(0, index));
            if (!f.mkdirs() && !f.isDirectory()) {
                Com.Printf("can't create path \"" + path + '"' + '\n');
            }
        }
    }

    /*
     * FCloseFile
     * 
     * For some reason, other dll's can't just call fclose() on files returned
     * by FS_FOpenFile...
     */
    public static void FCloseFile(RandomAccessFile file) throws IOException {
        file.close();
    }

    public static void FCloseFile(InputStream stream) throws IOException {
        stream.close();
    }

    public static int FileLength(String filename) {

        file_from_pak = 0;


        String netpath;
        for (filelink_t link : fs_links) {
            if (filename.regionMatches(0, link.from, 0, link.fromlength)) {
                netpath = link.to + filename.substring(link.fromlength);
                File file = new File(netpath);
                if (file.canRead()) {
                    Com.DPrintf("link file: " + netpath + '\n');
                    return (int) file.length();
                }
                return -1;
            }
        }

        

        for (searchpath_t search = fs_searchpaths; search != null; search = search.next) {
            
            if (search.pack != null) {

                pack_t pak = search.pack;
                filename = filename.toLowerCase();
                packfile_t entry = (packfile_t) pak.files.get(filename);

                if (entry != null) {
                    
                    file_from_pak = 1;
                    Com.DPrintf("PackFile: " + pak.filename + " : " + filename
                            + '\n');
                    
                    File file = new File(pak.filename);
                    if (!file.canRead()) {
                        Com.Error(Defines.ERR_FATAL, "Couldn't reopen "
                                + pak.filename);
                    }
                    return entry.filelen;
                }
            } else {
                
                netpath = search.filename + '/' + filename;

                File file = new File(netpath);
                if (!file.canRead())
                    continue;

                Com.DPrintf("FindFile: " + netpath + '\n');

                return (int) file.length();
            }
        }
        Com.DPrintf("FindFile: can't find " + filename + '\n');
        return -1;
    }

    public static int file_from_pak;

    /*
     * FOpenFile
     * 
     * Finds the file in the search path. returns a RadomAccesFile. Used for
     * streaming data out of either a pak file or a seperate file.
     */
    public static RandomAccessFile FOpenFile(String filename)
            throws IOException {

        file_from_pak = 0;


        File file;
        String netpath;
        for (filelink_t link : fs_links) {
            if (filename.regionMatches(0, link.from, 0, link.fromlength)) {
                netpath = link.to + filename.substring(link.fromlength);
                file = new File(netpath);
                if (file.canRead()) {

                    return new RandomAccessFile(file, "r");
                }
                return null;
            }
        }

        
        
        
        for (searchpath_t search = fs_searchpaths; search != null; search = search.next) {
            
            if (search.pack != null) {

                pack_t pak = search.pack;
                filename = filename.toLowerCase();
                packfile_t entry = (packfile_t) pak.files.get(filename);

                if (entry != null) {
                    
                    file_from_pak = 1;
                    
                    
                    file = new File(pak.filename);
                    if (!file.canRead())
                        Com.Error(Defines.ERR_FATAL, "Couldn't reopen "
                                + pak.filename);
                    if (pak.handle == null || !pak.handle.getFD().valid()) {
                        
                        pak.handle = new RandomAccessFile(pak.filename, "r");
                    }
                    

                    RandomAccessFile raf = new RandomAccessFile(file, "r");
                    raf.seek(entry.filepos);

                    return raf;
                }
            } else {
                
                netpath = search.filename + '/' + filename;

                file = new File(netpath);
                if (!file.canRead())
                    continue;

                

                return new RandomAccessFile(file, "r");
            }
        }
        
        return null;
    }


    private static final int MAX_READ =
            
            1024*1024; 

    /**
     * Read
     * 
     * Properly handles partial reads
     */
    public static void Read(byte[] buffer, int len, RandomAccessFile f) {

        int offset = 0;
        int read = 0;
        
        int remaining = len;

        while (remaining != 0) {
            int block = Math.min(remaining, MAX_READ);
            try {
                read = f.read(buffer, offset, block);
            } catch (IOException e) {
                Com.Error(Defines.ERR_FATAL, e.toString());
            }

            switch (read) {
                case 0 -> Com.Error(Defines.ERR_FATAL, "FS_Read: 0 bytes read");
                case -1 -> Com.Error(Defines.ERR_FATAL, "FS_Read: -1 bytes read");
            }
            
            
            
            remaining -= read;
            offset += read;
        }
    }

    /*
     * LoadFile
     * 
     * Filename are reletive to the quake search path a null buffer will just
     * return the file content as byte[]
     */
    public static byte[] LoadFile(String path) {


        int index = path.indexOf('\0');
        if (index != -1)
            path = path.substring(0, index);


        int len = FileLength(path);

        if (len < 1)
            return null;

        byte[] buf = null;
        try {

            RandomAccessFile file = FOpenFile(path);

            buf = new byte[len];
            file.readFully(buf);
            file.close();
        } catch (IOException e) {
            Com.Error(Defines.ERR_FATAL, e.toString());
        }
        return buf;
    }

    /*
     * LoadMappedFile
     * 
     * Filename are reletive to the quake search path a null buffer will just
     * return the file content as ByteBuffer (memory mapped)
     */
    public static ByteBuffer LoadMappedFile(String filename) {

        file_from_pak = 0;

        FileInputStream input = null;
        FileChannel channel = null;
        try {


            ByteBuffer buffer;
            int fileLength;
            File file;
            String netpath;
            for (filelink_t link : fs_links) {
                if (filename.regionMatches(0, link.from, 0, link.fromlength)) {
                    netpath = link.to + filename.substring(link.fromlength);
                    file = new File(netpath);
                    if (file.canRead()) {
                        input = new FileInputStream(file);
                        channel = input.getChannel();
                        fileLength = (int) channel.size();
                        buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0,
                                fileLength);
                        input.close();
                        return buffer;
                    }
                    return null;
                }
            }

            
            
            
            for (searchpath_t search = fs_searchpaths; search != null; search = search.next) {
                
                if (search.pack != null) {

                    pack_t pak = search.pack;
                    filename = filename.toLowerCase();
                    packfile_t entry = (packfile_t) pak.files.get(filename);

                    if (entry != null) {
                        
                        file_from_pak = 1;
                        
                        
                        file = new File(pak.filename);
                        if (!file.canRead())
                            Com.Error(Defines.ERR_FATAL, "Couldn't reopen "
                                    + pak.filename);
                        if (pak.handle == null || !pak.handle.getFD().valid()) {
                            
                            pak.handle = new RandomAccessFile(pak.filename, "r");
                        }
                        
                        if (pak.backbuffer == null) {
                            channel = pak.handle.getChannel();
                            pak.backbuffer = channel.map(
                                    FileChannel.MapMode.READ_ONLY, 0,
                                    pak.handle.length());
                            channel.close();
                        }
                        pak.backbuffer.position(entry.filepos);
                        buffer = pak.backbuffer.slice();
                        buffer.limit(entry.filelen);
                        return buffer;
                    }
                } else {
                    
                    netpath = search.filename + '/' + filename;

                    file = new File(netpath);
                    if (!file.canRead())
                        continue;

                    
                    input = new FileInputStream(file);
                    channel = input.getChannel();
                    fileLength = (int) channel.size();
                    buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0,
                            fileLength);
                    input.close();
                    return buffer;
                }
            }
        } catch (Exception e) {
        }
        try {
            if (input != null)
                input.close();
            else if (channel != null && channel.isOpen())
                channel.close();
        } catch (IOException ioe) {
        }
        return null;
    }

    /*
     * FreeFile
     */
    public static void FreeFile(byte[] buffer) {
        
    }

    private static final int IDPAKHEADER = (('K' << 24) + ('C' << 16) + ('A' << 8) + 'P');

    static class dpackheader_t {
        int ident; 

        int dirofs;

        int dirlen;
    }

    private static final int MAX_FILES_IN_PACK = 4096;

    
    private static final byte[] tmpText = new byte[packfile_t.NAME_SIZE];

    /*
     * LoadPackFile
     * 
     * Takes an explicit (not game tree related) path to a pak file.
     * 
     * Loads the header and directory, adding the files at the beginning of the
     * list so they override previous pack files.
     */
    private static pack_t LoadPackFile(String packfile) {

        HashMap newfiles;
        RandomAccessFile file;
        int numpackfiles;


        try {
        	file = new RandomAccessFile(packfile, "r");
        	FileChannel fc = file.getChannel();
            ByteBuffer packhandle = fc.map(FileChannel.MapMode.READ_ONLY, 0, file.length());
            packhandle.order(ByteOrder.LITTLE_ENDIAN);
 
            fc.close();
            
            if (packhandle == null || packhandle.limit() < 1)
                return null;

            dpackheader_t header = new dpackheader_t();
            header.ident = packhandle.getInt();
            header.dirofs = packhandle.getInt();
            header.dirlen = packhandle.getInt();

            if (header.ident != IDPAKHEADER)
                Com.Error(Defines.ERR_FATAL, packfile + " is not a packfile");

            numpackfiles = header.dirlen / packfile_t.SIZE;

            if (numpackfiles > MAX_FILES_IN_PACK)
                Com.Error(Defines.ERR_FATAL, packfile + " has " + numpackfiles
                        + " files");

            newfiles = new HashMap(numpackfiles);

            packhandle.position(header.dirofs);

            
            packfile_t entry;

            for (int i = 0; i < numpackfiles; i++) {
                packhandle.get(tmpText);

                entry = new packfile_t();
                entry.name = new String(tmpText).trim();
                entry.filepos = packhandle.getInt();
                entry.filelen = packhandle.getInt();

                newfiles.put(entry.name.toLowerCase(), entry);
            }

        } catch (IOException e) {
            Com.DPrintf(e.getMessage() + '\n');
            return null;
        }

        pack_t pack = new pack_t();
        pack.filename = packfile;
        pack.handle = file;
        pack.numfiles = numpackfiles;
        pack.files = newfiles;

        Com.Printf("Added packfile " + packfile + " (" + numpackfiles
                + " files)\n");

        return pack;
    }

    /*
     * AddGameDirectory
     * 
     * Sets fs_gamedir, adds the directory to the head of the path, then loads
     * and adds pak1.pak pak2.pak ...
     */
    private static void AddGameDirectory(String dir) {

        fs_gamedir = dir;


        searchpath_t search = new searchpath_t();
        search.filename = dir;
        if (fs_searchpaths != null) {
            search.next = fs_searchpaths.next;
            fs_searchpaths.next = search;
        } else {
            fs_searchpaths = search;
        }

        
        
        
        for (int i = 0; i < 10; i++) {
            String pakfile = dir + "/pak" + i + ".pak";
            if (!(new File(pakfile).canRead()))
                continue;

            pack_t pak = LoadPackFile(pakfile);
            if (pak == null)
                continue;

            search = new searchpath_t();
            search.pack = pak;
            search.filename = "";
            search.next = fs_searchpaths;
            fs_searchpaths = search;
        }
    }

    /*
     * Gamedir
     * 
     * Called to find where to write a file (demos, savegames, etc)
     * this is modified to <user.home>/.jake2 
     */
    public static String Gamedir() {
        return (fs_userdir != null) ? fs_userdir : Globals.BASEDIRNAME;
    }

    /*
     * BaseGamedir
     * 
     * Called to find where to write a downloaded file
     */
    public static String BaseGamedir() {
        return (fs_gamedir != null) ? fs_gamedir : Globals.BASEDIRNAME;
    }

    /*
     * ExecAutoexec
     */
    public static void ExecAutoexec() {
        String dir = fs_userdir;

        String name;
        if (dir != null && !dir.isEmpty()) {
            name = dir + "/autoexec.cfg";
        } else {
            name = fs_basedir.string + '/' + Globals.BASEDIRNAME
                    + "/autoexec.cfg";
        }

        int canthave = Defines.SFF_SUBDIR | Defines.SFF_HIDDEN
                | Defines.SFF_SYSTEM;

        if (Sys.FindAll(name, 0, canthave) != null) {
            Cbuf.AddText("exec autoexec.cfg\n");
        }
    }

    /*
     * SetGamedir
     * 
     * Sets the gamedir and path to a different directory.
     */
    public static void SetGamedir(String dir) {

        boolean b = Stream.of("..", "/", "\\", ":").anyMatch(dir::contains);
        if (b) {
            Com.Printf("Gamedir should be a single filename, not a path\n");
            return;
        }

        
        
        
        while (fs_searchpaths != fs_base_searchpaths) {
            if (fs_searchpaths.pack != null) {
                try {
                    fs_searchpaths.pack.handle.close();
                } catch (IOException e) {
                    Com.DPrintf(e.getMessage() + '\n');
                }
                
                fs_searchpaths.pack.files.clear();
                fs_searchpaths.pack.files = null;
                fs_searchpaths.pack = null;
            }
            searchpath_t next = fs_searchpaths.next;
            fs_searchpaths = null;
            fs_searchpaths = next;
        }

        
        
        
        if ((Globals.dedicated != null) && (Globals.dedicated.value == 0.0f))
            Cbuf.AddText("vid_restart\nsnd_restart\n");

        fs_gamedir = fs_basedir.string + '/' + dir;

        if (dir.equals(Globals.BASEDIRNAME) || (dir.isEmpty())) {
            Cvar.FullSet("gamedir", "", CVAR_SERVERINFO | CVAR_NOSET);
            Cvar.FullSet("game", "", CVAR_LATCH | CVAR_SERVERINFO);
        } else {
            Cvar.FullSet("gamedir", dir, CVAR_SERVERINFO | CVAR_NOSET);
            if (fs_cddir.string != null && !fs_cddir.string.isEmpty())
                AddGameDirectory(fs_cddir.string + '/' + dir);

            AddGameDirectory(fs_basedir.string + '/' + dir);
        }
    }

    /*
     * Link_f
     * 
     * Creates a filelink_t
     */
    private static void Link_f() {

        if (Cmd.Argc() != 3) {
            Com.Printf("USAGE: link <from> <to>\n");
            return;
        }


        filelink_t entry;
        for (Iterator it = fs_links.iterator(); it.hasNext();) {
            entry = (filelink_t) it.next();

            if (entry.from.equals(Cmd.Argv(1))) {
                if (Cmd.Argv(2).length() < 1) {
                    
                    it.remove();
                    return;
                }
                entry.to = Cmd.Argv(2);
                return;
            }
        }

        
        if (!Cmd.Argv(2).isEmpty()) {
            entry = new filelink_t();
            entry.from = Cmd.Argv(1);
            entry.fromlength = entry.from.length();
            entry.to = Cmd.Argv(2);
            fs_links.add(entry);
        }
    }

    /*
     * ListFiles
     */
    public static String[] ListFiles(String findname, int musthave, int canthave) {
        String[] list = null;

        File[] files = Sys.FindAll(findname, musthave, canthave);

        if (files != null) {
            list = new String[files.length];
            for (int i = 0; i < files.length; i++) {
                list[i] = files[i].getPath();
            }
        }

        return list;
    }

    /*
     * Dir_f
     */
    private static void Dir_f() {
        String wildcard = "*.*";

        if (Cmd.Argc() != 1) {
            wildcard = Cmd.Argv(1);
        }

        String findname = null;
        String path = null;
        while ((path = NextPath(path)) != null) {
            String tmp = findname;

            findname = path + '/' + wildcard;

            if (tmp != null)
                SLASHES.matcher(tmp).replaceAll("/");

            Com.Printf("Directory of " + findname + '\n');
            Com.Printf("----\n");

            String[] dirnames = ListFiles(findname, 0, 0);

            if (dirnames != null) {
                int index;
                for (String dirname : dirnames) {
                    if ((index = dirname.lastIndexOf('/')) > 0) {
                        Com.Printf(dirname.substring(index + 1) + '\n');
                    } else {
                        Com.Printf(dirname + '\n');
                    }
                }
            }

            Com.Printf("\n");
        }
    }

    /*
     * Path_f
     */
    private static void Path_f() {

        Com.Printf("Current search path:\n");
        for (searchpath_t s = fs_searchpaths; s != null; s = s.next) {
            if (s == fs_base_searchpaths)
                Com.Printf("----------\n");
            if (s.pack != null)
                Com.Printf(s.pack.filename + " (" + s.pack.numfiles
                        + " files)\n");
            else
                Com.Printf(s.filename + '\n');
        }

        Com.Printf("\nLinks:\n");
        for (filelink_t link : fs_links) {
            Com.Printf(link.from + " : " + link.to + '\n');
        }
    }

    /*
     * NextPath
     * 
     * Allows enumerating all of the directories in the search path
     */
    public static String NextPath(String prevpath) {

        if (prevpath == null || prevpath.isEmpty())
            return fs_gamedir;

        String prev = fs_gamedir;
        for (searchpath_t s = fs_searchpaths; s != null; s = s.next) {
            if (s.pack != null)
                continue;

            if (prevpath == prev)
                return s.filename;

            prev = s.filename;
        }

        return null;
    }

    /*
     * InitFilesystem
     */
    public static void InitFilesystem() {
        Cmd.AddCommand("path", new xcommand_t() {
            @Override
            public void execute() {
                Path_f();
            }
        });
        Cmd.AddCommand("link", new xcommand_t() {
            @Override
            public void execute() {
                Link_f();
            }
        });
        Cmd.AddCommand("dir", new xcommand_t() {
            @Override
            public void execute() {
                Dir_f();
            }
        });

        fs_userdir = System.getProperty("user.home") + "/.jake2";
        FS.CreatePath(fs_userdir + '/');
        FS.AddGameDirectory(fs_userdir);

        
        
        
        
        fs_basedir = Cvar.Get("basedir", ".", CVAR_NOSET);

        
        
        
        
        

        setCDDir();

        
        
        
        AddGameDirectory(fs_basedir.string + '/' + Globals.BASEDIRNAME);

        
        markBaseSearchPaths();

        
        fs_gamedirvar = Cvar.Get("game", "", CVAR_LATCH | CVAR_SERVERINFO);

        if (!fs_gamedirvar.string.isEmpty())
            SetGamedir(fs_gamedirvar.string);
    }

    /**
     * set baseq2 directory
     */
    static void setCDDir() {
        fs_cddir = Cvar.Get("cddir", "", CVAR_ARCHIVE);
        if (!fs_cddir.string.isEmpty())
            AddGameDirectory(fs_cddir.string);
    }
    
    static void markBaseSearchPaths() {
        
        fs_base_searchpaths = fs_searchpaths;
    }

    
    /*
     * Developer_searchpath
     */
    public static int Developer_searchpath(int who) {


        for (searchpath_t s = fs_searchpaths; s != null; s = s.next) {
            if (s.filename.contains("xatrix"))
                return 1;

            if (s.filename.contains("rogue"))
                return 2;
        }

        return 0;
    }
}