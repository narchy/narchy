/*
 * Java Arcade Learning Environment (A.L.E) Agent
 *  Copyright (C) 2011-2012 Marc G. Bellemare <mgbellemare@ualberta.ca>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package ale.agents;

import ale.gui.AbstractUI;
import ale.gui.AgentGUI;
import ale.gui.NullUI;
import ale.io.ALEPipes;
import ale.io.ConsoleRAM;
import ale.io.RLData;
import ale.screen.*;

import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * An abstract agent class. New agents can be created by extending this class
 * and implementing its abstract methods.
 *
 * @author Marc G. Bellemare
 */
abstract class AbstractAgent {
    /**
     * Used to convert ALE screen data to GUI images
     */
    final ScreenConverter converter;

    /**
     * The UI used for displaying images and receiving actions
     */
    final AbstractUI ui;
    /**
     * The I/O object used to communicate with ALE
     */
    final ALEPipes io;

    /**
     * Whether to use a GUI
     */
    private final boolean useGUI;
    /**
     * If non-null, we communicate via named pipes rather than stdin/stdout
     */
    private final String namedPipesBasename;

    /**
     * Create a new agent that communicates with ALE via stdin/out and
     * uses the graphical user interface.
     */
    AbstractAgent() {
        this(true, null);
    }

    /**
     * Create a new agent with the specified parameters. The user can specify
     * the base name for two FIFO pipes used to communicate with ALE. If
     * namedPipesBasename is not null, then the files namedPipesBasename+"_in"
     * and namedPipesBasename+"_out" are read and written to by the agent.
     * See ALE documentation for more details on running with named pipes.
     *
     * @param useGUI             If true, a GUI is used to display received screen data.
     * @param namedPipesBasename If non-null, the base filename for the two FIFO
     *                           files used to communicate with ALE.
     */
    AbstractAgent(boolean useGUI, String namedPipesBasename) {
        this.useGUI = useGUI;
        this.namedPipesBasename = namedPipesBasename;

        // Create the color palette we will use to interpret ALE data
        ColorPalette palette = makePalette("NTSC");

        // Create an object to convert indexed images to Java images
        converter = new ScreenConverter(palette);

        ui = this.useGUI ? new AgentGUI() : new NullUI();

        try {
            ALEPipes io;
            // Initialize the pipes; use named pipes if requested
            if (this.namedPipesBasename != null)
                io = new ALEPipes(this.namedPipesBasename + "out", this.namedPipesBasename + "in");
            else
                io = new ALEPipes();

            // Determine which information to request from ALE
            io.setUpdateScreen(this.useGUI || wantsScreenData());
            io.setUpdateRam(wantsRamData());
            io.setUpdateRL(wantsRLData());
            io.initPipes();
            this.io = io;
        } catch (IOException e) {
            throw new RuntimeException(e);
//            System.err.println("Could not initialize pipes: " + e.getMessage());
//            System.exit(-1);
        }
    }

    /**
     * Create a color palette used to display the screen. The currently available
     * choices are NTSC (128 colors) and SECAM (8 colors).
     *
     * @param paletteName The name of the palette (NTSC or SECAM).
     */
    private static ColorPalette makePalette(String paletteName) {
        if (paletteName.equals("NTSC"))
            return new NTSCPalette();
        else if (paletteName.equals("SECAM"))
            return new SECAMPalette();
        else
            throw new IllegalArgumentException("Invalid palette: " + paletteName);
    }

    /**
     * The main program loop. In turn, we will obtain a new screen from ALE,
     * pass it on to the agent and send back an action (which may be a reset
     * request).
     */
    void run() {
        boolean done = false;

        // Loop until we're done
        while (!done) {
            // Obtain relevant data from ALE
            done = io.observe();
            // The I/O channel will return true once EOF is received
            if (done) break;

            // Obtain the screen matrix
            ScreenMatrix screen = io.getScreen();
            // Pass it on to UI
            updateImage(screen);
            // ... and to the agent
            observe(screen, io.getRAM(), io.getRLData());

            // Request an action from the agent
            int action = selectAction();
            // Send it back to ALE
            done = io.act(action);

            // Ask the agent whether it wants us to pause
            long pauseLength = getPauseLength();
            // If so, pause!
            if (pauseLength > 0) {
                pause(pauseLength);
            }

            // The agent also tells us when to terminate
            done |= shouldTerminate();
        }

        // Clean up the GUI
        ui.die();
    }

    /**
     * Internal method to update the image displayed in the GUI.
     */
    private void updateImage(ScreenMatrix currentScreen) {
        // We know that the NullUI does not want image data, so don't spend time
        //  converting the image
        if (ui instanceof NullUI) {
            ui.updateFrameCount();
            return;
        }

        // Convert the screen matrix to an image
        BufferedImage img = converter.convert(currentScreen);

        // Provide the new image to the UI
        ui.updateFrameCount();
        ui.setImage(img);
        ui.refresh();
    }

    private static void pause(long waitTime) {
        try {
            Thread.sleep(waitTime);
        } catch (Exception e) {
        }
    }

    /**
     * Returns how long to pause for, in milliseconds, before the next time step.
     */
    protected abstract long getPauseLength();

    /**
     * Returns the agent's next action.
     */
    protected abstract int selectAction();

    /**
     * Provides the agent with the latest screen, RAM and RL data.
     */
    protected abstract void observe(ScreenMatrix screen, ConsoleRAM ram, RLData rlData);

    /**
     * Returns true to indicate that we should exit the program.
     */
    protected abstract boolean shouldTerminate();

    /**
     * Returns true if we want to receive the screen matrix from ALE.
     */
    protected abstract boolean wantsScreenData();

    /**
     * Returns true if we want to receive the RAM from ALE.
     */
    protected abstract boolean wantsRamData();

    /**
     * Returns true if we want to receive RL data from ALE.
     */
    protected abstract boolean wantsRLData();
}
