package jake2.render.opengl;

import com.jogamp.nativewindow.util.Dimension;
import com.jogamp.newt.MonitorMode;
import jake2.qcommon.xcommand_t;

import java.util.List;

public interface GLDriver {
    
    boolean init(int xpos, int ypos);
    
    int setMode(Dimension dim, int mode, boolean fullscreen);
    
    void shutdown();
    
    /** @return true if successful, otherwise false. */
    boolean beginFrame(float camera_separation);
    
    /** Performs <code>swapBuffers()</code>, ticks the fps counter and performs <code>QUIT</code> if requested. */ 
    void endFrame();

    void appActivate(boolean activate);
    
    void enableLogging(boolean enable);
    
    void logNewFrame();
    
    List<MonitorMode> getModeList();

    default void updateScreen(xcommand_t callback) {
        callback.execute();
    }

    void screenshot();
    
}
