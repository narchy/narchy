package jake2.sys;

import com.jogamp.newt.Window;
import com.jogamp.newt.event.KeyEvent;
import com.jogamp.newt.event.MouseEvent;
import jake2.client.Key;

public final class NEWTKBD extends KBD
{
	public static final InputListener listener = new InputListener();
	
	private static Window c;
	
	private static int win_w2;
	private static int win_h2;
	
	@Override
    public void Init() {
	}

        
        public static void Init(Window window) {
            c = window;
            handleCreateAndConfigureNotify(window);
        }

	@Override
    public void Update() {
		
		HandleEvents();
	}

	@Override
    public void Close() {
	}
	
	private void HandleEvents() 
	{

        Jake2InputEvent event;
		while ( (event=InputListener.nextEvent()) != null ) {
	                Window eventWin = null;
	                Object source = event.ev.getSource();
	                if(source instanceof Window) {
	                    eventWin = (Window)source;
	                }
            int key;
            switch(event.type) {
				case Jake2InputEvent.KeyPress:
				case Jake2InputEvent.KeyRelease:
					Do_Key_Event(XLateKeyCode((KeyEvent)event.ev), event.type == Jake2InputEvent.KeyPress);
					break;
				    
				case Jake2InputEvent.MotionNotify:




					if (IN.mouse_active) {
						mx = (((MouseEvent)event.ev).getX() - win_w2) * 2;
						my = (((MouseEvent)event.ev).getY() - win_h2) * 2;
					} else {
						mx = 0;
						my = 0;
					}
					break;
					
				case Jake2InputEvent.ButtonPress:
					key = mouseEventToKey((MouseEvent)event.ev); 
					Do_Key_Event(key, true);
					break;
 
				case Jake2InputEvent.ButtonRelease:
					key = mouseEventToKey((MouseEvent)event.ev); 
					Do_Key_Event(key, false);
					break;
					
				case Jake2InputEvent.WheelMoved:
					float dir = ((MouseEvent)event.ev).getRotation()[0];
					if (dir > 0) {
						Do_Key_Event(Key.K_MWHEELDOWN, true);
						Do_Key_Event(Key.K_MWHEELDOWN, false);
					} else {
						Do_Key_Event(Key.K_MWHEELUP, true);
						Do_Key_Event(Key.K_MWHEELUP, false);					    
					}
					break;
					 
				case Jake2InputEvent.CreateNotify :
				case Jake2InputEvent.ConfigureNotify :				    
                                        handleCreateAndConfigureNotify(eventWin);
					break;
			}
		}
            
		if (mx != 0 || my != 0) {
			
			c.warpPointer(c.getWidth()/2, c.getHeight()/2);
		}		
	}

        private static void handleCreateAndConfigureNotify(Window component) {
            if(null != component) {
                win_w2 = component.getWidth() / 2;
                win_h2 = component.getHeight() / 2;
            }
        }

	
	
	
	private static int mouseEventToKey(MouseEvent ev) {
        return switch (ev.getButton()) {
            case MouseEvent.BUTTON3 -> Key.K_MOUSE2;
            case MouseEvent.BUTTON2 -> Key.K_MOUSE3;
            default -> Key.K_MOUSE1;
        };
	}

	private static int XLateKeyCode(KeyEvent ev) { 
		int code = ev.getKeyCode();
                int key = 0;
		switch (code) {
			case KeyEvent.VK_PAGE_UP -> key = Key.K_PGUP;
			case KeyEvent.VK_PAGE_DOWN -> key = Key.K_PGDN;
			case KeyEvent.VK_HOME -> key = Key.K_HOME;
			case KeyEvent.VK_END -> key = Key.K_END;
			case KeyEvent.VK_LEFT -> key = Key.K_LEFTARROW;
			case KeyEvent.VK_RIGHT -> key = Key.K_RIGHTARROW;
			case KeyEvent.VK_DOWN -> key = Key.K_DOWNARROW;
			case KeyEvent.VK_UP -> key = Key.K_UPARROW;
			case KeyEvent.VK_ESCAPE -> key = Key.K_ESCAPE;
			case KeyEvent.VK_ENTER -> key = Key.K_ENTER;
			case KeyEvent.VK_TAB -> key = Key.K_TAB;
			case KeyEvent.VK_F1 -> key = Key.K_F1;
			case KeyEvent.VK_F2 -> key = Key.K_F2;
			case KeyEvent.VK_F3 -> key = Key.K_F3;
			case KeyEvent.VK_F4 -> key = Key.K_F4;
			case KeyEvent.VK_F5 -> key = Key.K_F5;
			case KeyEvent.VK_F6 -> key = Key.K_F6;
			case KeyEvent.VK_F7 -> key = Key.K_F7;
			case KeyEvent.VK_F8 -> key = Key.K_F8;
			case KeyEvent.VK_F9 -> key = Key.K_F9;
			case KeyEvent.VK_F10 -> key = Key.K_F10;
			case KeyEvent.VK_F11 -> key = Key.K_F11;
			case KeyEvent.VK_F12 -> key = Key.K_F12;
			case KeyEvent.VK_BACK_SPACE -> key = Key.K_BACKSPACE;
			case KeyEvent.VK_DELETE -> key = Key.K_DEL;
			case KeyEvent.VK_PAUSE -> key = Key.K_PAUSE;
			case KeyEvent.VK_SHIFT -> key = Key.K_SHIFT;
			case KeyEvent.VK_CONTROL -> key = Key.K_CTRL;
			case KeyEvent.VK_ALT, KeyEvent.VK_ALT_GRAPH -> key = Key.K_ALT;
			case KeyEvent.VK_INSERT -> key = Key.K_INS;
			case KeyEvent.VK_QUOTE, KeyEvent.VK_CIRCUMFLEX, KeyEvent.VK_BACK_QUOTE -> key = '`';
			case KeyEvent.VK_SPACE -> key = Key.K_SPACE;
			default -> {
				if (KeyEvent.VK_0 <= code && code <= KeyEvent.VK_9) {
					key = code - KeyEvent.VK_0 + '0';
				}
				if (KeyEvent.VK_A <= code && code <= KeyEvent.VK_Z) {
					key = code - KeyEvent.VK_A + 'a';
				}
			}
		}
		if (key > 255) key = 0;

		return key;
	}
	
	@Override
    public void Do_Key_Event(int key, boolean down) {
		Key.Event(key, down, Timer.Milliseconds());
	}
	
	private static void centerMouse() {
	    c.warpPointer(c.getWidth()/2, c.getHeight()/2);
	}
	
	@Override
    public void installGrabs()
	{
	    /*
		if (emptyCursor == null) {
			ImageIcon emptyIcon = new ImageIcon(new byte[0]);
			emptyCursor = c.getToolkit().createCustomCursor(emptyIcon.getImage(), new Point(0, 0), "emptyCursor");
		}
		c.setCursor(emptyCursor);
	     */
	    c.setPointerVisible(false);
	    centerMouse();
	}
	
	@Override
    public void uninstallGrabs()
	{
	    /*
		c.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
		*/
	    c.setPointerVisible(true);
	}
}
