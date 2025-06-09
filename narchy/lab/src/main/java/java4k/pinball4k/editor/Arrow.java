package java4k.pinball4k.editor;

import java.awt.*;
import java.lang.reflect.Field;

public class Arrow extends LevelObject {
	public static final int radius = 40;
	public static int nextId = 1;
	public int id = nextId++;
	public int angle;
	private final ArrowHandle handle;

    public Arrow(Point p1, Point p2) {
		this.p = new Point(p1);
		int dx = p2.x - p1.x;
		int dy = p2.y - p1.y;
		angle = (int) Math.toDegrees(Math.atan2(dy, dx));
		if (angle < 0) {
			angle += 360;
		}
		angle /= 2;

        Arrow outerInstance = this;
    	handle = new ArrowHandle();
    	handles.add(handle);
	}
	
	/**
	 * Draws itself to the specified graphics object
	 * @param g where to draw
	 */
	@Override
    public void draw(Graphics2D g, LevelPanel levelPanel) {
		g.setColor(Color.WHITE);
		g.fillArc(p.x, p.y, radius * 2, radius * 2, angle*2, 45);
	}	
	
	/**
	 * Draws the handles to the specified graphics object
	 * @param g where to draw
	 */
	@Override
    public void drawHandles(Graphics2D g, LevelPanel levelPanel) {
    	handle.draw(g, levelPanel);
	}	
	
	/**
	 * The public properties to show in the editor ui.
	 * @return the properties to show
	 */
	@Override
    public Field[] getProperties() {
		return getFields("visible", "collidable", "score", "bounce", "behaviorId", "p", "angle");
	}	
	
	@Override
    public int getSortValue() {
		return angle;
	}

	
	class ArrowHandle extends Handle {

		/**
		 * Gets the center of the handle.
		 * @return the center
		 */
		@Override
        public Point getCenter() {
			return new Point(p.x+40, p.y+40);
		}
		
		/**
		 * The handle is dragged to the specified location
		 * @param pos where the handle was dragged to 
		 */
		@Override
        public void dragged(int dx, int dy) {
			p.translate(dx, dy);
		}		
		
		/**
		 * Gets the level object the hangle controlls. Can be null.
		 * @return the level object
		 */
		@Override
        public LevelObject getLevelObject() {
			return Arrow.this;
		}
	}
	
	/*
	 *  (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		return "Arrow(" + id + ")";
	}
}
