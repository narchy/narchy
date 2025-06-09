package nars.experiment.checkers;

import java.awt.*;

/**
 * @author Arjen Hoogesteger
 * @author Elio Tolhoek
 * @version 0.1
 */
public class Piece {
    protected static final Color LIGHT = new Color(251, 249, 246);
    protected static final Color DARK = new Color(255, 15, 15);

    private final Color color;

    /**
     * @return
     */
    public static Piece createLightPiece() {
        return new Piece(LIGHT);
    }

    /**
     * @return
     */
    public static Piece createDarkPiece() {
        return new Piece(DARK);
    }

    /**
     * Creates a new Piece instance.
     *
     * @param color the piece's color
     */
    protected Piece(Color color) {
        this.color = color;
    }

    /**
     * Draws the piece.
     *
     * @param g the associated Graphics instance
     */
    public void draw(Graphics g) {

        g.setColor(color);
        int radius = 20;
        int y = 30;
        int x = 30;
        g.fillOval(x - radius, y - radius, radius * 2, radius * 2);
    }

    /**
     * Returns the piece's color.
     *
     * @return the color
     */
    public Color getColor() {
        return color;
    }

    /**
     * @return
     */
    public boolean isDark() {
        return color.equals(DARK);
    }

    /**
     * @return
     */
    public boolean isLight() {
        return color.equals(LIGHT);
    }

    @Override
    public String toString() {
        if (isDark())
            return "DARK";
        else if (isLight())
            return "LIGHT";
        else
            return "UNKNOWN";
    }
}
