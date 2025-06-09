package nars.experiment.mario.mapedit;

import nars.experiment.mario.Art;
import nars.experiment.mario.LevelRenderer;
import nars.experiment.mario.level.Level;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;


public class LevelEditView extends JComponent implements MouseListener, MouseMotionListener {
    private static final long serialVersionUID = -7696446733303717142L;

    private LevelRenderer levelRenderer;
    private Level level;

    private int xTile = -1;
    private int yTile = -1;
    private final TilePicker tilePicker;

    public LevelEditView(TilePicker tilePicker) {
        this.tilePicker = tilePicker;
        level = new Level(256, 15);
        Dimension size = new Dimension(level.width * 16, level.height * 16);
        setPreferredSize(size);
        setMinimumSize(size);
        setMaximumSize(size);

        addMouseListener(this);
        addMouseMotionListener(this);
    }

    public void setLevel(Level level) {
        this.level = level;
        Dimension size = new Dimension(level.width * 16, level.height * 16);
        setPreferredSize(size);
        setMinimumSize(size);
        setMaximumSize(size);
        repaint();
        levelRenderer.setLevel(level);
    }

    public Level getLevel() {
        return level;
    }

    @Override
    public void addNotify() {
        super.addNotify();
        Art.init(getGraphicsConfiguration());
        levelRenderer = new LevelRenderer(level, getGraphicsConfiguration(), level.width * 16, level.height * 16);
        levelRenderer.renderBehaviors = true;
    }

    @Override
    public void paintComponent(Graphics g) {
        g.setColor(new Color(0x8090ff));
        g.fillRect(0, 0, level.width * 16, level.height * 16);
        levelRenderer.render(g, 0, 0);
        g.setColor(Color.BLACK);
        g.drawRect(xTile * 16 - 1, yTile * 16 - 1, 17, 17);
    }

    @Override
    public void mouseClicked(MouseEvent e) {
    }

    @Override
    public void mouseEntered(MouseEvent e) {
    }

    @Override
    public void mouseExited(MouseEvent e) {
        xTile = -1;
        yTile = -1;
        repaint();
    }

    @Override
    public void mousePressed(MouseEvent e) {
        xTile = e.getX() / 16;
        yTile = e.getY() / 16;

        if (e.getButton() == 3) {
            tilePicker.setPickedTile(level.getBlock(xTile, yTile));
        } else {
            level.setBlock(xTile, yTile, tilePicker.pickedTile);
            levelRenderer.repaint(xTile - 1, yTile - 1, 3, 3);

            repaint();
        }
    }

    @Override
    public void mouseReleased(MouseEvent e) {
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        xTile = e.getX() / 16;
        yTile = e.getY() / 16;

        level.setBlock(xTile, yTile, tilePicker.pickedTile);
        levelRenderer.repaint(xTile - 1, yTile - 1, 3, 3);

        repaint();
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        xTile = e.getX() / 16;
        yTile = e.getY() / 16;
        repaint();
    }
}