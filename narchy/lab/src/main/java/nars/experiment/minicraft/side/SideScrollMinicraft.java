/*
 * Copyright 2012 Jonathan Leahey
 *
 * This file is part of Minicraft
 *
 * Minicraft is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * Minicraft is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Minicraft. If not, see http:
 */

package nars.experiment.minicraft.side;

import nars.experiment.minicraft.side.awtgraphics.AwtEventsHandler;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;


public class SideScrollMinicraft {

    public final GraphicsHandler gfx;
    public final AwtEventsHandler input;
    int worldHeight = 256;
    private int worldWidth = 512;
    public boolean leftClick;
    public boolean rightClick;
    public boolean paused = true;

    public ArrayList<Entity> entities = new ArrayList<>();

    private int tileSize = 64;

    private int breakingTicks;
    private Int2 breakingPos;

    private Sprite builderIcon;
    private Sprite minerIcon;
    private Sprite[] breakingSprites;
    private Sprite fullHeart;
    private Sprite halfHeart;
    private Sprite emptyHeart;
    private Sprite bubble;
    private Sprite emptyBubble;

    public boolean viewFPS;
    private boolean inMenu = true;
    private final MainMenu menu;
    public long ticksRunning;
    private final Random random = new Random();

    public Player player;
    public World world;


    public final Int2 screenMousePos = new Int2(0, 0);

    /**
     * Construct our game and set it running.
     */
    public SideScrollMinicraft() {
        menu = new MainMenu(this);
        gfx = GraphicsHandler.get();
        gfx.init(this);
        input = new AwtEventsHandler(this);
        System.gc();
    }

    /**
     * Start a fresh game, this should clear out any old data and
     * create a new setAt.
     */
    public void startGame(boolean load, int width) {
        inMenu = false;
        if (load) {
            System.out.println("Loading world, width: " + worldWidth);
        } else {
            System.out.println("Creating world, width: " + width);
            worldWidth = width;
        }

        entities.clear();
        if (load) {

            load = SaveLoad.doLoad(this);
        }

        if (load) {
            for (Entity entity : entities) {
                if (entity instanceof Player) {
                    player = (Player) entity;
                    player.widthPX = 7 * (tileSize / 8);
                    player.heightPX = 14 * (tileSize / 8);
                }
            }
        }
        if (!load) {

            world = new World(worldWidth, worldHeight, random);
            player = new Player(true, world.spawnLocation.x, world.spawnLocation.y,
                    7 * (tileSize / 8), 14 * (tileSize / 8));
            entities.add(player);
            if (Constants.DEBUG) {
                player.giveItem(Constants.itemTypes.get((char) 175).clone(), 1);
                player.giveItem(Constants.itemTypes.get((char) 88).clone(), 1);
                player.giveItem(Constants.itemTypes.get((char) 106).clone(), 64);
            }
        }


        SpriteStore ss = SpriteStore.get();
        builderIcon = ss.getSprite("sprites/other/builder.png");
        minerIcon = ss.getSprite("sprites/other/miner.png");
        fullHeart = ss.getSprite("sprites/other/full_heart.png");
        halfHeart = ss.getSprite("sprites/other/half_heart.png");
        emptyHeart = ss.getSprite("sprites/other/empty_heart.png");
        bubble = ss.getSprite("sprites/other/bubble.png");

        emptyBubble = ss.getSprite("sprites/other/bubble_pop2.png");

        breakingSprites = new Sprite[8];
        for (int i = 0; i < 8; i++) {
            breakingSprites[i] = ss.getSprite("sprites/tiles/break" + i + ".png");
        }


        System.gc();
    }

    public static void drawCenteredX(GraphicsHandler g, Sprite s, int top, int width, int height) {
        s.draw(g, GraphicsHandler.getScreenWidth() / 2 - width / 2, top, width, height);
    }

    public void start(boolean delay) {
        long lastLoopTime = System.currentTimeMillis();

        if (Constants.DEBUG) {
            startGame(false, 512);
        }


        boolean gameRunning = true;
        while (gameRunning) {
            long delta = 0;
            if (delay) {
                ticksRunning++;

                lastLoopTime = SystemTimer.getTime();
            }

            frame();


            if (delay)
                SystemTimer.sleep(lastLoopTime + 16 - SystemTimer.getTime());
        }
    }

    public float frame() {
        GraphicsHandler g = GraphicsHandler.get();
        g.startDrawing();

        if (inMenu) {
            menu.draw(g);
            drawMouse(g, screenMousePos);
            g.finishDrawing();

            return 0;
        }

        int screenWidth = GraphicsHandler.getScreenWidth();
        int screenHeight = GraphicsHandler.getScreenHeight();
        float cameraX = player.x - screenWidth / tileSize / 2;
        float cameraY = player.y - screenHeight / tileSize / 2;
        float worldMouseX = (cameraX * tileSize + screenMousePos.x) / tileSize;
        float worldMouseY = (cameraY * tileSize + screenMousePos.y) / tileSize - .5f;

        world.chunkUpdate();
        world.draw(g, 0, 0, screenWidth, screenHeight, cameraX, cameraY, tileSize);

        boolean inventoryFocus = player.inventory.updateInventory(screenWidth, screenHeight,
                screenMousePos, leftClick, rightClick);
        if (inventoryFocus) {
            leftClick = false;
            rightClick = false;
        }

        if (leftClick && player.handBreakPos.x != -1) {
            if (player.handBreakPos.equals(breakingPos)) {
                breakingTicks++;
            } else {
                breakingTicks = 0;
            }
            breakingPos = player.handBreakPos;

            InventoryItem inventoryItem = player.inventory.selectedItem();
            Item item = inventoryItem.getItem();
            int ticksNeeded = world.breakTicks(breakingPos.x, breakingPos.y, item);

            Int2 pos = StockMethods.computeDrawLocationInPlace(cameraX, cameraY, tileSize,
                    tileSize, tileSize, breakingPos.x, breakingPos.y);
            int sprite_index = (int) (Math.min(1, (double) breakingTicks / ticksNeeded) * (breakingSprites.length - 1));
            breakingSprites[sprite_index].draw(g, pos.x, pos.y, tileSize, tileSize);

            if (breakingTicks >= ticksNeeded) {
                if (item != null && item.getClass() == Tool.class) {
                    Tool tool = (Tool) item;
                    tool.uses++;
                    if (tool.uses >= tool.totalUses) {
                        inventoryItem.setEmpty();
                    }
                }

                breakingTicks = 0;
                TileID name = world.removeTile(player.handBreakPos.x, player.handBreakPos.y);
                if (name == TileID.GRASS) {
                    name = TileID.DIRT;
                }
                if (name == TileID.STONE) {
                    name = TileID.COBBLE;
                }
                if (name == TileID.LEAVES && random.nextDouble() < .1) {
                    name = TileID.SAPLING;
                }
                Item newItem = Constants.itemTypes.get((char) name.breaksInto);
                if (newItem != null) {
                    newItem = newItem.clone();
                    newItem.x = player.handBreakPos.x + random.nextFloat()
                            * (1 - (float) newItem.widthPX / tileSize);
                    newItem.y = player.handBreakPos.y + random.nextFloat()
                            * (1 - (float) newItem.widthPX / tileSize);
                    newItem.dy = -.07f;
                    entities.add(newItem);
                }
            }
        } else {
            breakingTicks = 0;
        }

        if (rightClick) {
            if (world.isCraft(player.handBreakPos.x, player.handBreakPos.y)) {


                player.inventory.tableSizeAvailable = 3;
                player.inventory.setVisible(true);
            } else {

                rightClick = false;
                InventoryItem current = player.inventory.selectedItem();
                if (!current.isEmpty()) {
                    TileID itemID = Constants.tileIDs.get(current.getItem().item_id);
                    if (itemID != null) {
                        boolean isPassable = Constants.tileTypes.get(itemID).type.passable;

                        if (isPassable || !player.inBoundingBox(player.handBuildPos, tileSize)) {
                            if (world.addTile(player.handBuildPos, itemID)) {

                                player.inventory.decreaseSelected(1);
                            }
                        }
                    }
                }
            }
        }

        player.updateHand(g, cameraX, cameraY, worldMouseX, worldMouseY, world, tileSize);

        Iterator<Entity> it = entities.iterator();
        while (it.hasNext()) {
            Entity entity = it.next();
            if (entity != player && player.collidesWith(entity, tileSize)) {
                if (entity instanceof Item || entity instanceof Tool) {
                    player.giveItem((Item) entity, 1);
                }
                it.remove();
                continue;
            }
            entity.updatePosition(world, tileSize);
            entity.draw(g, cameraX, cameraY, screenWidth, screenHeight, tileSize);
        }


        if (player.handBreakPos.x != -1) {
            Int2 pos = StockMethods.computeDrawLocationInPlace(cameraX, cameraY, tileSize,
                    tileSize, tileSize, player.handBuildPos.x, player.handBuildPos.y);
            builderIcon.draw(g, pos.x, pos.y, tileSize, tileSize);

            pos = StockMethods.computeDrawLocationInPlace(cameraX, cameraY, tileSize, tileSize,
                    tileSize, player.handBreakPos.x, player.handBreakPos.y);
            minerIcon.draw(g, pos.x, pos.y, tileSize, tileSize);
        }


        player.inventory.draw(g, screenWidth, screenHeight);


        Int2 mouseTest = StockMethods.computeDrawLocationInPlace(cameraX, cameraY, tileSize,
                tileSize, tileSize, worldMouseX, worldMouseY);
        drawMouse(g, mouseTest);


        int heartX = (screenWidth - 250) / 2;
        int heartY = screenHeight - 50;
        for (int heartIdx = 1; heartIdx <= 10; ++heartIdx) {
            int hpDiff = player.hitPoints - heartIdx * 10;
            if (hpDiff >= 0) {
                fullHeart.draw(g, heartX, heartY, 10, 10);
            } else if (hpDiff >= -5) {
                halfHeart.draw(g, heartX, heartY, 10, 10);
            } else {
                emptyHeart.draw(g, heartX, heartY, 10, 10);
            }
            heartX += 15;
        }

        if (player.isHeadUnderWater(world, tileSize)) {

            int bubbleX = (screenWidth + 50) / 2;
            int numBubbles = player.airRemaining();
            for (int bubbleIdx = 1; bubbleIdx <= 10; ++bubbleIdx) {
                if (bubbleIdx <= numBubbles) {
                    bubble.draw(g, bubbleX, heartY, 10, 10);
                } else {
                    emptyBubble.draw(g, bubbleX, heartY, 10, 10);
                }
                bubbleX += 15;
            }
        }

        g.finishDrawing();
        return player.hitPoints;
    }

    public static void drawMouse(GraphicsHandler g, Int2 pos) {
        g.setColor(Color.gray);
        int w1 = 2 * 8;
        g.fillOval(pos.x - w1 / 2, pos.y - w1 / 2, w1, w1);


    }

    public static void drawTileBackground(GraphicsHandler g, Sprite sprite, int tileSize) {
        int screenHeight = GraphicsHandler.getScreenHeight();
        int screenWidth = GraphicsHandler.getScreenWidth();
        for (int i = 0; i <= screenWidth / tileSize; i++) {
            for (int j = 0; j <= screenHeight / tileSize; j++) {
                sprite.draw(g, i * tileSize, j * tileSize, tileSize, tileSize);
            }
        }
    }

    public void zoom(int level) {
        switch (level) {
            case 0:
                if (tileSize < 32) {
                    zoom(1);
                    zoom(0);
                }
                if (tileSize > 32) {
                    zoom(-1);
                    zoom(0);
                }
                break;
            case 1:
                if (tileSize < 128) {
                    tileSize *= 2;
                    for (Entity entity : entities) {
                        entity.widthPX *= 2;
                        entity.heightPX *= 2;
                    }
                    for (Item item : Constants.itemTypes.values()) {
                        item.widthPX *= 2;
                        item.heightPX *= 2;
                    }
                }
                break;
            case -1:
                if (tileSize > 8) {
                    tileSize /= 2;
                    for (Entity entity : entities) {
                        entity.widthPX /= 2;
                        entity.heightPX /= 2;
                    }
                    for (Item item : Constants.itemTypes.values()) {
                        item.widthPX /= 2;
                        item.heightPX /= 2;
                    }
                }
                break;
        }
    }

    public void tossItem() {

        InventoryItem inventoryItem = player.inventory.selectedItem();
        if (!inventoryItem.isEmpty()) {
            Item newItem = inventoryItem.getItem();
            if (!(newItem instanceof Tool)) {
                newItem = newItem.clone();
            }
            inventoryItem.remove(1);
            if (player.facingRight) {
                newItem.x = player.x + 1 + random.nextFloat();
            } else {
                newItem.x = player.x - 1 - random.nextFloat();
            }
            newItem.y = player.y;
            newItem.dy = -.1f;
            entities.add(newItem);
        }
    }

    public void goToMainMenu() {
        zoom(0);
        SaveLoad.doSave(this);

        inMenu = true;
    }

    public static void quit() {

        System.exit(0);
    }

    /**
     * The entry point into the game. We'll simply create an
     * instance of class which will start the display and game
     * loop.
     *
     * @param argv The arguments that are passed into our game
     */
    public static void main(String[] argv) {

        Constants.DEBUG = true;
        for (String arg : argv) {
            if ("-d".equals(arg) || "--debug".equals(arg)) {
                Constants.DEBUG = true;
            } else {
                System.err.println("Unrecognized argument: " + arg);
            }
        }

        SideScrollMinicraft g = new SideScrollMinicraft();


        g.start(true);
    }
}
