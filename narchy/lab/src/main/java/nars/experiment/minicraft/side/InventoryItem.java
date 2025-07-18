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

import java.io.Serializable;

public class InventoryItem implements Serializable {
    private static final long serialVersionUID = -2389571032163510795L;

    public static final int maxCount = 64;
    public int count;
    public Item item;

    public InventoryItem(Item item) {
        this.setItem(item);
    }


    public int add(Item item, int count) {
        if (this.isEmpty()) {
            this.setItem(item);
        }
        if (this.getItem().item_id != item.item_id) {
            return count;
        }
        int maxCount = InventoryItem.maxCount;
        if (this.getItem().getClass() == Tool.class) {
            maxCount = 1;
        }
        if (this.getCount() + count <= maxCount) {
            this.setCount(this.getCount() + count);
            return 0;
        } else {
            int leftOver = count - (maxCount - this.getCount());
            this.setCount(maxCount);
            return leftOver;
        }
    }


    public int remove(int count) {
        if (0 <= this.getCount() - count) {
            this.setCount(this.getCount() - count);
            return 0;
        } else if (this.getCount() == count) {
            this.setEmpty();
            return 0;
        } else {
            int leftOver = count - this.getCount();
            this.setEmpty();
            return leftOver;
        }
    }

    public void setEmpty() {
        this.setCount(0);
        this.setItem(null);
    }

    public boolean isEmpty() {
        return this.getCount() == 0 || this.getItem() == null;
    }

    public boolean isFull() {
        return getCount() >= maxCount;
    }

    public void stack(InventoryItem other) {
        if (other.getItem().getClass() != Tool.class) {
            int result = this.add(other.getItem(), other.getCount());
            other.remove(other.getCount() - result);
        }
    }

    public void draw(GraphicsHandler g, int x, int y, int tileSize) {
        if (this.getCount() <= 0) {
            return;
        }
        this.getItem().sprite.draw(g, x, y, tileSize, tileSize);
        if (this.getCount() > 1) {
            g.setColor(Color.white);
            g.drawString(String.valueOf(this.getCount()), x, y + tileSize / 2);
        }
        if (item.getClass() == Tool.class) {
            Tool tool = (Tool) item;
            if (tool.uses != 0) {
                int width = (int) (((float) (tool.totalUses - tool.uses) / tool.totalUses) * (tileSize));
                g.setColor(Color.green);
                int height = 2;
                int top = y + tileSize - 4;
                int left = x + 2;
                g.fillRect(left, top, width, height);
            }
        }
    }

    public void setItem(Item item) {
        this.item = item;
    }

    public Item getItem() {
        return item;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public int getCount() {
        return count;
    }
}
