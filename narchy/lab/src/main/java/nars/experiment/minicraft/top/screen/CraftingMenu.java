package nars.experiment.minicraft.top.screen;

import nars.experiment.minicraft.top.crafting.Recipe;
import nars.experiment.minicraft.top.entity.Player;
import nars.experiment.minicraft.top.gfx.Color;
import nars.experiment.minicraft.top.gfx.Font;
import nars.experiment.minicraft.top.gfx.Screen;
import nars.experiment.minicraft.top.item.Item;
import nars.experiment.minicraft.top.item.ResourceItem;
import nars.experiment.minicraft.top.sound.Sound;

import java.util.ArrayList;
import java.util.List;

public class CraftingMenu extends Menu {
    private final Player player;
    private int selected;

    private final List<Recipe> recipes;

    public CraftingMenu(List<Recipe> recipes, Player player) {
        this.recipes = new ArrayList<>(recipes);
        this.player = player;

        for (int i = 0; i < recipes.size(); i++) {
            this.recipes.get(i).checkCanCraft(player);
        }

        this.recipes.sort((r1, r2) -> {
            if (r1.canCraft && !r2.canCraft) return -1;
            if (!r1.canCraft && r2.canCraft) return 1;
            return 0;
        });
    }

    @Override
    public void tick() {
        if (input.menu.clicked) game.setMenu(null);

        if (input.up.clicked) selected--;
        if (input.down.clicked) selected++;

        int len = recipes.size();
        if (len == 0) selected = 0;
        if (selected < 0) selected += len;
        if (selected >= len) selected -= len;

        if (input.attack.clicked && len > 0) {
            Recipe r = recipes.get(selected);
            r.checkCanCraft(player);
            if (r.canCraft) {
                r.deductCost(player);
                r.craft(player);
                Sound.craft.play();
            }
            for (int i = 0; i < recipes.size(); i++) {
                recipes.get(i).checkCanCraft(player);
            }
        }
    }

    @Override
    public void render(Screen screen) {
        Font.renderFrame(screen, "Have", 12, 1, 19, 3);
        Font.renderFrame(screen, "Cost", 12, 4, 19, 11);
        Font.renderFrame(screen, "Crafting", 0, 1, 11, 11);
        renderItemList(screen, 0, 1, 11, 11, recipes, selected);

        if (!recipes.isEmpty()) {
            Recipe recipe = recipes.get(selected);
            int hasResultItems = player.inventory.count(recipe.resultTemplate);
            int xo = 13 * 8;
            screen.render(xo, 2 * 8, recipe.resultTemplate.getSprite(), recipe.resultTemplate.getColor(), 0);
            Font.draw(String.valueOf(hasResultItems), screen, xo + 8, 2 * 8, Color.get(-1, 555, 555, 555));

            List<Item> costs = recipe.costs;
            for (int i = 0; i < costs.size(); i++) {
                Item item = costs.get(i);
                int yo = (5 + i) * 8;
                screen.render(xo, yo, item.getSprite(), item.getColor(), 0);
                int requiredAmt = 1;
                if (item instanceof ResourceItem) {
                    requiredAmt = ((ResourceItem) item).count;
                }
                int has = player.inventory.count(item);
                int color = Color.get(-1, 555, 555, 555);
                if (has < requiredAmt) {
                    color = Color.get(-1, 222, 222, 222);
                }
                if (has > 99) has = 99;
                Font.draw(requiredAmt + "/" + has, screen, xo + 8, yo, color);
            }
        }

    }
}