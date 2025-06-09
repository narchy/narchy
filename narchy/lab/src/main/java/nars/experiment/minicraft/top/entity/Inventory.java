package nars.experiment.minicraft.top.entity;

import nars.experiment.minicraft.top.item.Item;
import nars.experiment.minicraft.top.item.ResourceItem;
import nars.experiment.minicraft.top.item.resource.Resource;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Inventory {
    public List<Item> items = new ArrayList<>();

    public void add(Item item) {
        add(items.size(), item);
    }

    public synchronized void add(int slot, Item item) {
        if (item instanceof ResourceItem toTake) {
            ResourceItem has = findResource(toTake.resource);
            if (has == null) {
                items.add(slot, toTake);
            } else {
                has.count += toTake.count;
            }
        } else {
            items.add(slot, item);
        }
    }

    private ResourceItem findResource(Resource resource) {
        int bound = items.size();
        return IntStream.range(0, bound).filter(i -> items.get(i) instanceof ResourceItem).mapToObj(i -> (ResourceItem) items.get(i)).filter(has -> has.resource == resource).findFirst().orElse(null);
    }

    public boolean hasResources(Resource r, int count) {
        ResourceItem ri = findResource(r);
        if (ri == null) return false;
        return ri.count >= count;
    }

    public boolean removeResource(Resource r, int count) {
        ResourceItem ri = findResource(r);
        if (ri == null) return false;
        if (ri.count < count) return false;
        ri.count -= count;
        if (ri.count <= 0) items.remove(ri);
        return true;
    }

    public int count(Item item) {
        if (item instanceof ResourceItem rir) {
            ResourceItem ri = findResource(rir.resource);
            if (ri != null) return ri.count;
            return 0;
        } else {
            return (int) IntStream.range(0, items.size()).filter(i -> items.get(i).matches(item)).count();
        }
    }
}