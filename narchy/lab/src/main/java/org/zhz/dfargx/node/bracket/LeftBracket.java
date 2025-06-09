package org.zhz.dfargx.node.bracket;

import org.zhz.dfargx.automata.NFA;
import org.zhz.dfargx.node.BranchNode;
import org.zhz.dfargx.node.Node;
import org.zhz.dfargx.stack.OperatingStack;
import org.zhz.dfargx.stack.ShuntingStack;

/**
 * Created on 2015/5/12.
 */
public final class LeftBracket extends BranchNode {

    public static final LeftBracket the = new LeftBracket();

    private LeftBracket() {
        super();
    }

    @Override
    public void accept(NFA nfa) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void operate(Node left, Node right) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Node copy() {
        return this; 
    }

    @Override
    public void accept(OperatingStack operatingStack) {
        operatingStack.visit(this);
    }

    @Override
    public void accept(ShuntingStack shuntingStack) {
        shuntingStack.visit(this);
    }

    @Override
    public String toString() {
        return "[(]";
    }

    @Override
    public int getPri() {
        return -1;
    }
}
