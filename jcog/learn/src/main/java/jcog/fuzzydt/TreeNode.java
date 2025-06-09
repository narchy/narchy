/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jcog.fuzzydt;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Mohammed H. Jabreel
 */
public class TreeNode {

	private final NodeType nodeType;
	private final String title;
	private final List<TreeNode> children;
	private TreeNode parent;
	private double value;


	public TreeNode(NodeType nodeType, String title) {
		this.nodeType = nodeType;
		this.title = title;
		this.children = new ArrayList<>();
		this.parent = null;
	}

	public List<TreeNode> getChildren() {
		return children;
	}

	public NodeType type() {
		return nodeType;
	}

	public String getTitle() {
		return title;
	}

	public int getChildrenCount() {
		return this.children.size();
	}

	public void addChild(TreeNode child) {
		if (this.nodeType == NodeType.LEAF) {
			throw new IllegalArgumentException("Leaf node can not be parent ");
		}
		this.children.add(child);
		child.parent = this;
	}


	public boolean isRoot() {
		return this.parent == null;
	}

	public boolean isLeaf() {
		return this.nodeType == NodeType.LEAF;
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}


	public enum NodeType {
		ATTRIBUTE,
		VALUE,
		LEAF
	}
}
