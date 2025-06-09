package java4k.pinball4k.editor;

import javax.swing.*;
import javax.swing.border.BevelBorder;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Pinball level editor.
 * 
 * @author tombr
 *
 */
public class Editor extends JFrame {
	
	ObjectPropertiesUI propsPnl = new ObjectPropertiesUI();
    JPanel statusBar = new JPanel();
	
	/**
	 * Creates an editor.
	 *
	 */
	public Editor() {
		super("Pinball Editor");
		setSize(1400, 800);
		setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		
		LevelPanel levelPnl = new LevelPanel(this);
		
		JScrollPane levelScroll = new JScrollPane(levelPnl);
		levelScroll.setMinimumSize(new Dimension(200, 100));

		GroupUI groupsPnl = new GroupUI(levelPnl.level.groups, levelPnl);
		levelPnl.groupUI = groupsPnl;
		
		JSplitPane propsSplit = new JSplitPane(JSplitPane.VERTICAL_SPLIT, propsPnl, groupsPnl);
		propsSplit.setDividerLocation(200);
		
		JSplitPane levelSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, levelScroll, propsSplit);
		levelSplit.setDividerLocation(1100);

		
		statusBar.setBorder(new BevelBorder(BevelBorder.LOWERED));
		statusBar.add(new JLabel("Hello world"));
		
		getContentPane().setLayout(new BorderLayout());
		getContentPane().add(levelSplit, BorderLayout.CENTER);
		getContentPane().add(levelPnl.getToolBar(), BorderLayout.NORTH);		
		getContentPane().add(statusBar, BorderLayout.SOUTH);		
		
		
		setVisible(true);

		levelPnl.requestFocusInWindow();
	}
	
	public void select(List<Handle> selection) {
		/*
		ArrayList<ObjectProperties> props = new ArrayList<ObjectProperties>();
		for (Handle handle : selection) {
			if (handle.getLevelObject() != null) {
				LevelObject levelObj = handle.getLevelObject();
				
				
				if (props.contains(levelObj) == false) {
					props.addAt(levelObj.properties);
				}
			}
		}
		propsPnl.load(props);
		*/

		
		ArrayList<LevelObject> uniqueObjs = new ArrayList<>();
		String selectionStr = "";
		for (Handle handle : selection) {
			LevelObject obj = handle.getLevelObject();
			if (obj != null && !uniqueObjs.contains(obj)) {
				uniqueObjs.add(obj);
				selectionStr += "" + obj + " ";
			}
		}
		selectionStr = selectionStr.trim().isEmpty() ? "Nothing selected" : selectionStr;
		JLabel lbl = (JLabel) statusBar.getComponent(0);
		lbl.setText(selectionStr);
		

		propsPnl.setSelection(uniqueObjs);
	}

	/**
	 * Main entry point of editor application.
	 * @param args not used
	 */
	public static void main(String[] args) {
		new Editor();
	}
}