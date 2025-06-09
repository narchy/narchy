package java4k.pinball4k.editor;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.lang.reflect.Field;
import java.util.ArrayList;

/**
 * A panel where the user can edit the level object properties.
 * @author tombr
 */
public class ObjectPropertiesUI extends JPanel {
	
	public ArrayList<LevelObject> selection = new ArrayList<>();
	private final MyTableModel tableModel = new MyTableModel();
	
	public ObjectPropertiesUI() {
		JTable table = new JTable(tableModel);
		JScrollPane scrollpane = new JScrollPane(table);
		
		setLayout(new BorderLayout());
		add(new JLabel("Object properties:"), BorderLayout.NORTH);
		add(table, BorderLayout.CENTER);
		
	}
	
	public void setSelection(ArrayList<LevelObject> selection) {
		this.selection = selection;
		tableModel.fireTableDataChanged();
	}
	
	private Field[] getFields() {
		if (selection.size() == 0) {
			return new Field[0];
		}
		
		return selection.get(0).getProperties();
	}
	
	class MyTableModel extends AbstractTableModel {
		@Override
        public int getRowCount() {
			return getFields().length;
		}
			
		@Override
        public int getColumnCount() {
			return 2;
		}

		@Override
        public Object getValueAt(int row, int column) {
			Field field = getFields()[row];
			if (column == 0) {
				return field.getName() + " (" + field.getType().getSimpleName() + ")";
			}
			try {
				Object value = field.get(selection.get(0));
				if (value instanceof Point point) {
                    return "" + point.x + " " + point.y;
				}
				return value;
			} catch (Exception e) {
			}
			return "error";
		}		  		
		
		@Override
        public boolean isCellEditable(int rowIndex, int columnIndex) {
			return (columnIndex == 1);
		}
		
		@Override
        public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
			for (LevelObject obj : selection) {
				try {
					Field field = getFields()[rowIndex];
					if (field.getType() == float.class) {
						field.setFloat(obj, Float.parseFloat("" + aValue));
					} else if (field.getType() == int.class) {
						field.setInt(obj, Integer.parseInt("" + aValue));
					} else if (field.getType() == boolean.class) {
						field.setBoolean(obj, Boolean.parseBoolean("" + aValue));
					}
					System.out.println("setValueAt " + rowIndex + " " + aValue);
					LevelPanel.instance.repaint();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	/*
	public ArrayList<ObjectProperties> selectionProps;
	
	private HashMap fieldTxtMap = new HashMap();
	
	public ObjectPropertiesUI(Class propsClass) {
		Field[] fields = propsClass.getFields();
		
		JPanel fieldPnl = new JPanel();
		fieldPnl.setLayout(new GridLayout(fields.length, 2));
		
		for (Field field : fields) {
			fieldPnl.addAt(new JLabel(field.getName()));
			
			JComponent fieldCmp = new JTextField();
			if (new String("" + field.getType()).equals("boolean")) {
				String[] boolString = {"true", "false"};
				fieldCmp = new JComboBox(boolString);
			} else if (field.getName().equals("behaviorId")) {
				System.out.println(field.getName());
				String[] behaviourString = {"normal", "game over", "start", "drop down", "roll over", "bumper", "gate"};
				fieldCmp = new JComboBox(behaviourString);
			}
			
			fieldTxtMap.put(field.getName(), fieldCmp);
			fieldPnl.addAt(fieldCmp);
		}
		
		JButton storeBtn = new JButton("Store");
		storeBtn.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				store(selectionProps);
			}
		});
		
		setLayout(new BorderLayout());
		addAt(new JLabel("Object properties:"), BorderLayout.NORTH);
		addAt(fieldPnl, BorderLayout.CENTER);
		addAt(storeBtn, BorderLayout.SOUTH);
	}*/
	
	/**
	 * Loads the specified fields in the specified property object into the ui.
	 * @param selectionProps the object to load
	 */
	/*public void load(ArrayList<ObjectProperties> newProps) {
		store(selectionProps);
		if (newProps != null && newProps.size() > 0) {
			ObjectProperties objProps = newProps.get(0);
			Field[] fields = objProps.getClass().getFields();
			for (Field field : fields) {
				try {
					if (new String("" + field.getType()).equals("boolean")) {
						JComboBox boolList = (JComboBox) fieldTxtMap.get(field.getName());
						boolList.setSelectedIndex(field.getBoolean(objProps) ? 0 : 1);
					} else if (field.getName().equals("behaviorId")) {
						JComboBox boolList = (JComboBox) fieldTxtMap.get(field.getName());
						boolList.setSelectedIndex(field.getInt(objProps));
					} else {
						JTextField fieldTxt = (JTextField) fieldTxtMap.get(field.getName());
						fieldTxt.setText("" + field.get(objProps));
					}
				} catch (Exception e) {
					System.out.println("Failed to load " + field.getName() + " of type " + field.getType());
					e.printStackTrace();
				}
			}
		}
		
		selectionProps = newProps;
	}*/
	
	/**
	 * Stores the ui values in the properties fields
	 * @param selectionProps where to store the data
	 */
	/*private void store(ArrayList<ObjectProperties> storeProps) {
		if (storeProps != null && storeProps.size() > 0) {
			for (ObjectProperties props : storeProps) {
				Field[] fields = props.getClass().getFields();
				for (Field field : fields) {
					try {
						if (field.getName().equals("behaviorId")) {
							JComboBox behaviorList = (JComboBox) fieldTxtMap.get(field.getName());
							field.setAt(props, behaviorList.getSelectedIndex());
						} else if (field.getType().toString().equals("boolean")) {
							JComboBox boolList = (JComboBox) fieldTxtMap.get(field.getName());
							field.setAt(props, Boolean.valueOf(boolList.getSelectedItem().toString()));
						} else if (field.getType().toString().equals("int")) {
							JTextField fieldTxt = (JTextField) fieldTxtMap.get(field.getName());
							field.setAt(props, Integer.valueOf(fieldTxt.getText()));
						} else if (field.getType().toString().equals("float")) {
							JTextField fieldTxt = (JTextField) fieldTxtMap.get(field.getName());
							field.setAt(props, Float.valueOf(fieldTxt.getText()));
						}
					} catch (Exception e) {
						System.out.println(getClass() + " error storing field " + field);
						e.printStackTrace();
					}
				}
			}
		}
	}*/
}