package jcog.table;

import jcog.Log;
import jcog.util.SingletonIterator;
import org.slf4j.Logger;
import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;

public class ObjectTable extends DataTable {
    private final Class clazz;

    public ObjectTable(Object x) {
        this(x.getClass(), new SingletonIterator(x));
    }

    public <T> ObjectTable(Class<T> clazz, Iterable<T> objects) {
        super();
        this.clazz = clazz;
        Field[] fields = clazz.getDeclaredFields();
        List<Column<?>> columns = new ArrayList<>();

        // Create columns for each field
        for (Field field : fields) {
            if (!include(field)) continue;

            field.setAccessible(true);
            String columnName = field.getName();
            Class type = field.getType();

            Column<?> column;
            if (type == int.class || type == Integer.class) {
                column = IntColumn.create(columnName);
            } else if (type == double.class || type == Double.class) {
                column = DoubleColumn.create(columnName);
            } else if (type == float.class || type == Float.class) {
                column = FloatColumn.create(columnName);
            } else if (type == boolean.class || type == Boolean.class) {
                column = BooleanColumn.create(columnName);
            } else if (type == long.class || type == Long.class) {
                column = LongColumn.create(columnName);
            } else {
                column = StringColumn.create(columnName);
            }
            columns.add(column);
        }

        // Add columns to table
        columns.forEach(this::addColumns);

        // Populate table rows
        for (T object : objects) {
            //Row row = appendRow();
            for (Field field : fields) {
                if (!include(field)) continue;
                Object value = null;
                try {
                    value = field.get(object);
                } catch (IllegalAccessException e) {
                    logger.error("{} {}", object, field, e);
                }
                int columnIndex = columnIndex(field.getName());
                Column column = column(columnIndex);
                if (value != null) {
                    if (column.type()==ColumnType.STRING && !(value instanceof String))
                        value = value.toString();
                    column.append(value);
                } else {
                    column.appendMissing();
                }
            }
        }

    }

    private static boolean include(Field field) {
        return (field.getModifiers() & Modifier.PRIVATE) == 0;
    }

    private static final Logger logger = Log.log(ObjectTable.class);

    public static String toString(Object result) {
        return new ObjectTable(result).printAll();
    }
}
