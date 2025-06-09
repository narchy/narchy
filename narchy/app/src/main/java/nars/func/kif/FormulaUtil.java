package nars.func.kif;

import jcog.util.ArrayUtil;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiPredicate;

/**
 * Created by sserban on 2/17/15.
 */
public enum FormulaUtil {
    ;

    /** ***************************************************************
     * Must check that this is a simple clause before calling!
     */
    public static String toProlog(Formula f) {

        StringBuilder sb = new StringBuilder();
        sb.append(f.car()).append('(');
        for (int i = 1; i < f.argumentsToArrayList(0).size(); i ++) {
            if (i != 1)
                sb.append(',');
            sb.append(f.getArgument(i));
        }
        sb.append(')');
        return sb.toString();
    }

    /*******************************************************************************************
     * Generates all permutations of the given size which are valid according to the given callback function.
     * @param size
     * @param validateFn
     * @return
     */
    public static List<int[]> getPermutations(int size, BiPredicate<Integer, Integer> validateFn) {

        int[] array = new int[size];
        for( int i = 0; i< size; i++) {
            array[i] = i;
        }
        List<int[]> result = new LinkedList<>();
        permutation(ArrayUtil.EMPTY_INT_ARRAY, array, result, validateFn);
        return result;
    }

    /** ***************************************************************************************
     * @param prefix
     * @param array
     * @param permutations
     * @param validateFn
     */
    private static void permutation(int[] prefix, int[] array, List<int[]> permutations,
                                    BiPredicate<Integer, Integer> validateFn) {

        int n = array.length;
        if (n == 0) {
            permutations.add(prefix);
        }
        else {
            for (int i = 0; i < n; i++) {
                if (validateFn.test(prefix.length, array[i])) {
                    int[] newPrefix = Arrays.copyOf(prefix, prefix.length + 1);
                    newPrefix[prefix.length] = array[i];
                    int[] leftovers = new int[n - 1];
                    System.arraycopy(array, 0, leftovers, 0, i);
                    if (n - i + 1 >= 0) System.arraycopy(array, i + 1, leftovers, i + 1 - 1, n - i + 1);
                    permutation(newPrefix, leftovers, permutations, validateFn);
                }
            }
        }
    }

    /** ********************************************************************************************
     * Factory method for the memo map
     * @param s1
     * @param s2
     * @return
     */
    public static FormulaMatchMemoMapKey createFormulaMatchMemoMapKey(String s1, String s2) {

        return new FormulaMatchMemoMapKey(s1,s2);
    }

    public static class FormulaMatchMemoMapKey {

        String f1;
        String f2;

        public FormulaMatchMemoMapKey(String f1, String f2) {

            this.f1 = f1;
            this.f2 = f2;
        }

        /*******************************************************************************
         *
         * @param o
         * @return
         */
        @Override
        public boolean equals(Object o) {

            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            FormulaMatchMemoMapKey that = (FormulaMatchMemoMapKey) o;

            if (!Objects.equals(f1, that.f1)) return false;
            return Objects.equals(f2, that.f2);
        }

        /** ******************************************************************************************
         *
         * @return
         */
        @Override
        public int hashCode() {

            int result = f1 != null ? f1.hashCode() : 0;
            result = 31 * result + (f2 != null ? f2.hashCode() : 0);
            return result;
        }

        /***********************************************************************************************
         *
         * @return
         */
        @Override
        public String toString() {

            return '{' +
                    "f1='" + f1 + '\'' +
                    ", f2='" + f2 + '\'' +
                    '}';
        }
    }
}
