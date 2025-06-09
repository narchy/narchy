/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jcog.fuzzydt.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Mohammed H. Jabreel
 */
public class Attribute {

	protected final String name;
	//protected LinguisticVariable linguisticVariable;

	/** TODO ObjectIntHashMap<String> */
	protected final List<String> terms;

	protected Dataset dataset;

	@Deprecated protected int index;

	public Attribute(String name, String[] terms) {
		this.name = name;
		this.terms = new ArrayList<>(Arrays.asList(terms));
	}

	public Dataset getDataset() {
		return dataset;
	}
//
//    public int getIndex() {
//        return index;
//    }

	public String getName() {
		return name;
	}

	public List<String> terms() {
		return terms;
	}

	public double[] fuzzyValues(String termName) {
		int n = this.dataset.getRowsCount();
		double[] vals = new double[n];
		for (int i = 0; i < n; i++)
			vals[i] = this.dataset.getFuzzyValue(i, this.index, termName);
		return vals;
	}

	public double getFuzzyValue(int rowIdx, String termName) {
		return this.dataset.getFuzzyValue(rowIdx, this.index, termName);
	}

	public int termIndex(String termName) {
		return this.terms.indexOf(termName);
	}

	//    public String getBestLinguisticTerm(int rowIdx) {
//        double [] vals = this.getFuzzyValues(rowIdx);
//        double max = vals[0];
//
//        int maxIdx = 0;
//        for(int i = 1; i < vals.length; i++) {
//            if(vals[i] > max) {
//                maxIdx = i;
//                max = vals[i];
//            }
//        }
//        return this.linguisticTerms.get(maxIdx);
//    }
//
//    public String getBestLinguisticTerm(double [] vals) {
//        if(vals.length != linguisticTerms.size()) {
//            throw new IndexOutOfBoundsException("");
//        }
//        double max = vals[0];
//
//        int maxIdx = 0;
//        for(int i = 1; i < vals.length; i++) {
//            if(vals[i] > max) {
//                maxIdx = i;
//                max = vals[i];
//            }
//        }
//        return this.linguisticTerms.get(maxIdx);
//    }
	public int termCount() {
		return terms.size();
	}

}
