/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jcog.fuzzydt.utils;

import jcog.Util;
import jcog.fuzzydt.data.Dataset;

import java.util.List;

/**
 * @author MHJ
 */
public class GeneralizedFuzzyPartitionEntropyMeasure extends PreferenceMeasure.MinimumPreferenceMeasure {

	protected final MappingFunction mappingFunction;

	public GeneralizedFuzzyPartitionEntropyMeasure() {
		this.mappingFunction = new MappingFunction();
	}

	public GeneralizedFuzzyPartitionEntropyMeasure(MappingFunction mappingFunction) {
		this.mappingFunction = mappingFunction;
	}


	@Override
	public double getPreferenceValue(Dataset dataset, String attribute, String[] evidances) {

		String className = dataset.getClassName();
		List<String> terms = dataset.attr(attribute).terms();
		List<String> classTerms = dataset.attr(className).terms();

		double[] a = null;
		if (evidances != null) {
			for (int i = 0; i < evidances.length - 1; i += 2) {
				if (a == null) {
					a = dataset.attr(evidances[i]).fuzzyValues(evidances[i + 1]);
				} else {
					double[] aPrime = dataset.attr(evidances[i]).fuzzyValues(evidances[i + 1]);
					for (int j = 0; j < aPrime.length; j++) {
						a[j] = Math.min(a[j], aPrime[j]);
					}
				}
			}
		}


		double[] mij = new double[terms.size()];
		double[] entropies = new double[terms.size()];

		int i = 0;

		for (String term : terms) {
			double[] vals = dataset.attr(attribute).fuzzyValues(term);

			if (a != null)
				vals = Utils.min(a, vals);

			//vals = mappingFunction.map(vals);
            double[] a2 = mappingFunction.map(vals);
            mij[i] = Util.sum(a2);

			double[] mijk = new double[classTerms.size()];
			int j = 0;
//			double[] h = new double[classTerms.size()];
			for (String ck : classTerms) {
				double[] vals2 = dataset.attr(className).fuzzyValues(ck);
				//vals2 = mappingFunction.map(vals2);
				vals2 = Utils.min(vals, vals2);

                double[] a1 = mappingFunction.map(vals2);
                mijk[j++] = Util.sum(a1);
			}
            double mijPrime = Util.sum(mijk);
            double x1 = mappingFunction.map(mij[i]);
            double hgda = Math.log(x1);
			//Utils.normalizeWith(mijk, Utils.sum(mijk));
			double entropy = 0;
			for (double v : mijk) {
                double x = mappingFunction.map(v);
                entropy += ((v / mijPrime) * (Math.log(x) - hgda));
            }
			entropies[i++] = -entropy;
		}
        Utils.normalizeWith(mij, Util.sum(mij));
		double s = 0;
		for (i = 0; i < mij.length; i++)
			s += entropies[i] * mij[i];
		return s;
	}

}
