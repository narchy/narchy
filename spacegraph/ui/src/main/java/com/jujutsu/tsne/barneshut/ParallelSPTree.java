package com.jujutsu.tsne.barneshut;

import static java.lang.Math.sqrt;

public class ParallelSPTree extends SPTree {


	public ParallelSPTree(int D, double[] inp_data, int N) {
		super(D, inp_data, N);
	}

	public ParallelSPTree(int D, double[] inp_data, int N, double[] inp_corner, double[] inp_width) {
		super(D, inp_data, N, inp_corner, inp_width);
	}

	public ParallelSPTree(int D, double[] inp_data, double[] inp_corner, double[] inp_width) {
		super(D, inp_data, inp_corner, inp_width);
	}

	private ParallelSPTree(SPTree inp_parent, int D, double[] inp_data, double[] inp_corner, double[] inp_width) {
		super(inp_parent, D, inp_data, inp_corner, inp_width);
	}

	public ParallelSPTree(SPTree inp_parent, int D, double[] inp_data, int N, double[] inp_corner, double[] inp_width) {
		super(inp_parent, D, inp_data, N, inp_corner, inp_width);
	}

	@Override
	SPTree[] getTreeArray(int no_children) {
		return new ParallelSPTree[no_children];
	}

	@Override
	SPTree getNewTree(SPTree root, double[] new_corner, double[] new_width) {
		return new ParallelSPTree(root, dimension, data, new_corner, new_width);
	}


	@Override
	public double computeNonEdgeForces(int point_index, double theta, double[] neg_f, Object accumulator) {
		Double sum_Q = (Double) accumulator;
		double input_sum_Q = sum_Q;
		double[] buff = new double[dimension];


		if (cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return 0.0;


		double D = 0.0;
		int ind = point_index * dimension;
		double max_width = 0.0;

		for (int d = 0; d < dimension; d++) {
			buff[d] = data[ind + d] - center_of_mass[d];
			D += buff[d] * buff[d];

			double cur_width = boundary.getWidth(d);
			max_width = Math.max(max_width, cur_width);
		}

		if (is_leaf || max_width / sqrt(D) < theta) {

			D = 1.0 / (1.0 + D);
			double mult = cum_size * D;
			sum_Q += mult;
			mult *= D;
			for (int d = 0; d < dimension; d++) neg_f[d] += mult * buff[d];
		} else {

			for (int i = 0; i < no_children; i++)
				sum_Q += children[i].computeNonEdgeForces(point_index, theta, neg_f, input_sum_Q);
		}

		return sum_Q;
	}


}