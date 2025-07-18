///*
// * Java Genetic Algorithm Library (@__identifier__@).
// * Copyright (c) @__year__@ Franz Wilhelmstötter
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *      http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// *
// * Author:
// *    Franz Wilhelmstötter (franz.wilhelmstoetter@gmail.com)
// */
//package nars.evolve;
//
//import io.jenetics.Mutator;
//import io.jenetics.engine.Engine;
//import io.jenetics.engine.EvolutionResult;
//import io.jenetics.engine.Limits;
//import io.jenetics.ext.SingleNodeCrossover;
//import io.jenetics.ext.util.TreeNode;
//import io.jenetics.prog.ProgramGene;
//import io.jenetics.prog.op.*;
//import io.jenetics.prog.regression.Error;
//import io.jenetics.prog.regression.LossFunction;
//import io.jenetics.prog.regression.Regression;
//import io.jenetics.prog.regression.Sample;
//import io.jenetics.util.ISeq;
//import io.jenetics.util.RandomRegistry;
//
//import static java.lang.Math.abs;
//import static java.lang.String.format;
//
///**
// * Symbolic regression involves finding a mathematical expression, in symbolic
// * form, that provides a good, best, or perfect fit between a given finite
// * sampling of values of the independent variables and the associated values of
// * the dependent variables. --- John R. Koza, Genetic Programming I
// *
// * @author <a href="mailto:franz.wilhelmstoetter@gmail.com">Franz Wilhelmstötter</a>
// * @version 5.0
// * @since 3.9
// */
//public class JeneticSymbolicRegression {
//
//	// Definition of the allowed operations.
//	private static final ISeq<Op<Double>> OPS =
//		ISeq.of(MathOp.ADD, MathOp.SUB, MathOp.MUL);
//
//	// Definition of the terminals.
//	private static final ISeq<Op<Double>> TMS = ISeq.of(
//		Var.of("x", 0),
//		EphemeralConst.of(() -> (double)RandomRegistry.getRandom().nextInt(10))
//	);
//
//	private static final Regression<Double> REGRESSION = Regression.of(
//		Regression.codecOf(OPS, TMS, 5, t -> t.getGene().size() < 30),
//		Error.of(LossFunction::mse),
//		// Lookup table for 4*x^3 - 3*x^2 + x
//		Sample.ofDouble(-1.0, -8.0000),
//		Sample.ofDouble(-0.9, -6.2460),
//		Sample.ofDouble(-0.8, -4.7680),
//		Sample.ofDouble(-0.7, -3.5420),
//		Sample.ofDouble(-0.6, -2.5440),
//		Sample.ofDouble(-0.5, -1.7500),
//		Sample.ofDouble(-0.4, -1.1360),
//		Sample.ofDouble(-0.3, -0.6780),
//		Sample.ofDouble(-0.2, -0.3520),
//		Sample.ofDouble(-0.1, -0.1340),
//		Sample.ofDouble(0.0, 0.0000),
//		Sample.ofDouble(0.1, 0.0740),
//		Sample.ofDouble(0.2, 0.1120),
//		Sample.ofDouble(0.3, 0.1380),
//		Sample.ofDouble(0.4, 0.1760),
//		Sample.ofDouble(0.5, 0.2500),
//		Sample.ofDouble(0.6, 0.3840),
//		Sample.ofDouble(0.7, 0.6020),
//		Sample.ofDouble(0.8, 0.9280),
//		Sample.ofDouble(0.9, 1.3860),
//		Sample.ofDouble(1.0, 2.0000)
//	);
//
//	public static void main(String[] args) {
//		Engine<ProgramGene<Double>, Double> engine = Engine
//			.builder(REGRESSION)
//			.minimizing()
//			.alterers(
//				new SingleNodeCrossover<>(0.1),
//				new Mutator<>())
//			.build();
//
//		EvolutionResult<ProgramGene<Double>, Double> result = engine.stream()
//			//.limit(Limits.byFitnessThreshold(0.01))
//			.limit(Limits.byFixedGeneration(10))
//			.collect(EvolutionResult.toBestEvolutionResult());
//
//		ProgramGene<Double> program = result.getBestPhenotype()
//			.getGenotype()
//			.getGene();
//
//		TreeNode<Op<Double>> tree = program.toTreeNode();
//		MathExpr.rewrite(tree);
//		System.out.println("Generations: " + result.getTotalGenerations());
//		System.out.println("Function:    " + new MathExpr(tree));
//		System.out.println("Error:       " + REGRESSION.error(tree));
//
//		System.out.println("x: y, y', error");
//		for (Sample<Double> sample : REGRESSION.samples()) {
//			double x = sample.argAt(0);
//			double y = program.eval(x);
//
//			System.out.println(format(
//				"%2.2f: %2.4f, %2.4f: %2.5f",
//				x, f(x), y, abs(f(x) - y)
//			));
//		}
//
//	}
//
//	// The function we want to determine.
//	private static double f(double x) {
//		return 4*x*x*x - 3*x*x + x;
//	}
//
//}
