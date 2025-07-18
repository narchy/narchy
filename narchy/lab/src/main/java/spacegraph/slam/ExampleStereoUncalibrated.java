///*
// * Copyright (c) 2020, Peter Abeles. All Rights Reserved.
// *
// * This file is part of BoofCV (http://boofcv.org).
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package spacegraph.slam;
//
//import boofcv.abst.feature.associate.AssociateDescription;
//import boofcv.abst.feature.associate.ScoreAssociation;
//import boofcv.abst.feature.detdesc.DetectDescribePoint;
//import boofcv.abst.feature.detect.interest.ConfigFastHessian;
//import boofcv.abst.geo.bundle.*;
//import boofcv.alg.geo.MultiViewOps;
//import boofcv.alg.geo.PerspectiveOps;
//import boofcv.alg.geo.bundle.cameras.BundlePinholeSimplified;
//import boofcv.alg.geo.selfcalib.EstimatePlaneAtInfinityGivenK;
//import boofcv.concurrency.BoofConcurrency;
//import boofcv.core.image.ConvertImage;
//import boofcv.factory.feature.associate.ConfigAssociateGreedy;
//import boofcv.factory.feature.associate.FactoryAssociation;
//import boofcv.factory.feature.detdesc.FactoryDetectDescribe;
//import boofcv.factory.geo.*;
//import boofcv.gui.image.ImageGridPanel;
//import boofcv.gui.image.ShowImages;
//import boofcv.io.UtilIO;
//import boofcv.io.image.ConvertBufferedImage;
//import boofcv.io.image.UtilImageIO;
//import boofcv.struct.calib.CameraPinholeBrown;
//import boofcv.struct.feature.AssociatedIndex;
//import boofcv.struct.feature.TupleDesc_F64;
//import boofcv.struct.geo.AssociatedPair;
//import boofcv.struct.image.GrayF32;
//import boofcv.struct.image.GrayU8;
//import boofcv.struct.image.ImageType;
//import boofcv.struct.image.Planar;
//import com.github.sarxos.webcam.Webcam;
//import georegression.struct.point.Vector3D_F64;
//import georegression.struct.se.Se3_F64;
//import jcog.Util;
//import org.ddogleg.fitting.modelset.ModelFitter;
//import org.ddogleg.fitting.modelset.ModelMatcher;
//import org.ddogleg.optimization.lm.ConfigLevenbergMarquardt;
//import org.ddogleg.struct.FastAccess;
//import org.ejml.data.DMatrixRMaj;
//import org.ejml.dense.row.CommonOps_DDRM;
//
//import javax.swing.*;
//import java.awt.*;
//import java.awt.image.BufferedImage;
//import java.util.ArrayDeque;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.concurrent.ThreadLocalRandom;
//
//import static spacegraph.slam.ExampleTrifocalStereoUncalibrated.computeStereoCloud;
//
//
///**
// * <p>
// * In this example a stereo point cloud is computed from two uncalibrated stereo images. The uncalibrated problem
// * is much more difficult from the calibrated or semi-calibrated problem and the solution below will often fail.
// * The root cause of this difficulty is that it is impossible to remove all false associations given two views,
// * even if the true fundamental matrix is provided! For that reason it is recommended that you use a minimum of
// * three views with uncalibrated observations.
// * </p>
// *
// * <p>
// * A summary of the algorithm is provided below. There are many ways in which it can be improved upon, but would
// * increase the complexity. There is also no agreed upon best solution found in the literature and the solution
// * presented below is "impractical" because of its sensitivity to tuning parameters. If you got a solution
// * which does a better job let us know!
// * </p>
// *
// * <ol>
// *     <li>Feature association</li>
// *     <li>RANSAC to estimate Fundamental matrix</li>
// *     <li>Guess and check focal length and compute projective to metric homography</li>
// *     <li>Upgrade to metric geometry</li>
// *     <li>Set up bundle adjustment and triangulate 3D points</li>
// *     <li>Run bundle adjustment</li>
// *     <li>Prune outlier points</li>
// *     <li>Run bundle adjustment again</li>
// *     <li>Compute stereo rectification</li>
// *     <li>Rectify images</li>
// *     <li>Compute stereo disparity</li>
// *     <li>Compute and display point cloud</li>
// * </ol>
// *
// * @author Peter Abeles
// */
//public class ExampleStereoUncalibrated {
//
//	public static void main( String[] args ) {
//		// Solution below is unstable. Turning concurrency off so that it always produces a valid solution
//		// The two view case is very challenging and I've not seen a stable algorithm yet
//		BoofConcurrency.USE_CONCURRENT = false;
//
//		PointCloudViewerSwing pcv = new PointCloudViewerSwing();
//		JFrame j = new JFrame();
//		j.getContentPane().add(pcv.getComponent());
//		j.setSize(1200, 900);
//		j.setVisible(true);
//
////		pcv.setTranslationStep(ab.baseline/3);
//		pcv.setDotSize(2);
//		pcv.setFog(false);
//
//
////		stereodemos(pcv);
//		webcam(pcv);
//
//	}
//
//	private static void webcam(PointCloudViewerSwing pcv) {
//		final Webcam wc = Webcam.getDefault();
//		wc.setViewSize(new Dimension(640,480));
//		ImageGridPanel p = null;
//		wc.open();
//
//		ArrayDeque<BufferedImage> i = new ArrayDeque(2);
//		while (true) {
//
//			final BufferedImage x = wc.getImage();
//			if (x!=null) {
//				if (i.size()==2) i.removeFirst();
//				i.addLast(x);
//
//				final BufferedImage y = i.getFirst();
//				if (y!=x) {
//					try {
//						(p == null ? (p=ShowImages.showGrid(2, "xy", x, y)) : p).setImages(x,y);
//						p.repaint();
//
//						pcv.addCloud(pointCloud(x, y));
//						System.out.println("added");
//					} catch (Throwable t) {
////						System.out.println(t);
//						if (ThreadLocalRandom.current().nextBoolean())
//							i.removeLast();
//						else
//							i.removeFirst();
//					}
//
//				}
//			}
//			Util.sleepMS(200);
//		}
//	}
//
//	private static void stereodemos(PointCloudViewerSwing pcv) {
//		// Successful
//		String name = "minecraft_cave1_";
////		String name = "bobcats_";
////		String name = "mono_wall_";
////		String name = "chicken_";
////		String name = "books_";
//
//		// Successful Failures
////		String name = "triflowers_";
//
//		// Failures
////		String name = "rock_leaves_";
////		String name = "minecraft_distant_";
////		String name = "rockview_";
////		String name = "pebbles_";
////		String name = "skull_";
////		String name = "turkey_";
//
//		BufferedImage b = UtilImageIO.loadImage(UtilIO.pathExample("/home/me/boofcv/data/example/triple/" + name + "01.jpg"));
//		BufferedImage c = UtilImageIO.loadImage(UtilIO.pathExample("/home/me/boofcv/data/example/triple/" + name + "02.jpg"));
//		BufferedImage a = UtilImageIO.loadImage(UtilIO.pathExample("/home/me/boofcv/data/example/triple/" + name + "03.jpg"));
//
//		pcv.addCloud(pointCloud(a, b));
//
//		pcv.addCloud(pointCloud(b, c));
//
//		pcv.addCloud(pointCloud(a, c));
//	}
//
//
//	private static ExampleTrifocalStereoUncalibrated.PointCloudResult pointCloud(BufferedImage buff01, BufferedImage buff02) {
//		Planar<GrayU8> color01 = ConvertBufferedImage.convertFrom(buff01, true, ImageType.pl(3, GrayU8.class));
//		Planar<GrayU8> color02 = ConvertBufferedImage.convertFrom(buff02, true, ImageType.pl(3, GrayU8.class));
//
//		GrayU8 image01 = ConvertImage.average(color01, null);
//		GrayU8 image02 = ConvertImage.average(color02, null);
//
//		// Find a set of point feature matches
//		List<AssociatedPair> matches = computeMatches(buff01, buff02);
//
//		// Prune matches using the epipolar constraint. use a low threshold to prune more false matches
//		var inliers = new ArrayList<AssociatedPair>();
//		DMatrixRMaj F = robustFundamental(matches, inliers, 0.1);
//
//		// Perform self calibration using the projective view extracted from F
//		// Note that P1 = [I|0]
////		System.out.println("Self calibration");
//		DMatrixRMaj P2 = MultiViewOps.fundamentalToProjective(F);
//
//		// Take a crude guess at the intrinsic parameters. Bundle adjustment will fix this later.
//		int width = buff01.getWidth(), height = buff02.getHeight();
//		double fx = width/2.0;
//		double fy = fx;
//		double cx = width/2.0;
//		double cy = height/2.0;
//
//		// Compute a transform from projective to metric by assuming we know the camera's calibration
//		var estimateV = new EstimatePlaneAtInfinityGivenK();
//		estimateV.setCamera1(fx, fy, 0, cx, cy);
//		estimateV.setCamera2(fx, fy, 0, cx, cy);
//
//		var v = new Vector3D_F64(); // plane at infinity
//		if (!estimateV.estimatePlaneAtInfinity(P2, v))
//			throw new RuntimeException("Failed!");
//
//		DMatrixRMaj K = PerspectiveOps.pinholeToMatrix(fx, fy, 0, cx, cy);
//		DMatrixRMaj H = MultiViewOps.createProjectiveToMetric(K, v.x, v.y, v.z, 1, null);
//		DMatrixRMaj P2m = new DMatrixRMaj(3, 4);
//		CommonOps_DDRM.mult(P2, H, P2m);
//
//		// Decompose and get the initial estimate for translation
//		var tmp = new DMatrixRMaj(3, 3);
//		var view1_to_view2 = new Se3_F64();
//		MultiViewOps.decomposeMetricCamera(P2m, tmp, view1_to_view2);
//
//		//------------------------- Setting up bundle adjustment
//		// bundle adjustment will provide a more refined and accurate estimate of these parameters
////		System.out.println("Configuring bundle adjustment");
//
//		// Construct bundle adjustment data structure
//		var structure = new SceneStructureMetric(false);
//		var observations = new SceneObservations();
//
//		// We will assume that the camera has fixed intrinsic parameters
//		final int n = inliers.size();
//		structure.initialize(1, 2, n);
//		observations.initialize(2);
//
//		var bp = new BundlePinholeSimplified();
//		bp.f = fx;
//		structure.setCamera(0, false, bp);
//
//		// The first view is the world coordinate system
//		structure.setView(0, 0, true, new Se3_F64());
//		// Second view was estimated previously
//		structure.setView(1, 0, false, view1_to_view2);
//
//		for (int i = 0; i < n; i++) {
//			AssociatedPair t = inliers.get(i);
//
//			// substract out the camera center from points. This allows a simple camera model to be used and
//			// errors in the this coordinate tend to be non-fatal
//			observations.getView(0).add(i, (float)(t.p1.x - cx), (float)(t.p1.y - cy));
//			observations.getView(1).add(i, (float)(t.p2.x - cx), (float)(t.p2.y - cy));
//
//			// each point is visible in both of the views
//			structure.connectPointToView(i, 0);
//			structure.connectPointToView(i, 1);
//		}
//
//		// initial location of points is found through triangulation
//		MultiViewOps.triangulatePoints(structure, observations);
//
//		//------------------ Running Bundle Adjustment
////		System.out.println("Performing bundle adjustment");
//		var configLM = new ConfigLevenbergMarquardt();
//		configLM.dampeningInitial = 1e-3;
//		configLM.hessianScaling = false;
//		var configSBA = new ConfigBundleAdjustment();
//		configSBA.configOptimizer = configLM;
//
//		// Create and configure the bundle adjustment solver
//		BundleAdjustment<SceneStructureMetric> bundleAdjustment = FactoryMultiView.bundleSparseMetric(configSBA);
//		// prints out useful debugging information that lets you know how well it's converging
////		bundleAdjustment.setVerbose(System.out, null);
//		// Specifies convergence criteria
//		bundleAdjustment.configure(1e-6, 1e-6, 100);
//
//		// Scaling improve accuracy of numerical calculations
//		var bundleScale = new ScaleSceneStructure();
//		bundleScale.applyScale(structure, observations);
//
//		bundleAdjustment.setParameters(structure, observations);
//		bundleAdjustment.optimize(structure);
//
//		// Sometimes pruning outliers help improve the solution. In the stereo case the errors are likely
//		// to already fatal
//		var pruner = new PruneStructureFromSceneMetric(structure, observations);
//		pruner.pruneObservationsByErrorRank(0.85); //0.85
//		pruner.prunePoints(1);
//		bundleAdjustment.setParameters(structure, observations);
//		bundleAdjustment.optimize(structure);
//
//		bundleScale.undoScale(structure, observations);
//
//
////		System.out.println("\nCamera");
////		for (int i = 0; i < structure.cameras.size; i++) {
////			System.out.println(structure.cameras.data[i].getModel().toString());
////		}
////		System.out.println("\n\nworldToView");
////		for (int i = 0; i < structure.views.size; i++) {
////			System.out.println(structure.getParentToView(i).toString());
////		}
//
//		// display the inlier matches found using the robust estimator
////		System.out.println("\n\nComputing Stereo Disparity");
//		BundlePinholeSimplified cp = structure.getCameras().get(0).getModel();
//		var intrinsic = new CameraPinholeBrown();
//		intrinsic.fsetK(cp.f, cp.f, 0, cx, cy, width, height);
//		intrinsic.fsetRadial(cp.k1, cp.k2);
//
//		return computeStereoCloud(image01, image02, color01, color02, intrinsic, intrinsic, structure.getParentToView(1), 0, 250);
//	}
//
//
//	/**
//	 * Use the associate point feature example to create a list of {@link AssociatedPair} for use in computing the
//	 * fundamental matrix.
//	 */
//	public static List<AssociatedPair> computeMatches( BufferedImage left, BufferedImage right ) {
//		DetectDescribePoint detDesc = FactoryDetectDescribe.surfStable(
//				new ConfigFastHessian(0, 2, 400, 1, 9, 4, 4), null, null, GrayF32.class);
////		DetectDescribePoint detDesc = FactoryDetectDescribe.sift(null,new ConfigSiftDetector(2,0,200,5),null,null);
//
//		ScoreAssociation<TupleDesc_F64> scorer = FactoryAssociation.scoreEuclidean(TupleDesc_F64.class, true);
//		AssociateDescription<TupleDesc_F64> associate = FactoryAssociation.greedy(new ConfigAssociateGreedy(true, 0.1), scorer);
//
//		ExampleAssociatePoints<GrayF32, TupleDesc_F64> findMatches =
//				new ExampleAssociatePoints<>(detDesc, associate, GrayF32.class);
//
//		findMatches.associate(left, right);
//
//		List<AssociatedPair> matches = new ArrayList<>();
//		FastAccess<AssociatedIndex> matchIndexes = associate.getMatches();
//
//		for (int i = 0; i < matchIndexes.size; i++) {
//			AssociatedIndex a = matchIndexes.get(i);
//			matches.add(new AssociatedPair(findMatches.pointsA.get(a.src), findMatches.pointsB.get(a.dst)));
//		}
//
//		return matches;
//	}
//	/**
//	 * Given a set of noisy observations, compute the Fundamental matrix while removing the noise.
//	 *
//	 * @param matches List of associated features between the two images
//	 * @param inliers List of feature pairs that were determined to not be noise.
//	 * @return The found fundamental matrix.
//	 */
//	public static DMatrixRMaj robustFundamental( List<AssociatedPair> matches,
//												 List<AssociatedPair> inliers, double inlierThreshold ) {
//		ConfigRansac configRansac = new ConfigRansac();
//		configRansac.inlierThreshold = inlierThreshold;
//		configRansac.iterations = 1000;
//		ConfigFundamental configFundamental = new ConfigFundamental();
//		configFundamental.which = EnumFundamental.LINEAR_7;
//		configFundamental.numResolve = 2;
//		configFundamental.errorModel = ConfigFundamental.ErrorModel.SAMPSON;
//		// geometric error is the most accurate error metric, but also the slowest to compute. See how the
//		// results change if you switch to sampson and how much faster it is. You also should adjust
//		// the inlier threshold.
//
//		ModelMatcher<DMatrixRMaj, AssociatedPair> ransac =
//				FactoryMultiViewRobust.fundamentalRansac(configFundamental, configRansac);
//
//		// Estimate the fundamental matrix while removing outliers
//		if (!ransac.process(matches))
//			throw new IllegalArgumentException("Ransac Failed");
//
//		// save the set of features that were used to compute the fundamental matrix
//		inliers.addAll(ransac.getMatchSet());
//
//		// Improve the estimate of the fundamental matrix using non-linear optimization
//		DMatrixRMaj F = new DMatrixRMaj(3, 3);
//		ModelFitter<DMatrixRMaj, AssociatedPair> refine =
//				FactoryMultiView.fundamentalRefine(1e-8, 400, EpipolarError.SAMPSON);
//		if (!refine.fitModel(inliers, ransac.getModelParameters(), F))
//			throw new IllegalArgumentException("Modelfit Failed");
//
//		// Return the solution
//		return F;
//	}
//}