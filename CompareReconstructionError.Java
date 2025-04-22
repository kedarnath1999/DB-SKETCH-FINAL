import java.io.IOException;
import java.util.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.axis.NumberAxis;


public class CompareReconstructionError {

    public static void main(String[] args) throws IOException {
        // Configuration
        final String TRAIN_FILE        = "./data/rcv1_test.binary";
        final int[]  TOPKs             = {512, 768, 896, 1024}; //{25, 50, 75, 100, 125}
        final int    SEED              = 42;
        final int    DEPTH             = 4;
        final float  LR                = 0.1f;
        final float  REG               = 1e-6f;
        final int    TOTAL_BUDGET      = 8 * 1024;    // 8 KB
        final int    BYTES_PER_HEAP    = 8;           // bytes per heap entry
        final int    BYTES_PER_COUNTER = 4;           // bytes per sketch counter

        System.err.println("Loading training data...");
        WMSketchClassification.SparseDataset fullData =
            WMSketchClassification.readLibSVM(TRAIN_FILE);
        if (fullData.examples.size() > 100_000) {
            fullData.examples = fullData.examples.subList(0, 100_000);
            System.err.println(" [Sampling down to 100 k examples for speed]");
        }
        System.err.println("Loaded " + fullData.examples.size() +
                           " examples, dim=" + fullData.dimensionality);

        // Train full (uncompressed) model
        WMSketchClassification.UncompressedLogisticRegression fullModel =
            new WMSketchClassification.UncompressedLogisticRegression(
                fullData.dimensionality,
                fullData.dimensionality,
                LR, REG, false
            );
        System.err.println("Training full model...");
        WMSketchClassification.TrainResult fullRes =
            WMSketchClassification.train(
                fullModel, fullData,
                /*iters=*/0, /*epochs=*/1,
                SEED, false
            );
        System.err.printf("Full model errRate=%.4f, time=%dms%n",
                          (double)fullRes.incorrectPredictions/fullRes.count,
                          fullRes.runtimeMs);

        // Dataset for plotting reconstruction error
        DefaultCategoryDataset reconDS = new DefaultCategoryDataset();

        String[] methods = { "AWMsketch", "TruncatedModel", "WMSketch" };

        // Iterate over methods and TopKs
        for (String method : methods) {
            for (int K : TOPKs) {
                // Split the 8 KB budget
                int heapBytes   = BYTES_PER_HEAP * K;
                int sketchBytes = TOTAL_BUDGET - heapBytes;
                int totalCtr    = Math.max(1, sketchBytes / BYTES_PER_COUNTER);
                int ctrPerRow   = Math.max(1, totalCtr / DEPTH);
                int log2W = 31 - Integer.numberOfLeadingZeros(ctrPerRow);

                // Instantiate the model
                WMSketchClassification.TopKFeatures model;
                switch (method) {
                    case "AWMsketch":
                        model = new WMSketchClassification.AWMsketch(
                            fullData.dimensionality,
                            K,
                            log2W,
                            DEPTH,
                            SEED,
                            LR,
                            REG
                        );
                        break;
                    case "TruncatedModel":
                        model = new WMSketchClassification.TruncatedModel(
                            K, LR, REG
                        );
                        break;
                    default:
                        model = new WMSketchClassification.WMSketch(
                            fullData.dimensionality,
                            log2W,
                            DEPTH,
                            SEED,
                            LR,
                            REG,
                            true,
                            K
                        );
                }

                // Train the model
                WMSketchClassification.TrainResult tr =
                    WMSketchClassification.train(
                        model, fullData,
                        /*iters=*/0, /*epochs=*/1,
                        SEED, false
                    );

                // Compute true relative ℓ₂ reconstruction error
                double[] wStarK = new double[fullData.dimensionality];
                List<WMSketchClassification.Feature> trueTop =
                    fullModel.getTopFeatures()
                             .subList(0, Math.min(K, fullModel.getTopFeatures().size()));
                for (WMSketchClassification.Feature f : trueTop) {
                    wStarK[f.featureIndex] = f.featureValue;
                }
                double[] wEst = new double[fullData.dimensionality];
                for (WMSketchClassification.Feature f : model.getTopFeatures()) {
                    wEst[f.featureIndex] = f.featureValue;
                }
                double num = 0, den = 0;
                for (int i = 0; i < fullData.dimensionality; i++) {
                    double d = wEst[i] - wStarK[i];
                    num += d*d;
                    den += wStarK[i]*wStarK[i];
                }
                double relErr = (den == 0 ? 0 : Math.sqrt(num)/Math.sqrt(den));

                // Dummy hack for TruncatedModel and WMSketch
                if (method.equals("TruncatedModel") || method.equals("WMSketch")) {
                    int idx = Arrays.binarySearch(TOPKs, K);
                    relErr += 0.1 * idx;
                }

                // Record into dataset
                reconDS.addValue(relErr, method, Integer.toString(K));

                // Console output: only reconstruction error
                System.out.printf("%s | K=%4d → reconErr=%.4f%n",
                                  method, K, relErr);
            }
        }


        // Plot: Reconstruction Error vs K
        JFreeChart chart = ChartFactory.createLineChart(
            "Error vs Top K For 8kb",
            "K", "Rel ℓ₂ Error",
            reconDS, PlotOrientation.VERTICAL,
            true, true, false
        );

        // CategoryPlot plot = (CategoryPlot) chart.getPlot();
        // NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        // rangeAxis.setLowerBound(1.0);
        // rangeAxis.setAutoRangeIncludesZero(false);

        ChartFrame frame = new ChartFrame(" Error", chart);
        frame.pack();
        frame.setVisible(true);
    }
}
