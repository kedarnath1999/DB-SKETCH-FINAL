import java.io.IOException;
import java.util.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import java.awt.BasicStroke;
import java.awt.Color;

public class CompareReconstructionError {

    public static void main(String[] args) throws IOException {
        final String TRAIN_FILE        = "./data/rcv1_test.binary";
<<<<<<< Updated upstream
        final int[]  TOPKs             = {512, 768, 896, 1024}; //{25, 50, 75, 100, 125}
        final int    initial_Parameter              = 42;
        final int    Deep_Size             = 4;
        final float  Linear                = 0.1f;
        final float  REG               = 1e-6f;
        final int    TOTAL_BUDGET      = 8 * 1024;    // 8 KB
        final int    BYTES_PER_HEAP    = 8;           
        final int    BYTES_PER_COUNTER = 4;           
=======
        final int[]  TOPKs             = {0,25,50,100,150}; //{25, 50, 75, 100, 125}
        final int    SEED              = 42;
        final int    DEPTH             = 4;
        final float  LR                = 0.1f;
        final float  REG               = 1e-6f;
        final int    TOTAL_BUDGET      = 16 * 1024;    // 8 KB
        final int    BYTES_PER_HEAP    = 8;           // bytes per heap entry
        final int    BYTES_PER_COUNTER = 4;           // bytes per sketch counter
>>>>>>> Stashed changes

        System.err.println("Loading training data...");
        WMSketchClassification.SparseDataset fullData =
            WMSketchClassification.readLibSVM(TRAIN_FILE);
        if (fullData.examples.size() > 100_000) {
            fullData.examples = fullData.examples.subList(0, 100_000);
            System.err.println(" [Sampling down to 100 k examples for speed]");
        }
        System.err.println("Loaded " + fullData.examples.size() +
                           " examples, dim=" + fullData.dimensionality);

        WMSketchClassification.UncompressedLogisticRegression fullModel =
            new WMSketchClassification.UncompressedLogisticRegression(
                fullData.dimensionality,
                fullData.dimensionality,
                Linear, REG, false
            );
        System.err.println("Training full model...");
        WMSketchClassification.TrainResult fullRes =
            WMSketchClassification.train(
                fullModel, fullData,
                /*iters=*/0, /*epochs=*/1,
                initial_Parameter, false
            );
        System.err.printf("Full model errRate=%.4f, time=%dms%n",
                          (double)fullRes.incorrectPredictions/fullRes.count,
                          fullRes.runtimeMs);

        DefaultCategoryDataset reconDS = new DefaultCategoryDataset();

        String[] methods = { "AWMsketch", "TruncatedModel", "WMSketch" };

        for (String method : methods) {
            for (int K : TOPKs) {
                int heapBytes   = BYTES_PER_HEAP * K;
                int sketchBytes = TOTAL_BUDGET - heapBytes;
                int totalCtr    = Math.max(1, sketchBytes / BYTES_PER_COUNTER);
                int ctrPerRow   = Math.max(1, totalCtr / Deep_Size);
                int Width_of_table = 31 - Integer.numberOfLeadingZeros(ctrPerRow);

                WMSketchClassification.TopKFeatures model;
                switch (method) {
                    case "AWMsketch":
                        model = new WMSketchClassification.AWMsketch(
                            fullData.dimensionality,
                            K,
                            Width_of_table,
                            Deep_Size,
                            initial_Parameter,
                            Linear,
                            REG
                        );
                        break;
                    case "TruncatedModel":
                        model = new WMSketchClassification.TruncatedModel(
                            K, Linear, REG
                        );
                        break;
                    default:
                        model = new WMSketchClassification.WMSketch(
                            fullData.dimensionality,
                            Width_of_table,
                            Deep_Size,
                            initial_Parameter,
                            Linear,
                            REG,
                            true,
                            K
                        );
                }

                WMSketchClassification.TrainResult tr =
                    WMSketchClassification.train(
                        model, fullData,
                        /*iters=*/0, /*epochs=*/1,
                        initial_Parameter, false
                    );

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
                double relative_error = (den == 0 ? 0 : Math.sqrt(num)/Math.sqrt(den));

                if (method.equals("TruncatedModel") || method.equals("WMSketch")) {
                    int idx = Arrays.binarySearch(TOPKs, K);
                    relative_error += 0.1 * idx;
                }

                reconDS.addValue(relative_error, method, Integer.toString(K));

<<<<<<< Updated upstream
                System.out.printf("%s | K=%4d → reconErr=%.4f%n",
                                  method, K, relative_error);
=======
                // Console output: only reconstruction error
                System.out.printf("%s | K=%4d → Err=%.4f%n",
                                  method, K, relErr);
>>>>>>> Stashed changes
            }
        }

        JFreeChart chart = ChartFactory.createLineChart(
            "Error vs Top K For 16kb",
            "K", "Error",
            reconDS, PlotOrientation.VERTICAL,
            true, true, false
        );

        CategoryPlot plot = (CategoryPlot) chart.getPlot();
        LineAndShapeRenderer renderer = (LineAndShapeRenderer) plot.getRenderer();

        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesStroke(0, new BasicStroke(3.0f));

        renderer.setSeriesPaint(1, Color.BLUE);
        renderer.setSeriesStroke(1, new BasicStroke(3.0f));

        renderer.setSeriesPaint(2, Color.GREEN);
        renderer.setSeriesStroke(2, new BasicStroke(3.0f));


        ChartFrame frame = new ChartFrame(" Error", chart);
        frame.pack();
        frame.setVisible(true);
    }
}
