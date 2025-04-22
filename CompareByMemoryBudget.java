import java.io.*;
import java.util.*;
import java.util.regex.*;
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



public class CompareByMemoryBudget {

    public static void main(String[] args) {
        boolean debug = true;
        String trainFile = "./data/rcv1_test.binary";

        String[] methods = {
            "UncompressedLogisticRegression",
            "AWMsketch",
            "WMSketch"
        };

        Map<String, int[]> memoryConfigs = new LinkedHashMap<>();
        memoryConfigs.put("2KB", new int[]{8, 2});     // 256 x 2 = 512 counters * 4 bytes = 2048 bytes => 2KB
        memoryConfigs.put("4KB", new int[]{8, 4});
        memoryConfigs.put("8KB", new int[]{9, 4});
        memoryConfigs.put("64KB", new int[]{11, 8});

        Pattern errorPattern = Pattern.compile("(?i).*train error rate\\s*[:=]\\s*([-+]?[0-9]*\\.?[0-9]+).*");
        Pattern timePattern = Pattern.compile("(?i).*train time \\(ms\\):\\s*([-+]?[0-9]*\\.?[0-9]+).*");

        Map<String, Map<String, Double>> errorRateMap = new LinkedHashMap<>();
        Map<String, Map<String, Double>> runtimeMap = new LinkedHashMap<>();

        for (String method : methods) {
            Map<String, Double> methodErrors = new LinkedHashMap<>();
            Map<String, Double> methodRuntimes = new LinkedHashMap<>();
            for (Map.Entry<String, int[]> entry : memoryConfigs.entrySet()) {
                String label = entry.getKey();
                int[] config = entry.getValue();
                int Width_of_table = config[0];
                int depth = config[1];

                List<String> command = new ArrayList<>();
                command.add("java");
                command.add("-cp");
                command.add("json.jar:.:jfreechart-1.5.0.jar:jcommon-1.0.23.jar");
                command.add("WMSketchClassification");
                command.add("--train=" + trainFile);
                command.add("--method=" + method);
                command.add("--log2_width=" + Width_of_table);
                command.add("--depth=" + depth);

                ProcessBuilder pb = new ProcessBuilder(command);
                pb.redirectErrorStream(true);
                try {
                    Process process = pb.start();
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    StringBuilder outputBuilder = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        outputBuilder.append(line).append("\n");
                    }
                    process.waitFor();
                    String outputStr = outputBuilder.toString();

                    Matcher mError = errorPattern.matcher(outputStr);
                    double trainErrRate = -1;
                    if (mError.find()) {
                        try {
                            trainErrRate = Double.parseDouble(mError.group(1));
                        } catch (NumberFormatException nfe) {
                            System.err.println("Error parsing train error rate for method " + method + ", " + label);
                        }
                    } else {
                        System.err.println("Could not extract train error rate for method " + method + ", " + label);
                        if (debug) {
                            System.err.println("Raw output:\n" + outputStr);
                        }
                        continue;
                    }

                    Matcher mTime = timePattern.matcher(outputStr);
                    double trainTime = -1;
                    if (mTime.find()) {
                        try {
                            trainTime = Double.parseDouble(mTime.group(1));
                        } catch (NumberFormatException nfe) {
                            System.err.println("Error parsing train time for method " + method + ", " + label);
                        }
                    } else {
                        System.err.println("Could not extract train time for method " + method + ", " + label);
                        if (debug) {
                            System.err.println("Raw output:\n" + outputStr);
                        }
                        continue;
                    }

                    System.out.println("Method: " + method + ", Budget: " + label +
                                       " => Train error rate: " + trainErrRate +
                                       ", Train time (ms): " + trainTime);

                    methodErrors.put(label, trainErrRate);
                    methodRuntimes.put(label, trainTime);

                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
            errorRateMap.put(method, methodErrors);
            runtimeMap.put(method, methodRuntimes);
        }

        DefaultCategoryDataset errorDataset = new DefaultCategoryDataset();
        for (String method : methods) {
            Map<String, Double> values = errorRateMap.get(method);
            if (values != null) {
                for (Map.Entry<String, Double> entry : values.entrySet()) {
                    errorDataset.addValue(entry.getValue(), method, entry.getKey());
                }
            }
        }

        DefaultCategoryDataset runtimeDataset = new DefaultCategoryDataset();
        for (String method : methods) {
            Map<String, Double> values = runtimeMap.get(method);
            if (values != null) {
                for (Map.Entry<String, Double> entry : values.entrySet()) {
                    runtimeDataset.addValue(entry.getValue(), method, entry.getKey());
                }
            }
        }

        JFreeChart errorChart = ChartFactory.createLineChart(
                "Error Rate vs Memory Budget",
                "Memory Budget",
                "Classification Error Rate",
                errorDataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        JFreeChart runtimeChart = ChartFactory.createLineChart(
                "Training Runtime vs Memory Budget",
                "Memory Budget",
                "Training Runtime (ms)",
                runtimeDataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        CategoryPlot errorPlot = (CategoryPlot) errorChart.getPlot();
        LineAndShapeRenderer errorRenderer =
            (LineAndShapeRenderer) errorPlot.getRenderer();
        
        errorRenderer.setSeriesPaint(0, Color.RED);
        errorRenderer.setSeriesStroke(0, new BasicStroke(3.0f));
        errorRenderer.setSeriesPaint(1, Color.BLUE);
        errorRenderer.setSeriesStroke(1, new BasicStroke(3.0f));
        errorRenderer.setSeriesPaint(2, Color.BLACK);
        errorRenderer.setSeriesStroke(2, new BasicStroke(3.0f));
        
        CategoryPlot runPlot = (CategoryPlot) runtimeChart.getPlot();
        LineAndShapeRenderer runRenderer =
            (LineAndShapeRenderer) runPlot.getRenderer();
        
        runRenderer.setSeriesPaint(0, Color.RED);
        runRenderer.setSeriesStroke(0, new BasicStroke(3.0f));
        runRenderer.setSeriesPaint(1, Color.BLUE);
        runRenderer.setSeriesStroke(1, new BasicStroke(3.0f));
        runRenderer.setSeriesPaint(2, Color.BLACK);
        runRenderer.setSeriesStroke(2, new BasicStroke(3.0f));

        ChartFrame errorFrame = new ChartFrame("Error Rate by Budget", errorChart);
        errorFrame.pack();
        errorFrame.setVisible(true);

        ChartFrame runtimeFrame = new ChartFrame("Runtime Comparison", runtimeChart);
        runtimeFrame.pack();
        runtimeFrame.setVisible(true);

    }
}
