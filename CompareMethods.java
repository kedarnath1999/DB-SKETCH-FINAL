import java.io.*;
import java.util.*;
import java.util.regex.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

public class CompareMethods {

    public static void main(String[] args) {
        
        String trainFile = "./data/rcv1_test.binary";
        
        String[] methods = {
            "AWMsketch",
            "TruncatedModel", 
            "WMSketch" 
        };
        
        int[] budgets = {25,50,75,100,200};

        Map<String, Map<Integer, Double>> errorRateMap = new LinkedHashMap<>();
        Map<String, Map<Integer, Double>> runtimeMap = new LinkedHashMap<>();
        
        Pattern errorPattern = Pattern.compile("(?i).*train error rate\\s*[:=]\\s*([-+]?[0-9]*\\.?[0-9]+).*");
        Pattern timePattern = Pattern.compile("(?i).*train time \\(ms\\)\\s*[:=]\\s*([-+]?[0-9]*\\.?[0-9]+).*");
        
        for (String method : methods) {
            Map<Integer, Double> budgetErrors = new TreeMap<>();
            Map<Integer, Double> budgetRuntimes = new TreeMap<>();
            for (int budget : budgets) {
                List<String> command = new ArrayList<>();
                command.add("java");
                command.add("-cp");
                command.add("json.jar:.:jfreechart-1.5.0.jar:jcommon-1.0.23.jar");
                command.add("WMSketchClassification");
                command.add("--train=" + trainFile);
                command.add("--method=" + method);
                command.add("--topk=" + budget);
                command.add("--depth=" + 4);
                command.add("--log2_width="+9);
                
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
                    int exitCode = process.waitFor();
                    String outputStr = outputBuilder.toString();
                    
                    Matcher mError = errorPattern.matcher(outputStr);
                    double trainErrRate = -1;
                    if (mError.find()) {
                        try {
                            trainErrRate = Double.parseDouble(mError.group(1));
                        } catch (NumberFormatException nfe) {
                            System.err.println("Error parsing train error rate for method " + method + " with TopK " + budget);
                        }
                    } else {
                        System.err.println("Could not extract train error rate for method " + method + " with TopK " + budget);
                        if(debug) {
                            System.err.println("Raw output: " + outputStr);
                        }
                        continue;
                    }
                    
                    Matcher mTime = timePattern.matcher(outputStr);
                    double trainTime = -1;
                    if (mTime.find()) {
                        try {
                            trainTime = Double.parseDouble(mTime.group(1));
                        } catch (NumberFormatException nfe) {
                            System.err.println("Error parsing train time for method " + method + " with budget " + budget);
                        }
                    } else {
                        System.err.println("Could not extract train time for method " + method + " with budget " + budget);
                        if(debug) {
                            System.err.println("Raw output: " + outputStr);
                        }
                        continue;
                    }
                    
                    System.out.println("Method: " + method + ", TopK: " + budget 
                        + " => Train error rate: " + trainErrRate + ", Train time (ms): " + trainTime);
                    
                    budgetErrors.put(budget, trainErrRate);
                    budgetRuntimes.put(budget, trainTime);
                    
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
            errorRateMap.put(method, budgetErrors);
            runtimeMap.put(method, budgetRuntimes);
        }
        
        DefaultCategoryDataset errorDataset = new DefaultCategoryDataset();
        for (String method : methods) {
            Map<Integer, Double> budgetResults = errorRateMap.get(method);
            if (budgetResults != null) {
                for (Map.Entry<Integer, Double> entry : budgetResults.entrySet()) {
                    int budget = entry.getKey();
                    double errorRate = entry.getValue();
                    errorDataset.addValue(errorRate, method, Integer.toString(budget));
                }
            }
        }
        
        
        JFreeChart errorChart = ChartFactory.createLineChart(
                "Classification Error Rate vs TopK For 8KB", 
                "TopK",                       
                "Classification Error Rate",                    
                errorDataset,
                PlotOrientation.VERTICAL,
                true,   
                true,
                false);
        
        
        ChartFrame errorFrame = new ChartFrame("Error Rate Comparison", errorChart);
        errorFrame.pack();
        errorFrame.setVisible(true);

    }
}
