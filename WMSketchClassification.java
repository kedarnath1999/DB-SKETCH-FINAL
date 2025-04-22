import java.io.*;
import java.util.*;
import java.util.regex.*;
import org.json.JSONObject;

public class WMSketchClassification {

    // Data Structures
 
    public static class Feature {
        public int featureIndex;
        public float featureValue;
        public Feature(int featureIndex, float featureValue) {
            this.featureIndex = featureIndex;
            this.featureValue = featureValue;
        }
    }

    public static class Triple {
        public int featureId;
        public float featureValue;
        public float weight;
        public Triple(int featureId, float featureValue, float weight) {
            this.featureId = featureId;
            this.featureValue = featureValue;
            this.weight = weight;
        }
    }

    public static class Pair {
        public int key;
        public float featureValue;
        public Pair(int key, float featureValue) {
            this.key = key;
            this.featureValue = featureValue;
        }
    }

    public static class SparseExample {
        public int Given_Sign;
        public List<Feature> featureList;
        public SparseExample(int Given_Sign, List<Feature> featureList) {
            this.Given_Sign = Given_Sign;
            this.featureList = featureList;
        }
    }

    public static class SparseDataset {
        public List<SparseExample> examples = new ArrayList<>();
        public int dimensionality = 0;
    }

    public static class Helper_Function {
        public static float sigmoid(float x) {
            return (float)(1.0 / (1.0 + Math.exp(-x)));
        }
        public static float logisticGrad(float x) {
            return -(1 - sigmoid(x));
        }
    }

    public interface TopKFeatures {
        boolean Value_Prediction(List<Feature> featureList);
        boolean Internal_weiight_update(List<Feature> featureList, boolean Given_Sign);
        List<Feature> getTopFeatures();
        float Learning_update();
    }

    // Base Logistic Regression Model
    public static class UncompressedLogisticRegression implements TopKFeatures {
        protected float[] modelWeights;
        protected float modelBias;
        protected int topKFeatures;
        protected float learningRateInitial;
        protected float regularizationFactor;
        protected long iterationCount;

        public UncompressedLogisticRegression(int dimensionality, int topKFeatures, float learningRateInitial, float regularizationFactor, boolean noBias) {
            this.modelWeights = new float[dimensionality];
            this.modelBias = 0;
            this.topKFeatures = topKFeatures;
            this.learningRateInitial = learningRateInitial;
            this.regularizationFactor = regularizationFactor;
            this.iterationCount = 1;
        }

        protected float product(List<Feature> featureList) {
            float sum = modelBias;
            for (Feature f : featureList) {
                if (f.featureIndex < modelWeights.length) {
                    sum += modelWeights[f.featureIndex] * f.featureValue;
                }
            }
            return sum;
        }

        @Override
        public boolean Value_Prediction(List<Feature> featureList) {
            return product(featureList) >= 0;
        }

        @Override
        public boolean Internal_weiight_update(List<Feature> featureList, boolean Given_Sign) {
            int classifier_label = Given_Sign ? 1 : -1;
            float raw_model_score = product(featureList);
            float Gradient = Helper_Function.logisticGrad(classifier_label * raw_model_score);
            float denominator = 1.0f
                            + learningRateInitial 
                            * regularizationFactor 
                            * iterationCount;
            float scaledLearningRate        = learningRateInitial 
                            / denominator;
            featureList.stream()
            .filter(OneFeature -> OneFeature.featureIndex < modelWeights.length)
            .forEach(OneFeature ->
                modelWeights[OneFeature.featureIndex] -= scaledLearningRate * classifier_label * Gradient * OneFeature.featureValue
            );
 
            float adjustmentFactor = scaledLearningRate * classifier_label;
            float biasChange       = adjustmentFactor * Gradient;
            modelBias              = modelBias - biasChange;
            iterationCount++;
            return raw_model_score >= 0;
        }

        @Override
        public List<Feature> getTopFeatures() {
            List<Feature> list = new ArrayList<>();
            for (int weightIndex = 0; weightIndex < modelWeights.length; weightIndex++) {
                Feature OneFeature = new Feature(weightIndex, modelWeights[weightIndex]);
                list.add(OneFeature);
            }

            list.sort((first, second) -> {
                float First_Value  = Math.abs(first.featureValue);
                float Second_Value = Math.abs(second.featureValue);
                return Float.compare(Second_Value, First_Value);
            });

            List<Feature> resultList;
            if (list.size() > topKFeatures) {
                resultList = list.subList(0, topKFeatures);
            } else {
                resultList = list;
            }
            return resultList;

        }

        @Override
        public float Learning_update() {
            return modelBias;
        }
    }

    // WM-Sketch Implementation
    public static class WMSketch implements TopKFeatures {
        private int k;
        private LogisticSketch logisticSketch;
        private Heap priorityQueue;
        private float[] updatedWeights;
        private int hashTableSize;
    
        public WMSketch(int dimensionality, int Width_of_table, int Deep_Size, int initial_Parameter,
                        float learningRateInitial, float regularizationFactor, boolean medianUpdate, int topKFeatures) {
            this.k = topKFeatures;
            this.logisticSketch = new LogisticSketch(Width_of_table, Deep_Size, initial_Parameter, learningRateInitial, regularizationFactor, medianUpdate);
            this.priorityQueue = new Heap(k);
            this.hashTableSize = 1 << Width_of_table;
            this.updatedWeights = new float[hashTableSize];
        }
    
        @Override
        public boolean Value_Prediction(List<Feature> featureList) {
            float sum = logisticSketch.Learning_b();
            Iterator<Feature> it = featureList.iterator();
            while (it.hasNext()) {
                Feature f = it.next();
                int rawHash   = Integer.hashCode(f.featureIndex);
                int absHash   = Math.abs(rawHash);
                int slot      = absHash % hashTableSize;
                if (priorityQueue.contains(slot)) {
                    float weight  = priorityQueue.getMap().get(slot);
                    float scaleFactor = logisticSketch.Measure();
                    float weighted   = weight * f.featureValue;
                    float pre_feature_value    = weighted * scaleFactor;
                    sum          += pre_feature_value;
                }
            }                       
            return sum >= 0;
        }
    
        @Override
        public boolean Internal_weiight_update(List<Feature> featureList, boolean Given_Sign) {
            logisticSketch.Internal_weiight_update(updatedWeights, featureList, Given_Sign);
            int pos = 0;
            while (pos < featureList.size()) {
                Feature OneFeature     = featureList.get(pos);
                int rawHash      = Integer.hashCode(OneFeature.featureIndex);
                int absHash      = Math.abs(rawHash);
                int slot         = absHash % hashTableSize;
                priorityQueue.insertOrChange(slot, updatedWeights[slot]);
                pos++;
            }
            return Value_Prediction(featureList);
            
        }
    
        @Override
        public List<Feature> getTopFeatures() {
            List<Integer> keyList = priorityQueue.keys();
            for (int i = 0; i < keyList.size(); i++) {
                int slotKey    = keyList.get(i);
                float newValue = updatedWeights[slotKey];
                priorityQueue.changeVal(slotKey, newValue);
            }

            List<Feature> Feature_List = new ArrayList<>();
            Iterator<Pair> pairIter = priorityQueue.items().iterator();
            while (pairIter.hasNext()) {
                Pair p              = pairIter.next();
                float scaledWeight  = p.featureValue * logisticSketch.Measure();
                Feature OneFeature        = new Feature(p.key, scaledWeight);
                Feature_List.add(OneFeature);
            }

            Collections.sort(Feature_List, new Comparator<Feature>() {
                @Override
                public int compare(Feature a, Feature b) {
                    float featureImpactA = Math.abs(a.featureValue);
                    float featureImpactB = Math.abs(b.featureValue);
                    return Float.compare(featureImpactB, featureImpactA);
                }
            });

            List<Feature> result;
            if (Feature_List.size() > k) {
                result = Feature_List.subList(0, k);
            } else {
                result = Feature_List;
            }

            return result;

        }
    
        @Override
        public float Learning_update() {
            return logisticSketch.Learning_b();
        }
    }

    // Active-Set WM-Sketch Implementation
    public static class AWMsketch extends UncompressedLogisticRegression {
        private int dimensionality;
        private LogisticSketch logisticSketch;
        private Heap priorityQueue;
        private float modelBias;
        private float learningRateInitial;
        private float regularizationFactor;
        private float Measure;
        private long iterationCount;
    
        public AWMsketch(int dimensionality, int k, int Width_of_table, int Deep_Size, int initial_Parameter,
                         float learningRateInitial, float regularizationFactor) {
            super(dimensionality, k, learningRateInitial, regularizationFactor, false);
            this.dimensionality = dimensionality;
            this.learningRateInitial = learningRateInitial;
            this.regularizationFactor = regularizationFactor;
            this.modelBias = 0.0f;
            this.Measure = 1.0f;
            this.iterationCount = 0;
            this.logisticSketch = new LogisticSketch(
                Width_of_table, Deep_Size, initial_Parameter,
                learningRateInitial, regularizationFactor, false
            );
            this.priorityQueue = new Heap(k);
        }
    
        public float product(List<Feature> featureValues) {
            float raw_model_score = 0.0f;
            Iterator<Feature> iter = featureValues.iterator();
            while (iter.hasNext()) {
                Feature f = iter.next();
                int weightIndex = f.featureIndex;
                float Current_weight;
                if (priorityQueue.contains(weightIndex)) {
                    Current_weight = priorityQueue.get(weightIndex);
                } else {
                    Current_weight = logisticSketch.get(weightIndex);
                }
                raw_model_score += Current_weight * f.featureValue;
            }
            return raw_model_score * Measure;
        }
    
        private float minAbs(Heap heap) {
            float min = Float.MAX_VALUE;
            Iterator<Float> it = heap.getMap().values().iterator();
            while (it.hasNext()) {
                float Current_weight = it.next();
                float weightMagnitude = Math.abs(Current_weight);
                if (weightMagnitude < min) {
                    min = weightMagnitude;
                }
            }
            return min;
        }
    
        @Override
        public boolean Value_Prediction(List<Feature> featureValues) {
            float raw_model_score = product(featureValues) + modelBias;
            return raw_model_score >= 0 ? true : false;
        }
    
        @Override
        public boolean Internal_weiight_update(List<Feature> featureValues, boolean Given_Sign) {
            if (featureValues.isEmpty()) {
                return modelBias >= 0;
            }
    
            float tmp2 = product(featureValues);
            float raw_model_score = tmp2 + modelBias;
            int classifier_label = Given_Sign ? 1 : -1;
    
            float tmp1 = 1.0f
                       + learningRateInitial
                         * regularizationFactor
                         * iterationCount;
            float scaledLearningRate = learningRateInitial / tmp1;
    
            boolean predictedLabel = raw_model_score >= 0 ? true : false;
            float g = Helper_Function.logisticGrad(classifier_label * raw_model_score);
    
            Measure = Measure * (1 - scaledLearningRate * regularizationFactor);
    
            int i = 0;
            while (i < featureValues.size()) {
                Feature f = featureValues.get(i);
                i++;
                int weightIndex = f.featureIndex;
                float Adjusted_weight = scaledLearningRate * classifier_label * g * f.featureValue;
    
                if (priorityQueue.contains(weightIndex)) {
                    float Updated_weight = priorityQueue.get(weightIndex) - Adjusted_weight;
                    priorityQueue.changeVal(weightIndex, Updated_weight);
                } else {
                    float storedWeight = logisticSketch.get(weightIndex);
                    float Updated_weight = storedWeight - Adjusted_weight;
                    logisticSketch.Internal_weiight_update(weightIndex, -Adjusted_weight);
    
                    int heapSize = priorityQueue.getMap().size();
                    if (heapSize < priorityQueue.capacity
                     || Math.abs(Updated_weight) > minAbs(priorityQueue)) {
                        priorityQueue.insertOrChange(weightIndex, Updated_weight);
                    }
                }
            }
    
            modelBias = modelBias - scaledLearningRate * classifier_label * g;
            iterationCount = iterationCount + 1;
    
            return predictedLabel;
        }
    
        @Override
        public List<Feature> getTopFeatures() {
            List<Pair> items = priorityQueue.items();
    
            for (int j = 0; j < items.size(); j++) {
                Pair p = items.get(j);
                p.featureValue = p.featureValue * Measure;
            }
    
            Collections.sort(items, new Comparator<Pair>() {
                @Override
                public int compare(Pair a, Pair b) {
                    return Float.compare(
                        Math.abs(b.featureValue),
                        Math.abs(a.featureValue)
                    );
                }
            });
    
            List<Feature> Feature_List = new ArrayList<>();
            Iterator<Pair> pairIterator = items.iterator();
            while (pairIterator.hasNext()) {
                Pair p = pairIterator.next();
                Feature_List.add(new Feature(p.key, p.featureValue));
            }
    
            return Feature_List;
        }
    
        @Override
        public float Learning_update() {
            return modelBias;
        }
    }
    

// Truncated Model
public static class TruncatedModel implements TopKFeatures {
    private float modelBias;
    private float learningRateInitial;
    private float regularizationFactor;
    private float Measure;
    private long iterationCount;
    private Heap priorityQueue;
    private int capacity;

    public TruncatedModel(int k, float learningRateInitial, float regularizationFactor) {
        this.capacity = k;
        this.modelBias = 0.0f;
        this.learningRateInitial = learningRateInitial;
        this.regularizationFactor = regularizationFactor;
        this.Measure = 1.0f;
        this.iterationCount = 0;
        this.priorityQueue = new Heap(k);
    }

    @Override
    public List<Feature> getTopFeatures() {
        List<Pair> items = priorityQueue.items();

        for (int weightIndex = 0; weightIndex < items.size(); weightIndex++) {
            Pair p = items.get(weightIndex);
            p.featureValue = p.featureValue * Measure;
        }

        Collections.sort(items, new Comparator<Pair>() {
            @Override
            public int compare(Pair a, Pair b) {
                float featureImpactA = Math.abs(a.featureValue);
                float featureImpactB = Math.abs(b.featureValue);
                return Float.compare(featureImpactB, featureImpactA);
            }
        });

        List<Feature> Feature_List = new ArrayList<>();
        int i = 0;
        while (i < items.size()) {
            Pair p = items.get(i);
            Feature_List.add(new Feature(p.key, p.featureValue));
            i++;
        }
        return Feature_List;
    }

    private float getWeight(int key) {
        if (priorityQueue.contains(key)) {
            return priorityQueue.get(key);
        } else {
            return 0.0f;
        }
    }

    private float product(List<Feature> featureValues) {
        float sum = 0.0f;
        int i = 0;
        while (i < featureValues.size()) {
            Feature f = featureValues.get(i);
            float Current_weight = getWeight(f.featureIndex);
            sum += Current_weight * f.featureValue;
            i++;
        }
        return sum * Measure;
    }

    @Override
    public boolean Value_Prediction(List<Feature> featureValues) {
        float score = product(featureValues) + modelBias;
        return score >= 0 ? true : false;
    }

    @Override
    public boolean Internal_weiight_update(List<Feature> featureValues, boolean Given_Sign) {
        float raw = product(featureValues);
        float raw_model_score = raw + modelBias;

        int classifier_label = Given_Sign ? 1 : -1;

        float rateScalingFactor = 1.0f
                    + learningRateInitial * regularizationFactor * iterationCount;
        float scaledLearningRate    = learningRateInitial / rateScalingFactor;

        float regularizationDecay = 1.0f - scaledLearningRate * regularizationFactor;
        Measure       = Measure * regularizationDecay;

        float g = Helper_Function.logisticGrad(classifier_label * raw_model_score);

        int j = 0;
        while (j < featureValues.size()) {
            Feature f = featureValues.get(j);
            int key    = f.featureIndex;
            float Adjusted_weight = scaledLearningRate * classifier_label * g * f.featureValue / Measure;
            float priorWeight  = getWeight(key);
            float Updated_weight  = priorWeight - Adjusted_weight;
            priorityQueue.insertOrChange(key, Updated_weight);
            j++;
        }

        modelBias      = modelBias - scaledLearningRate * classifier_label * g;
        iterationCount = iterationCount + 1;

        return raw_model_score >= 0;
    }

    @Override
    public float Learning_update() {
        return modelBias;
    }
}


    // Supporting Classes for the Sketch Implementation
    public static class LogisticSketch {
        private float modelBias;
        private float Measure;
        private float[] modelWeights;
        private int hashTableSize;
    
        public LogisticSketch(int Width_of_table, int Deep_Size, int initial_Parameter,
                              float learningRateInitial, float regularizationFactor, boolean medianUpdate) {
            int size = 1 << Width_of_table;
            this.hashTableSize = size;
            this.modelWeights = new float[size];
            this.modelBias = 0.0f;
            this.Measure     = 1.0f;
        }
    
        private int hashIndex(int featureIndex) {
            int raw   = Integer.hashCode(featureIndex);
            int abs   = Math.abs(raw);
            return abs % hashTableSize;
        }
    
        public boolean Value_Prediction(List<Feature> featureList) {
            float sum = modelBias;
    
            Iterator<Feature> it = featureList.iterator();
            while (it.hasNext()) {
                Feature f = it.next();
                int weightIndex   = hashIndex(f.featureIndex);
                float Current_weight   = modelWeights[weightIndex];
                float pre_feature_value = Current_weight * f.featureValue;
                sum += pre_feature_value;
            }
    
            return sum >= 0 ? true : false;
        }
    
        public boolean Internal_weiight_update(float[] updatedWeights, List<Feature> featureList, boolean Given_Sign) {
            int classifier_label = Given_Sign ? 1 : -1;
    
            float raw_model_score = modelBias;
            Iterator<Feature> it1 = featureList.iterator();
            while (it1.hasNext()) {
                Feature f = it1.next();
                int weightIndex = hashIndex(f.featureIndex);
                float Current_weight = modelWeights[weightIndex];
                float pre_feature_value = Current_weight * f.featureValue;
                raw_model_score += pre_feature_value;
            }
    
            float yZ   = classifier_label * raw_model_score;
            float Gradient = Helper_Function.logisticGrad(yZ);
            float scaledLearningRate   = 0.1f;
    
            Iterator<Feature> it2 = featureList.iterator();
            while (it2.hasNext()) {
                Feature f = it2.next();
                int weightIndex = hashIndex(f.featureIndex);
                float priorWeight = modelWeights[weightIndex];
                float Adjusted_weight = scaledLearningRate * Gradient * classifier_label * f.featureValue;
                float Updated_weight  = priorWeight - Adjusted_weight;
                modelWeights[weightIndex] = Updated_weight;
            }
    
            int i = 0;
            while (i < hashTableSize) {
                updatedWeights[i] = modelWeights[i];
                i++;
            }
    
            return Value_Prediction(featureList);
        }
    
        public void Internal_weiight_update(int key, float Adjusted_weight) {
            int weightIndex    = hashIndex(key);
            float priorWeight = modelWeights[weightIndex];
            float Updated_weight = priorWeight + Adjusted_weight;
            modelWeights[weightIndex] = Updated_weight;
        }
    
        public float Measure() {
            return Measure;
        }
    
        public float Learning_b() {
            return modelBias;
        }
    
        public float get(int key) {
            int weightIndex = hashIndex(key);
            return modelWeights[weightIndex];
        }
    }
    

    public static class Heap {
        public int capacity;
        private PriorityQueue<Pair> pq;
        private Map<Integer, Float> map;
    
        public Heap(int capacity) {
            this.capacity = capacity;
            this.map = new HashMap<>();
            this.pq = new PriorityQueue<>(
                new Comparator<Pair>() {
                    @Override
                    public int compare(Pair a, Pair b) {
                        return Double.compare(
                            Math.abs(a.featureValue),
                            Math.abs(b.featureValue)
                        );
                    }
                }
            );
        }
    
        public boolean contains(int key) {
            return map.containsKey(key);
        }
    
        public float get(int key) {
            return map.get(key);
        }
    
        public void changeVal(int key, float featureValue) {
            if (map.containsKey(key)) {
                map.put(key, featureValue);
                rebuildPQ();
            }
        }
    
        public void insertOrChange(int key, float featureValue) {
            if (map.containsKey(key)) {
                map.put(key, featureValue);
                return;
            }
    
            if (map.size() < capacity) {
                map.put(key, featureValue);
                return;
            }
    
            float minAbs = Float.MAX_VALUE;
            int minKey   = -1;
            Iterator<Map.Entry<Integer, Float>> entryIter = map.entrySet().iterator();
            while (entryIter.hasNext()) {
                Map.Entry<Integer, Float> entry = entryIter.next();
                float absVal = Math.abs(entry.getValue());
                if (absVal < minAbs) {
                    minAbs = absVal;
                    minKey  = entry.getKey();
                }
            }
    
            if (Math.abs(featureValue) > minAbs) {
                map.remove(minKey);
                map.put(key, featureValue);
            }
        }
    
        public Optional<Pair> insert(int key, float featureValue) {
            if (map.containsKey(key)) {
                changeVal(key, featureValue);
                return Optional.empty();
            }
    
            if (map.size() < capacity) {
                map.put(key, featureValue);
                pq.add(new Pair(key, featureValue));
                return Optional.empty();
            }
    
            Pair smallest = pq.peek();
            if (Math.abs(featureValue) > Math.abs(smallest.featureValue)) {
                pq.poll();
                map.remove(smallest.key);
                map.put(key, featureValue);
                pq.add(new Pair(key, featureValue));
                return Optional.of(smallest);
            } else {
                return Optional.empty();
            }
        }
    
        public List<Integer> keys() {
            return new ArrayList<>(map.keySet());
        }
    
        public Map<Integer, Float> getMap() {
            return map;
        }
    
        private void rebuildPQ() {
            pq.clear();
            Iterator<Map.Entry<Integer, Float>> iter = map.entrySet().iterator();
            while (iter.hasNext()) {
                Map.Entry<Integer, Float> entry = iter.next();
                pq.add(new Pair(entry.getKey(), entry.getValue()));
            }
        }
    
        public List<Pair> items() {
            rebuildPQ();
            List<Pair> items = new ArrayList<>();
            Iterator<Map.Entry<Integer, Float>> it = map.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry<Integer, Float> entry = it.next();
                items.add(new Pair(entry.getKey(), entry.getValue()));
            }
            return items;
        }
    }
    
    // PMI Implementation
    public static class PMI implements TopKFeatures {
        private int dimensionality;
        private int topKFeatures;
        private float learningRateInitial;
        private float regularizationFactor;
        private long iterationCount;
        private float modelBias;
        private float[] modelWeights;
        private double smooth;
        private Map<Integer, Integer> positiveFeatureCounts;
        private Map<Integer, Integer> negativeFeatureCounts;
        private int totalPositiveExamples;
        private int totalNegativeExamples;
    
        public PMI(int dimensionality, int topKFeatures, float learningRateInitial, float regularizationFactor) {
            this.dimensionality              = dimensionality;
            this.topKFeatures            = topKFeatures;
            this.learningRateInitial     = learningRateInitial;
            this.regularizationFactor    = regularizationFactor;
            this.iterationCount          = 1;
            this.modelBias               = 0.0f;
            this.modelWeights            = new float[dimensionality];
            this.smooth                  = 1.0;
            this.positiveFeatureCounts   = new HashMap<>();
            this.negativeFeatureCounts   = new HashMap<>();
            this.totalPositiveExamples   = 0;
            this.totalNegativeExamples   = 0;
        }
    
        private float product(List<Feature> featureList) {
            float sum = modelBias;
            for (Feature f : featureList) {
                int weightIndex = f.featureIndex;
                if (weightIndex < modelWeights.length) {
                    sum += modelWeights[weightIndex] * f.featureValue;
                }
            }
            return sum;
        }
    
        private double computePMI(int featureIndex, boolean Given_Sign) {
            int countForLabel = Given_Sign
                ? positiveFeatureCounts.getOrDefault(featureIndex, 0)
                : negativeFeatureCounts.getOrDefault(featureIndex, 0);
    
            int totalCountForFeature = positiveFeatureCounts.getOrDefault(featureIndex, 0)
                                     + negativeFeatureCounts.getOrDefault(featureIndex, 0);
    
            int totalLabelCount = Given_Sign
                ? totalPositiveExamples
                : totalNegativeExamples;
    
            int totalExamples = totalPositiveExamples + totalNegativeExamples;
    
            double featureGivenLabelProb = (countForLabel + smooth) / (totalLabelCount + smooth * dimensionality);
            double featureProb = (totalCountForFeature + 2 * smooth) / (totalExamples + 2 * smooth * dimensionality);
    
            if (featureProb <= 0 || featureGivenLabelProb <= 0) {
                return 0.0;
            }
            return Math.log(featureGivenLabelProb / featureProb);
        }
    
        @Override
        public boolean Value_Prediction(List<Feature> featureList) {
            float raw_model_score = product(featureList);
            return raw_model_score >= 0;
        }
    
        @Override
        public boolean Internal_weiight_update(List<Feature> featureList, boolean Given_Sign) {
            if (Given_Sign) {
                totalPositiveExamples++;
            } else {
                totalNegativeExamples++;
            }
            for (Feature f : featureList) {
                int weightIndex = f.featureIndex;
                if (Given_Sign) {
                    positiveFeatureCounts.put(weightIndex, positiveFeatureCounts.getOrDefault(weightIndex, 0) + 1);
                } else {
                    negativeFeatureCounts.put(weightIndex, negativeFeatureCounts.getOrDefault(weightIndex, 0) + 1);
                }
            }
    
            int classifier_label = Given_Sign ? 1 : -1;
            float raw_model_score = product(featureList);
            float Gradient = Helper_Function.logisticGrad(classifier_label * raw_model_score);
            float scaledLearningRate = learningRateInitial / (1.0f + learningRateInitial * regularizationFactor * iterationCount);
            for (Feature f : featureList) {
                int weightIndex = f.featureIndex;
                if (weightIndex < modelWeights.length) {
                    modelWeights[weightIndex] -= scaledLearningRate * classifier_label * Gradient * f.featureValue;
                }
            }
            modelBias -= scaledLearningRate * classifier_label * Gradient;
            iterationCount++;
            return raw_model_score >= 0;
        }
    
        @Override
        public List<Feature> getTopFeatures() {
            List<Feature> list = new ArrayList<>();
            for (int i = 0; i < modelWeights.length; i++) {
                list.add(new Feature(i, modelWeights[i]));
            }
            list.sort((a, b) -> Float.compare(Math.abs(b.featureValue), Math.abs(a.featureValue)));
            return list.size() > topKFeatures ? list.subList(0, topKFeatures) : list;
        }
    
        @Override
        public float Learning_update() {
            return modelBias;
        }
    
        public Map<Integer, Double> getTopFeaturesPMI() {
            List<Feature> topFeatures = getTopFeatures();
            Map<Integer, Double> featureAssociationMap = new HashMap<>();
            for (Feature f : topFeatures) {
                double associationScore = computePMI(f.featureIndex, true);
                featureAssociationMap.put(f.featureIndex, associationScore);
            }
            return featureAssociationMap;
        }
    }
    
    

    // Training 
    public static class TrainResult {
        public long runtimeMs;
        public int incorrectPredictions;
        public int count;
        public TrainResult(long runtimeMs, int incorrectPredictions, int count) {
            this.runtimeMs = runtimeMs;
            this.incorrectPredictions = incorrectPredictions;
            this.count = count;
        }
    }

    public static TrainResult train(TopKFeatures model, SparseDataset dataset,
                                    int iters, int epochs, int initial_Parameter, boolean sample) {
        int incorrectPredictions = 0;
        int count = 0;
        long startTime = System.currentTimeMillis();
        Random rand = new Random(initial_Parameter);
        if (iters == 0) {
            for (int e = 0; e < epochs; e++) {
                for (SparseExample example : dataset.examples) {
                    boolean predictedLabel = model.Internal_weiight_update(example.featureList, example.Given_Sign == 1);
                    if (predictedLabel != (example.Given_Sign == 1)) incorrectPredictions++;
                    count++;
                }
            }
        } else {
            for (int i = 0; i < iters; i++) {
                SparseExample example = dataset.examples.get(rand.nextInt(dataset.examples.size()));
                boolean predictedLabel = model.Internal_weiight_update(example.featureList, example.Given_Sign == 1);
                if (predictedLabel != (example.Given_Sign == 1)) incorrectPredictions++;
                count++;
            }
        }
        long runtime = System.currentTimeMillis() - startTime;
        return new TrainResult(runtime, incorrectPredictions, count);
    }

    public static class TestResult {
        public long runtimeMs;
        public float precision;
        public float recall;
        public TestResult(long runtimeMs, float precision, float recall) {
            this.runtimeMs = runtimeMs;
            this.precision = precision;
            this.recall = recall;
        }
    }

    public static TestResult test(TopKFeatures model, SparseDataset dataset) {
        int tp = 0, fp = 0, fn = 0;
        long startTime = System.currentTimeMillis();
        for (SparseExample example : dataset.examples) {
            boolean classifier_label = (example.Given_Sign == 1);
            boolean predictedLabel = model.Value_Prediction(example.featureList);
            if (classifier_label && predictedLabel) {
                tp++;
            }           
            if (!classifier_label && predictedLabel) {
                fp++;
            }           
            if (classifier_label && !predictedLabel) {
                fn++;
            }            
        }
        long runtime = System.currentTimeMillis() - startTime;
        float precision = (tp + fp == 0) ? 1.0f : (float) tp / (tp + fp);
        float recall = (tp + fn == 0) ? 1.0f : (float) tp / (tp + fn);
        return new TestResult(runtime, precision, recall);
    }

    // Data Loading (LIBSVM Format)
    public static SparseDataset readLibSVM(String filePath) throws IOException {
        SparseDataset dataset = new SparseDataset();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] tokens = line.split("\\s+");
            int Given_Sign = Integer.parseInt(tokens[0]);
            List<Feature> featureList = new ArrayList<>();
            for (int i = 1; i < tokens.length; i++) {
                String[] pair = tokens[i].split(":");
                int featureIndex = Integer.parseInt(pair[0]);
                float featureValue = Float.parseFloat(pair[1]);
                featureList.add(new Feature(featureIndex, featureValue));
                dataset.dimensionality = Math.max(dataset.dimensionality, featureIndex + 1);
            }
            dataset.examples.add(new SparseExample(Given_Sign, featureList));
        }
        br.close();
        return dataset;
    }

    // Command-Line Parsing and Main
    public static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--")) {
                arg = arg.substring(2);
                if (arg.contains("=")) {
                    String[] parts = arg.split("=", 2);
                    map.put(parts[0], parts[1]);
                } else {
                    if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
                        map.put(arg, args[i + 1]);
                        i++;
                    } else {
                        map.put(arg, "true");
                    }
                }
            } else if (arg.startsWith("-")) {
                arg = arg.substring(1);
                if (arg.contains("=")) {
                    String[] parts = arg.split("=", 2);
                    map.put(parts[0], parts[1]);
                } else {
                    if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
                        map.put(arg, args[i + 1]);
                        i++;
                    } else {
                        map.put(arg, "true");
                    }
                }
            }
        }
        return map;
    }

    public static void main(String[] args) {
        Map<String, String> argMap = parseArgs(args);
        if (!argMap.containsKey("train")) {
            System.err.println("Error: trainingFilePath must be specified");
            System.exit(1);
        }
        String method = argMap.getOrDefault("method", "AWMsketch");
        String trainingFilePath = argMap.get("train");
        String testingFilePath = argMap.getOrDefault("test", "");
        int Width_of_table = Integer.parseInt(argMap.getOrDefault("log2_width", "10"));
        int Deep_Size = Integer.parseInt(argMap.getOrDefault("Deep_Size", "1"));
        int initial_Parameter = argMap.containsKey("initial_Parameter") ? Integer.parseInt(argMap.get("initial_Parameter"))
                                              : (int)System.currentTimeMillis();
        int iters = Integer.parseInt(argMap.getOrDefault("iters", "0"));
        int epochs = Integer.parseInt(argMap.getOrDefault("epochs", "1"));
        int topKFeatures = Integer.parseInt(argMap.getOrDefault("topk", "512"));
        float learningRateInitial = Float.parseFloat(argMap.getOrDefault("lr_init", "0.1"));
        float regularizationFactor = Float.parseFloat(argMap.getOrDefault("l2_reg", "1e-6"));
        float smooth = Float.parseFloat(argMap.getOrDefault("count_smooth", "1.0"));
        boolean medianUpdate = argMap.containsKey("median_update");
        boolean noBias = argMap.containsKey("no_bias");
        boolean sample = argMap.containsKey("sample");

        System.err.println("Reading training data from " + trainingFilePath);
        SparseDataset trainDataset = null;
        try {
            long start = System.currentTimeMillis();
            trainDataset = readLibSVM(trainingFilePath);
            long dataLoadMs = System.currentTimeMillis() - start;
            System.err.println("Read training data in " + dataLoadMs + "ms");
        } catch (IOException e) {
            System.err.println("Error reading training data: " + e.getMessage());
            System.exit(1);
        }
        
                if (topKFeatures == 0) {
                    topKFeatures = trainDataset.dimensionality;
                }
        
                JSONObject params = new JSONObject();
                params.put("method", method);
                System.err.println(params.toString(2));
        
                TopKFeatures model = null;
                switch (method) {
                    case "UncompressedLogisticRegression":
                        model = new UncompressedLogisticRegression(trainDataset.dimensionality, topKFeatures, learningRateInitial, regularizationFactor, noBias);
                        break;
                    case "WMSketch":
                        model = new WMSketch(trainDataset.dimensionality, Width_of_table, Deep_Size, initial_Parameter, learningRateInitial, regularizationFactor, medianUpdate, topKFeatures);
                        break;
                    case "AWMsketch":
                        model = new AWMsketch(trainDataset.dimensionality, topKFeatures, Width_of_table, Deep_Size, initial_Parameter, learningRateInitial, regularizationFactor);
                        break;
                    case "TruncatedModel":
                        model = new TruncatedModel(topKFeatures, learningRateInitial, regularizationFactor);
                        break;
                    case "PMI":
                        model = new PMI(trainDataset.dimensionality, topKFeatures, learningRateInitial, regularizationFactor);
                        break;
                    default:
                        System.err.println("Error: invalid method " + method);
                        System.err.println("Options: UncompressedLogisticRegression, WMSketch, AWMsketch, TruncatedModel, PMI");
                        System.exit(1);
                }
        
                // Train
                TrainResult trainingResults = train(model, trainDataset, iters, epochs, initial_Parameter, sample);
                JSONObject results = new JSONObject();
                results.put("Training_time", trainingResults.runtimeMs);
                results.put("Train_Error_Count", trainingResults.incorrectPredictions);
                results.put("Total_no_of_features_trained", trainingResults.count);
                results.put("Train_error_rate", (double) trainingResults.incorrectPredictions / trainingResults.count);
                results.put("Learning_b", model.Learning_update());
        
        
        List<Feature> topFeaturesList = model.getTopFeatures();
        List<Integer> indices = new ArrayList<>();
        List<Float> weightsList = new ArrayList<>();
        for (Feature f : topFeaturesList) {
            indices.add(f.featureIndex);
            weightsList.add(f.featureValue);
        }
        results.put("top_indices", indices);
        results.put("top_weights", weightsList);

        if (method.equals("PMI")) {
            PMI pmModel = (PMI) model;
            Map<Integer, Double> featureAssociationMap = pmModel.getTopFeaturesPMI();
            results.put("top_feature_pmi", featureAssociationMap);
        }

        // Output JSON
        //System.out.println(results.toString(2));

        StringBuilder sb = new StringBuilder();
        sb.append("Parameters:\n");
        sb.append("  Method: " + params.get("method") + "\n\n");
        sb.append("Results:\n");
        sb.append("  Train time (ms): " + trainingResults.runtimeMs + "\n");
        sb.append("  Train error count: " + trainingResults.incorrectPredictions + "\n");
        sb.append("  Train count: " + trainingResults.count + "\n");
        sb.append("  Train error rate: " + ((double) trainingResults.incorrectPredictions / trainingResults.count) + "\n");
        sb.append("  Bias: " + model.Learning_update() + "\n");
        sb.append("  Top indices: " + indices.toString() + "\n");
        sb.append("  Top weights: " + weightsList.toString() + "\n");
        if (method.equals("PMI")) {
            PMI pmModel = (PMI) model;
            sb.append("  Top feature PMI: " + pmModel.getTopFeaturesPMI().toString() + "\n");
        }
        System.out.println(sb.toString());
    }
}
