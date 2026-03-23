import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * RIOncoScan - AI-based Breast Cancer Classification System
 * Entry point that loads data, trains a simple neural network, evaluates, and saves predictions.
 *
 * Example CSV dataset structure (wdbc-like):
 * id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,...
 * 842302,M,17.99,10.38,122.8,1001.0,0.1184,...
 * 842517,B,20.57,17.77,132.9,1326.0,0.08474,...
 *
 * Diagnosis values: M (malignant) or B (benign)
 */
public class RIOncoScan {
    public static void main(String[] args) {
        try {
            String datasetPath = args.length > 0 ? args[0] : "cancer_data.csv";
            String outputPath = args.length > 1 ? args[1] : "predictions.csv";

            System.out.println("Loading dataset: " + datasetPath);
            CancerDataLoader loader = new CancerDataLoader(datasetPath);
            double[][] features = loader.getFeatures();
            int[] labels = loader.getLabels();
            System.out.println("Loaded " + features.length + " samples with " + features[0].length + " features");

            DataSplit split = DataSplit.shuffleAndSplit(features, labels, 0.8);
            System.out.println("Training set: " + split.trainFeatures.length + " samples");
            System.out.println("Test set: " + split.testFeatures.length + " samples");

            CancerNeuralNetwork model = new CancerNeuralNetwork(split.trainFeatures[0].length, 24, 1);
            model.train(split.trainFeatures, split.trainLabels, 1500, 0.015);

            double accuracy = model.evaluate(split.testFeatures, split.testLabels);
            System.out.println(String.format("Test Accuracy: %.2f%%", accuracy * 100));

            int[] predictions = model.predict(split.testFeatures);
            PredictionSaver saver = new PredictionSaver(outputPath);
            saver.savePredictions(predictions, split.testLabels, model.predictProbabilities(split.testFeatures));
            System.out.println("Predictions saved to " + outputPath);

        } catch (IOException e) {
            System.err.println("IO Error: " + e.getMessage());
        } catch (NumberFormatException e) {
            System.err.println("Data format error: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Argument error: " + e.getMessage());
        }
    }
}

class CancerDataLoader {
    private double[][] features;
    private int[] labels;

    public CancerDataLoader(String filePath) throws IOException {
        loadData(filePath);
    }

    private void loadData(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path) || !Files.isReadable(path)) {
            throw new IOException("File not found or not readable: " + filePath);
        }

        List<double[]> featureList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String header = reader.readLine();
            if (header == null) {
                throw new IOException("CSV file is empty: " + filePath);
            }

            String line;
            int lineNumber = 1;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                String[] tokens = line.trim().split(",");
                if (tokens.length < 3) {
                    System.err.println("Skipping invalid row " + lineNumber + ": not enough columns");
                    continue;
                }

                String diag = tokens[1];
                int label = parseLabel(diag, lineNumber);

                double[] rowFeatures = new double[tokens.length - 2];
                for (int j = 2; j < tokens.length; j++) {
                    rowFeatures[j - 2] = Double.parseDouble(tokens[j]);
                }

                featureList.add(rowFeatures);
                labelList.add(label);
            }
        }

        if (featureList.isEmpty()) {
            throw new IOException("No valid data rows found in file: " + filePath);
        }

        int columns = featureList.get(0).length;
        for (double[] f : featureList) {
            if (f.length != columns) {
                throw new IOException("Inconsistent feature count in rows");
            }
        }

        features = featureList.toArray(new double[0][]);
        labels = labelList.stream().mapToInt(Integer::intValue).toArray();
        normalizeFeatures();
    }

    private int parseLabel(String diag, int lineNumber) {
        if (diag.equalsIgnoreCase("M")) return 1;
        if (diag.equalsIgnoreCase("B")) return 0;
        try {
            int numeric = Integer.parseInt(diag);
            if (numeric == 1 || numeric == 0) return numeric;
        } catch (NumberFormatException ignored) {
        }
        throw new IllegalArgumentException("Invalid diagnosis value at line " + lineNumber + ": " + diag);
    }

    private void normalizeFeatures() {
        int n = features.length;
        int m = features[0].length;

        double[] mean = new double[m];
        double[] std = new double[m];

        for (int j = 0; j < m; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += features[i][j];
            }
            mean[j] = sum / n;
        }

        for (int j = 0; j < m; j++) {
            double variance = 0;
            for (int i = 0; i < n; i++) {
                double diff = features[i][j] - mean[j];
                variance += diff * diff;
            }
            std[j] = Math.sqrt(variance / n);
            if (std[j] == 0) std[j] = 1;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                features[i][j] = (features[i][j] - mean[j]) / std[j];
            }
        }
    }

    public double[][] getFeatures() {
        return features;
    }

    public int[] getLabels() {
        return labels;
    }
}

class CancerNeuralNetwork {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    private final double[][] w1;
    private final double[] b1;
    private final double[][] w2;
    private final double[] b2;

    public CancerNeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        w1 = new double[inputSize][hiddenSize];
        b1 = new double[hiddenSize];
        w2 = new double[hiddenSize][outputSize];
        b2 = new double[outputSize];
        initializeParameters();
    }

    private void initializeParameters() {
        Random random = new Random(1234);
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                w1[i][j] = random.nextGaussian() * 0.01;
            }
        }
        Arrays.fill(b1, 0);
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                w2[i][j] = random.nextGaussian() * 0.01;
            }
        }
        Arrays.fill(b2, 0);
    }

    public void train(double[][] trainFeatures, int[] trainLabels, int epochs, double learningRate) {
        if (trainFeatures.length != trainLabels.length) {
            throw new IllegalArgumentException("Feature count and label count must match");
        }

        int n = trainFeatures.length;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double totalLoss = 0;
            for (int i = 0; i < n; i++) {
                double[] x = trainFeatures[i];
                int y = trainLabels[i];

                double[] hidden = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++) {
                    hidden[h] = b1[h];
                    for (int k = 0; k < inputSize; k++) {
                        hidden[h] += w1[k][h] * x[k];
                    }
                    hidden[h] = sigmoid(hidden[h]);
                }

                double[] out = new double[outputSize];
                for (int o = 0; o < outputSize; o++) {
                    out[o] = b2[o];
                    for (int h = 0; h < hiddenSize; h++) {
                        out[o] += w2[h][o] * hidden[h];
                    }
                    out[o] = sigmoid(out[o]);
                }

                totalLoss += - (y * Math.log(out[0] + 1e-12) + (1 - y) * Math.log(1 - out[0] + 1e-12));

                double dOut = out[0] - y;
                double[] dW2 = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++) {
                    dW2[h] = dOut * hidden[h];
                }

                double[] dHidden = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++) {
                    dHidden[h] = w2[h][0] * dOut * hidden[h] * (1 - hidden[h]);
                }

                for (int h = 0; h < hiddenSize; h++) {
                    w2[h][0] -= learningRate * dW2[h];
                }
                b2[0] -= learningRate * dOut;

                for (int h = 0; h < hiddenSize; h++) {
                    for (int k = 0; k < inputSize; k++) {
                        w1[k][h] -= learningRate * dHidden[h] * x[k];
                    }
                    b1[h] -= learningRate * dHidden[h];
                }
            }
            if (epoch % 100 == 0 || epoch == 1 || epoch == epochs) {
                System.out.println(String.format("Epoch %d/%d - loss %.6f", epoch, epochs, totalLoss / n));
            }
        }
    }

    public int[] predict(double[][] testFeatures) {
        int[] preds = new int[testFeatures.length];
        for (int i = 0; i < testFeatures.length; i++) {
            preds[i] = predictProbabilities(testFeatures[i]) >= 0.5 ? 1 : 0;
        }
        return preds;
    }

    public double[] predictProbabilities(double[][] testFeatures) {
        double[] probs = new double[testFeatures.length];
        for (int i = 0; i < testFeatures.length; i++) {
            probs[i] = predictProbabilities(testFeatures[i]);
        }
        return probs;
    }

    private double predictProbabilities(double[] input) {
        double[] hidden = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++) {
            hidden[h] = b1[h];
            for (int k = 0; k < inputSize; k++) {
                hidden[h] += w1[k][h] * input[k];
            }
            hidden[h] = sigmoid(hidden[h]);
        }

        double out = b2[0];
        for (int h = 0; h < hiddenSize; h++) {
            out += w2[h][0] * hidden[h];
        }
        return sigmoid(out);
    }

    public double evaluate(double[][] testFeatures, int[] testLabels) {
        if (testFeatures.length != testLabels.length) {
            throw new IllegalArgumentException("Test features and labels mismatch");
        }
        int correct = 0;
        for (int i = 0; i < testFeatures.length; i++) {
            int pred = predict(new double[][]{testFeatures[i]})[0];
            if (pred == testLabels[i]) correct++;
        }
        return (double) correct / testFeatures.length;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}

class PredictionSaver {
    private final String filePath;

    public PredictionSaver(String filePath) {
        this.filePath = filePath;
    }

    public void savePredictions(int[] predictions, int[] actual, double[] probabilities) throws IOException {
        if (predictions.length != actual.length || predictions.length != probabilities.length) {
            throw new IllegalArgumentException("Predictions, actual, and probabilities must match length");
        }

        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("Index,Predicted,Actual,Probability,Correct");
            for (int i = 0; i < predictions.length; i++) {
                writer.println(String.format("%d,%d,%d,%.5f,%b", i, predictions[i], actual[i], probabilities[i], predictions[i] == actual[i]));
            }
        }
    }
}

class DataSplit {
    public final double[][] trainFeatures;
    public final int[] trainLabels;
    public final double[][] testFeatures;
    public final int[] testLabels;

    DataSplit(double[][] trainFeatures, int[] trainLabels, double[][] testFeatures, int[] testLabels) {
        this.trainFeatures = trainFeatures;
        this.trainLabels = trainLabels;
        this.testFeatures = testFeatures;
        this.testLabels = testLabels;
    }

    static DataSplit shuffleAndSplit(double[][] features, int[] labels, double trainRatio) {
        if (features.length != labels.length) {
            throw new IllegalArgumentException("Features and labels must have same length");
        }
        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("trainRatio must be between 0 and 1");
        }

        int n = features.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Collections.shuffle(Arrays.asList(indices), new Random(1234));

        int trainSize = (int) Math.round(n * trainRatio);
        double[][] trainF = new double[trainSize][];
        int[] trainL = new int[trainSize];
        double[][] testF = new double[n - trainSize][];
        int[] testL = new int[n - trainSize];

        for (int i = 0; i < n; i++) {
            if (i < trainSize) {
                trainF[i] = features[indices[i]];
                trainL[i] = labels[indices[i]];
            } else {
                testF[i - trainSize] = features[indices[i]];
                testL[i - trainSize] = labels[indices[i]];
            }
        }

        return new DataSplit(trainF, trainL, testF, testL);
    }
}
