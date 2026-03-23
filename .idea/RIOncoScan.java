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
            model.train(split.trainFeatures, split.trainLabels, 1500, 0.015); // epochs, learning rate

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

/**
 * CancerDataLoader: loads CSV dataset into memory and handles validation.
 */
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
                if (tokens.length < 2) {
                    System.err.println("Skipping invalid row " + lineNumber + ": not enough columns");
                    continue;
                }

                // Determine diagnosis column: assume 2nd column or last value
                String diag = tokens.length >= 2 ? tokens[1] : tokens[tokens.length - 1];
                int label = parseLabel(diag, lineNumber);

                double[] rowFeatures;
                if (tokens.length == 2) {
                    throw new IOException("No numeric features found in row " + lineNumber);
                } else {
                    // features are the remaining numeric columns after diagnosis
                    int featureCount = tokens.length - 2;
                    if (featureCount <= 0) {
                        throw new IOException("Data row " + lineNumber + " has no features");
                    }
                    rowFeatures = new double[featureCount];
                    for (int j = 2; j < tokens.length; j++) {
                        rowFeatures[j - 2] = Double.parseDouble(tokens[j]);
                    }
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
            for (int i = 0; i < n; i++) sum += features[i][j];
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

/**
 * Neural network with one hidden layer and sigmoid activation.
 */
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
                w1[i][j] = (random.nextGaussian() * 0.01);
            }
        }
        for (int j = 0; j < hiddenSize; j++) b1[j] = 0;

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                w2[i][j] = (random.nextGaussian() * 0.01);
            }
        }
        for (int j = 0; j < outputSize; j++) b2[j] = 0;
    }

    public void train(double[][] trainFeatures, int[] trainLabels, int epochs, double learningRate) {
        if (trainFeatures.length != trainLabels.length) {
            throw new IllegalArgumentException("Feature count and label count must match");
        }
        int n = trainFeatures.length;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            double loss = 0;
            for (int i = 0; i < n; i++) {
                double[] x = trainFeatures[i];
                int y = trainLabels[i];

                // forward
                double[] z1 = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    z1[j] = b1[j];
                    for (int k = 0; k < inputSize; k++) z1[j] += w1[k][j] * x[k];
                    z1[j] = sigmoid(z1[j]);
                }

                double[] z2 = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    z2[j] = b2[j];
                    for (int k = 0; k < hiddenSize; k++) z2[j] += w2[k][j] * z1[k];
                    z2[j] = sigmoid(z2[j]);
                }

                // compute loss
                loss += - (y * Math.log(z2[0] + 1e-12) + (1 - y) * Math.log(1 - z2[0] + 1e-12));

                // backward
                double[] dz2 = new double[outputSize];
                dz2[0] = z2[0] - y;

                double[] dw2j = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    dw2j[j] = dz2[0] * z1[j];
                }
                double db2 = dz2[0];

                double[] dz1 = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    dz1[j] = w2[j][0] * dz2[0] * z1[j] * (1 - z1[j]);
                }

                // update output layer
                for (int j = 0; j < hiddenSize; j++) {
                    w2[j][0] -= learningRate * dw2j[j];
                }
                b2[0] -= learningRate * db2;

                // update hidden layer
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < inputSize; k++) {
                        w1[k][j] -= learningRate * dz1[j] * x[k];
                    }
                    b1[j] -= learningRate * dz1[j];
                }
            }

            if (epoch % 100 == 0 || epoch == 1 || epoch == epochs) {
                double avgLoss = loss / n;
                System.out.println(String.format("Epoch %d/%d, loss=%.6f", epoch, epochs, avgLoss));
            }
        }
    }

    public int[] predict(double[][] testFeatures) {
        int[] predicted = new int[testFeatures.length];
        for (int i = 0; i < testFeatures.length; i++) {
            predicted[i] = predictProbabilities(testFeatures[i]) >= 0.5 ? 1 : 0;
        }
        return predicted;
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
        for (int j = 0; j < hiddenSize; j++) {
            hidden[j] = b1[j];
            for (int k = 0; k < inputSize; k++) {
                hidden[j] += w1[k][j] * input[k];
            }
            hidden[j] = sigmoid(hidden[j]);
        }

        double output = b2[0];
        for (int j = 0; j < hiddenSize; j++) {
            output += w2[j][0] * hidden[j];
        }
        return sigmoid(output);
    }

    public double evaluate(double[][] testFeatures, int[] testLabels) {
        if (testFeatures.length != testLabels.length) {
            throw new IllegalArgumentException("Test set features and labels length mismatch");
        }

        int correct = 0;
        int n = testFeatures.length;
        for (int i = 0; i < n; i++) {
            int p = predict(new double[][]{testFeatures[i]})[0];
            if (p == testLabels[i]) correct++;
        }
        return (double) correct / n;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}

/**
 * PredictionSaver: saves predicted + actual labels to output CSV.
 */
class PredictionSaver {
    private final String filePath;

    public PredictionSaver(String filePath) {
        this.filePath = filePath;
    }

    public void savePredictions(int[] predictions, int[] actual, double[] probabilities) throws IOException {
        if (predictions.length != actual.length || predictions.length != probabilities.length) {
            throw new IllegalArgumentException("Predictions/actual/probabilities length mismatch");
        }

        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("Index,Predicted,Actual,Probability,Correct");
            for (int i = 0; i < predictions.length; i++) {
                writer.println(String.format("%d,%d,%d,%.5f,%s", i, predictions[i], actual[i], probabilities[i], predictions[i] == actual[i]));
            }
        }
    }
}

/**
 * DataSplit: helper to shuffle and split data.
 */
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
        Collections.shuffle(Arrays.asList(indices), new Random(123));

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
