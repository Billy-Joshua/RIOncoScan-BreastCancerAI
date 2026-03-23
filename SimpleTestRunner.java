import java.io.*;
import java.util.*;

/**
 * SimpleTestRunner - Basic unit tests for RIOncoScan components
 * Runs without external dependencies for minimal setup
 */
public class SimpleTestRunner {
    public static void main(String[] args) {
        System.out.println("Running RIOncoScan Unit Tests...\n");

        int passed = 0;
        int total = 0;

        // Test DataSplit
        total++;
        if (testDataSplit()) {
            passed++;
            System.out.println("✓ DataSplit test passed");
        } else {
            System.out.println("✗ DataSplit test failed");
        }

        // Test CancerNeuralNetwork evaluation
        total++;
        if (testNeuralNetwork()) {
            passed++;
            System.out.println("✓ NeuralNetwork evaluation test passed");
        } else {
            System.out.println("✗ NeuralNetwork evaluation test failed");
        }

        // Test CancerDataLoader with sample data
        total++;
        if (testDataLoader()) {
            passed++;
            System.out.println("✓ DataLoader test passed");
        } else {
            System.out.println("✗ DataLoader test failed");
        }

        System.out.println("\nTest Results: " + passed + "/" + total + " tests passed");
        if (passed == total) {
            System.out.println("All tests passed! ✓");
        }
    }

    private static boolean testDataSplit() {
        try {
            // Create sample data
            double[][] features = {
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0},
                {7.0, 8.0}
            };
            int[] labels = {0, 1, 0, 1};

            DataSplit split = DataSplit.shuffleAndSplit(features, labels, 0.5);

            // Check sizes
            if (split.trainFeatures.length != 2 || split.testFeatures.length != 2) {
                return false;
            }

            // Check that all original data is present
            Set<String> originalData = new HashSet<>();
            for (int i = 0; i < features.length; i++) {
                originalData.add(features[i][0] + "," + labels[i]);
            }

            Set<String> splitData = new HashSet<>();
            for (int i = 0; i < split.trainFeatures.length; i++) {
                splitData.add(split.trainFeatures[i][0] + "," + split.trainLabels[i]);
            }
            for (int i = 0; i < split.testFeatures.length; i++) {
                splitData.add(split.testFeatures[i][0] + "," + split.testLabels[i]);
            }

            return originalData.equals(splitData);
        } catch (Exception e) {
            System.err.println("DataSplit test error: " + e.getMessage());
            return false;
        }
    }

    private static boolean testNeuralNetwork() {
        try {
            // Simple 2-input, 1-hidden, 1-output network
            CancerNeuralNetwork nn = new CancerNeuralNetwork(2, 4, 1);

            // Simple linearly separable data: class 0 if x1 + x2 < 1, else class 1
            double[][] trainFeatures = {
                {0.1, 0.1}, {0.2, 0.3}, {0.1, 0.4}, // class 0
                {0.6, 0.6}, {0.7, 0.8}, {0.8, 0.5}  // class 1
            };
            int[] trainLabels = {0, 0, 0, 1, 1, 1};

            // Train for more epochs
            nn.train(trainFeatures, trainLabels, 500, 0.1);

            // Evaluate on same data
            double accuracy = nn.evaluate(trainFeatures, trainLabels);

            // Should achieve high accuracy on this simple problem
            return accuracy > 0.8;
        } catch (Exception e) {
            System.err.println("NeuralNetwork test error: " + e.getMessage());
            return false;
        }
    }

    private static boolean testDataLoader() {
        try {
            // Create temporary test CSV
            String testCsv = "test_sample.csv";
            try (PrintWriter writer = new PrintWriter(testCsv)) {
                writer.println("id,diagnosis,feature1,feature2");
                writer.println("1,M,1.5,2.3");
                writer.println("2,B,3.7,4.1");
                writer.println("3,M,5.2,6.8");
            }

            // Load and verify
            CancerDataLoader loader = new CancerDataLoader(testCsv);
            double[][] features = loader.getFeatures();
            int[] labels = loader.getLabels();

            // Check dimensions
            if (features.length != 3 || features[0].length != 2) {
                System.err.println("Wrong dimensions: " + features.length + "x" + features[0].length);
                return false;
            }

            // Check labels (M=1, B=0)
            if (labels[0] != 1 || labels[1] != 0 || labels[2] != 1) {
                System.err.println("Wrong labels: " + Arrays.toString(labels));
                return false;
            }

            // Check that features are normalized (mean ~0, reasonable range)
            double sum1 = 0, sum2 = 0;
            for (double[] row : features) {
                sum1 += row[0];
                sum2 += row[1];
            }
            double mean1 = sum1 / features.length;
            double mean2 = sum2 / features.length;

            // Normalized features should have mean close to 0
            if (Math.abs(mean1) > 0.1 || Math.abs(mean2) > 0.1) {
                System.err.println("Features not properly normalized: means " + mean1 + ", " + mean2);
                return false;
            }

            // Cleanup
            new File(testCsv).delete();
            return true;

        } catch (Exception e) {
            System.err.println("DataLoader test error: " + e.getMessage());
            return false;
        }
    }
}