/**
 * LossCategoricalCrossentropy.java
 * Represents the categorical crossentropy loss function.
 */
public class LossCategoricalCrossentropy {

    private double MIN = 1e-7;
    private double MAX = 1 - 1e-7;

    public LossCategoricalCrossentropy() {
        super();
    }

    /**
     * * Forward pass through the categorical crossentropy loss layer.
     * This method calculates the categorical crossentropy loss for each sample in
     * the batch.
     * 
     * @param yPred 2D array of predicted probabilities (batch size x number of
     *              classes).
     * @param yTrue 1D array of true class labels (for binary classification) or 2D
     *              array of one-hot encoded labels (for categorical
     *              classification).
     * @return 1D array of categorical crossentropy losses for each sample in the
     *         batch.
     */
    public double[] forward(double[][] yPred, Object yTrue) {
        // Calculate the categorical crossentropy loss
        int rows = yPred.length;
        int cols = yPred[0].length;

        // Clip data to prevent division by 0 (log(0))
        double[][] clipped = new double[rows][cols];
        int samples = clipped.length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clipped[i][j] = Math.max(this.MIN, Math.min(this.MAX, yPred[i][j])); // Clip values between 1e-15 and 1
                                                                                     // - 1e-15
            }
        }

        // Probabilities for target values
        double[] correctConfidences = new double[samples];

        // Binary classification
        if (yTrue instanceof int[]) {
            correctConfidences = correctConfidences(clipped, (int[]) yTrue);
        } else if (yTrue instanceof int[][]) {
            correctConfidences = correctConfidences(clipped, (int[][]) yTrue);
        } else {
            throw new IllegalArgumentException("Invalid type for yTrue. Expected int[] or double[][]");
        }

        // Calculate the loss
        double[] losses = negativeLogLikelihoods(correctConfidences);
        return losses;
    }

    /**
     * Calculate the mean loss for the batch.
     * 
     * @param output 2D array of predicted probabilities (batch size x number of
     *               classes).
     * @param yTrue  1D array of true class labels (for binary classification) or 2D
     *               array of one-hot encoded labels (for categorical
     *               classification).
     * @return Mean categorical crossentropy loss for the batch.
     */
    public double calculate(double[][] output, Object yTrue) {
        // Calculate the sample losses
        double[] sampleLosses = this.forward(output, yTrue);

        // Calculate the mean loss
        double meanLoss = mean(sampleLosses);

        return meanLoss;
    }

    private double mean(double[] sampleLosses) {
        double sum = 0.0;
        for (double loss : sampleLosses) {
            sum += loss;
        }
        return sum / sampleLosses.length;
    }

    private static double[] correctConfidences(double[][] yPredClipped, int[] yTrue) {
        int samples = yTrue.length;
        double[] correctConfidences = new double[samples];

        for (int i = 0; i < samples; i++) {
            correctConfidences[i] = yPredClipped[i][yTrue[i]];
        }
        return correctConfidences;
    }

    private static double[] correctConfidences(double[][] yPredClipped, int[][] yTrue) {
        int samples = yTrue.length;
        int classes = yTrue[0].length;
        double[] correctConfidences = new double[samples];

        for (int i = 0; i < samples; i++) {
            double sum = 0.0;
            for (int j = 0; j < classes; j++) {
                sum += yPredClipped[i][j] * yTrue[i][j];
            }
            correctConfidences[i] = sum;
        }
        return correctConfidences;
    }

    private double[] negativeLogLikelihoods(double[] correctConfidences) {
        int samples = correctConfidences.length;
        double[] losses = new double[samples];

        for (int i = 0; i < samples; i++) {
            losses[i] = -Math.log(correctConfidences[i]);
        }

        return losses;
    }
}
