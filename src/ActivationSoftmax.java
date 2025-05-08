/**
 * Softmax activation function implementation.
 * This class applies the softmax function to the input data.
 */
public class ActivationSoftmax {

    private double[][] output;

    public ActivationSoftmax() {
    }

    /**
     * Forward pass through the softmax layer.
     * @param inputs 2D array of inputs (batch size x number of classes).
     */
    public void forward(double[][] inputs) {

        // Unnormalized probabilities
        int rows = inputs.length;
        int cols = inputs[0].length;
        double[][] expValues = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            // Find the maximum value in the row to prevent overflow
            double rowMax = inputs[i][0];
            for (int j = 1; j < cols; j++) {
                if (inputs[i][j] > rowMax) {
                    rowMax = inputs[i][j];
                }
            }

            // Subtract the maximum value from each element in the row and calculate the exponentials
            for (int j = 0; j < cols; j++) {
                expValues[i][j] = Math.exp(inputs[i][j] - rowMax);
            }
        }

        // Normalize the probabilities
        double[][] probabilities = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += expValues[i][j];
            }

            for (int j = 0; j < cols; j++) {
                probabilities[i][j] = expValues[i][j] / sum;
            }
        }

        this.output = probabilities;
    }

    /*
     * Get the output of the softmax layer.
     * @return 2D array of probabilities (batch size x number of classes).
     */
    public double[][] getOutput() {
        return this.output;
    }
}
