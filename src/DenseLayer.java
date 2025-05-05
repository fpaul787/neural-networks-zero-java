/**
 * DenseLayer.java
 * Represents a dense layer in a neural network.
 */
public class DenseLayer {

    private Integer inputSize;
    private Integer numberOfNeurons;
    private double[][] weights; // Weights for the layer
    private double[] biases; // Biases for the layer

    /**
     * @param inputSize Input size of the layer.
     * @param numberOfNeurons Number of neurons in the layer.
     * @throws IllegalArgumentException if inputSize or numberOfNeurons is null or less than 1.
     */
    public DenseLayer(Integer inputSize, Integer numberOfNeurons) {
        this.inputSize = inputSize;
        this.numberOfNeurons = numberOfNeurons;

        if (inputSize == null || numberOfNeurons == null) {
            throw new IllegalArgumentException("Input size and number of neurons must not be null.");
        }

        if (inputSize < 1 || numberOfNeurons < 1) {
            throw new IllegalArgumentException("Input size and number of neurons must be greater than 0.");
        }

        // Initialize weights and biases here
        this.weights = new double[numberOfNeurons][inputSize];
        this.biases = new double[numberOfNeurons];

        this.weights = randomizeWeights(numberOfNeurons, inputSize); // Random initialization of weights
    }

    private double[][] randomizeWeights(int rows, int cols) {
        double[][] randomWeights = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                randomWeights[i][j] = Math.random(); // Random initialization of weights
            }
        }
        return randomWeights;
    }


}
