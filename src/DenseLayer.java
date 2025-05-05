/**
 * DenseLayer.java
 * Represents a dense layer in a neural network.
 */
public class DenseLayer {

    private int inputSize;
    private int numberOfNeurons;
    private double[][] weights; // Weights for the layer
    private double[] biases; // Biases for the layer
    private double[][] output; // Output of the layer

    /**
     * @param inputSize Input size of the layer.
     * @param numberOfNeurons Number of neurons in the layer.
     * @throws IllegalArgumentException if inputSize or numberOfNeurons is null or less than 1.
     */
    public DenseLayer(int inputSize, int numberOfNeurons) {
        this.inputSize = inputSize;
        this.numberOfNeurons = numberOfNeurons;

        if (inputSize < 1 || numberOfNeurons < 1) {
            throw new IllegalArgumentException("Input size and number of neurons must be greater than 0.");
        }

        // Initialize weights and biases here
        this.weights = new double[inputSize][numberOfNeurons];
        this.biases = new double[numberOfNeurons];

        this.weights = initializeWeights(inputSize, numberOfNeurons); // Random initialization of weights
        this.biases = initializeBiases(numberOfNeurons); // Initialization of biases
    }

    public void forward(double[][] input) {
        // Forward pass through the layer
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size does not match the layer's input size.");
        }
        if (input[0].length != inputSize) {
            throw new IllegalArgumentException("Input size does not match the layer's input size.");
        }

        // Calculate the output values from inputs, weights, and biases
        int batchSize = input.length;
        int numberOfNeurons = this.numberOfNeurons;

        // Initialize the output array
        this.output = new double[batchSize][numberOfNeurons];

        // Perform the matrix multiplication (dot product) and add biases
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < numberOfNeurons; j++) {
                double sum = 0.0;
                for (int k = 0; k < inputSize; k++) {
                    sum += input[i][k] * weights[k][j]; // Weighted sum of inputs
                }
                sum += biases[j]; // Add bias
            }
        }

    }

    private double[][] initializeWeights(int rows, int cols) {
        double[][] randomWeights = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                randomWeights[i][j] = Math.random(); // Random initialization of weights
            }
        }
        return randomWeights;
    }

    private double[] initializeBiases(int numberOfNeurons) {
        double[] biases = new double[numberOfNeurons];
        for (int i = 0; i < numberOfNeurons; i++) {
            biases[i] = 0; // initialization of biases
        }
        return biases;
    }


}
