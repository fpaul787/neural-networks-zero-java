/**
 * DenseLayer.java
 * Represents a dense layer in a neural network.
 */
public class DenseLayer {

    private int inputSize;
    private int numberOfNeurons;
    private double[][] weights; // Weights for the layer
    private double[] biases; // Biases for the layer
    private double[][] inputs; // Inputs to the layer
    private double[][] output; // Output of the layer
    private double[][] dWeights; // Gradient of weights
    private double[] dBiases; // Gradient of biases
    private double[][] dInputs; // Gradient of inputs

    /**
     * Constructor for the DenseLayer.
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

    /**
     * Forward pass through the layer.
     * * @param input 2D array of inputs (batch size x input size).
     */
    public void forward(double[][] input) {

        this.inputs = input; // Store the inputs for backpropagation
        
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
                this.output[i][j] = sum + biases[j]; // Add bias
            }
        }

    }

    /**
     * Backward pass through the layer.
     * @param dValues Gradient of the loss with respect to the output of this layer (2D array).
     */
    public void backward(double[][] dValues) {
        
        // Gradient on parameters, dot product of inputs and gradient on output
        this.dWeights = new double[inputSize][numberOfNeurons];
        double[][] inputsT = transpose(this.inputs); // Transpose the inputs for matrix multiplication
        this.dWeights = dot(inputsT, dValues); // Gradient of weights

        // Gradient on biases, sum of gradient on output
        this.dBiases = new double[dValues[0].length];
        for (int i = 0; i < dValues.length; i++) {
            for (int j = 0; j < dValues[0].length; j++) {
                this.dBiases[j] += dValues[i][j]; // Sum of gradients on output
            }
        }

        // Gradient on inputs
        double[][] weightsT = transpose(this.weights); // Transpose the weights for matrix multiplication
        this.dInputs = dot(dValues, weightsT); // Gradient of inputs
        
    }

    /**
     * Get the output of the layer.
     * @return 2D array of outputs (batch size x number of neurons).
     */
    public double[][] getOutput() {
        return this.output; // Return the output of the layer
    }

    /**
     * Get the weights of the layer.
     * @return 2D array of weights (input size x number of neurons).
     */
    public double[][] getWeights() {
        return this.weights; // Return the weights of the layer
    }

    /**
     * Get the biases of the layer.
     * @return 1D array of biases (number of neurons).
     */
    public double[] getBiases() {
        return this.biases; // Return the biases of the layer
    }

    /**
     * Get the gradients of the weights.
     * @return 2D array of gradients of weights (input size x number of neurons).
     */
    public double[][] getDWeights() {
        return this.dWeights; // Return the gradient of weights
    }

    /**
     * Get the gradients of the biases.
     * @return 1D array of gradients of biases (number of neurons).
     */
    public double[] getDBiases() {
        return this.dBiases; // Return the gradient of biases
    }

    /**
     * Get the gradients of the inputs.
     * @return 2D array of gradients of inputs (batch size x input size).
     */
    public double[][] getDInputs() {
        return this.dInputs; // Return the gradient of inputs
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

    /**
     * Transpose a matrix.
     * @param matrix 2D array to be transposed.
     * @return Transposed matrix (2D array).
     */
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j]; // Transpose the matrix
            }
        }
        return transposed;
    }

    /**
     * Matrix multiplication (dot product) of two matrices.
     * @param a 
     * @param b 
     * @return Result of the dot product of a and b.
     */
    private double[][] dot(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int rowsB = b.length;
        int colsB = b[0].length;

        if (colsA != rowsB) {
            throw new IllegalArgumentException("Incompatible dimensions for matrix multiplication.");
        }

        double[][] result = new double[rowsA][colsB];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < rowsB; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }
}
