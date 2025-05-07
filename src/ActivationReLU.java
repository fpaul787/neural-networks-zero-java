public class ActivationReLU {

    private double[][] inputs; // Input to the layer
    private double[][] output; // Output of the layer
    private double[][] dInputs; // Gradient of inputs

    public ActivationReLU() {
    }

    /**
     * Forward pass through the ReLU activation function.
     * This method applies the ReLU activation function to the input data.
     * The negative values are set to zero, and the positive values remain unchanged.
     * The ReLU function is defined as f(x) = max(0, x).
     * 
     * @param inputs The input data (2D array)
     */
    public void forward(double[][] inputs) {
        this.inputs = inputs; // Store the input for backpropagation
        this.output = new double[inputs.length][inputs[0].length]; // Initialize output array
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                output[i][j] = Math.max(0, inputs[i][j]); // Apply ReLU activation function
            }
        }
    }

    /**
     * Backward pass through the ReLU activation function.
     * @param dValues Gradient of the loss with respect to the output of this layer (2D array).
     */
    public void backward(double[][] dValues) {
        // Copy original since we will need to modify it
        dInputs = new double[dValues.length][dValues[0].length]; // Initialize dInputs array
        for (int i = 0; i < dValues.length; i++) {
            for (int j = 0; j < dValues[0].length; j++) {
                dInputs[i][j] = dValues[i][j];
            }
        }

        // Zero gradient for negative inputs (could merge with the above loop)
        for (int i = 0; i < dValues.length; i++) {
            for (int j = 0; j < dValues[0].length; j++) {
                if (inputs[i][j] <= 0) {
                    dInputs[i][j] = 0; // Set gradient to zero for negative inputs
                }
            }
        }
    }

    /**
     * Output of the ReLU activation function.
     * 
     * @return The output of the ReLU activation function (2D array)
     */
    public double[][] getOutput() {
        return output; // Return the output of the layer
    }

    /**
     * Get the gradient of the inputs.
     * @return The gradient of the inputs (2D array)
     */
    public double[][] getDInputs() {
        return dInputs; // Return the gradient of inputs
    }
}
