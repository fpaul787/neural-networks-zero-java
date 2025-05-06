public class ActivationReLU {

    private double[][] inputs; // Input to the layer
    private double[][] output; // Output of the layer


    public ActivationReLU() {
    }

    /**
     * Forward pass through the ReLU activation function.
     * The ReLU function is defined as f(x) = max(0, x).
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
     * Output of the ReLU activation function.
     * @return The output of the ReLU activation function (2D array)
     */
    public double[][] getOutput() {
        return output; // Return the output of the layer
    }
}
