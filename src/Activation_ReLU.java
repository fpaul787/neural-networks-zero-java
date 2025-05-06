public class Activation_ReLU {

    private double[][] inputs; // Input to the layer
    private double[][] output; // Output of the layer


    public Activation_ReLU() {
    }

    public void forward(double[][] inputs) {
        this.inputs = inputs; // Store the input for backpropagation
        this.output = new double[inputs.length][inputs[0].length]; // Initialize output array
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                output[i][j] = Math.max(0, inputs[i][j]); // Apply ReLU activation function
            }
        }
    }
}
