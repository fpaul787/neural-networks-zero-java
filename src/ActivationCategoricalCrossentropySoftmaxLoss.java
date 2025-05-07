public class ActivationCategoricalCrossentropySoftmaxLoss {

    private ActivationSoftmax activation;
    private LossCategoricalCrossentropy loss;
    private double[][] output; // Output of the softmax activation
    double[][] dinputs; // Gradient of the loss with respect to the inputs

    public ActivationCategoricalCrossentropySoftmaxLoss() {
        this.activation = new ActivationSoftmax(); // Softmax activation
        this.loss = new LossCategoricalCrossentropy(); // Categorical crossentropy loss
    }

    /**
     * * Forward pass through the softmax activation and calculate the loss using categorical crossentropy.
     * @param inputs The input data (2D array)
     * @param yTrue The true labels (2D array)
     * @return The loss value calculated using categorical crossentropy.
     */
    public double forward(double[][] inputs, Object yTrue) {
        // Forward pass through softmax activation
        this.activation.forward(inputs);
        
        // Set the output
        this.output = this.activation.getOutput(); // Get the output from softmax activation

        // Calculate the loss using categorical crossentropy
        return this.loss.calculate(this.output, yTrue); // Calculate the loss using categorical crossentropy
    }

    public void backward(double[][] dvalues, Object yTrue) {
        // number of samples
        int samples = dvalues.length;
        // number of classes
        int numClasses = dvalues[0].length;

        int [] yTrueint = null; // Initialize yTrueint

        // If labels are one-hot encoded,
        // turn them into discrete values
        if (yTrue instanceof int[]) {
            yTrueint = (int[]) yTrue;
        }else if (yTrue instanceof double[]) {
            System.out.println("yTrue is double[]");
        } else {
            throw new IllegalArgumentException("yTrue must be an int[] or double[]");
        }

        // Copy to modify the original array
        for (int i = 0; i < samples; i++){
            for (int j = 0; j < numClasses; j++){
                this.dinputs[i][j] = dvalues[i][j]; // Copy the values from dvalues to dinputs
            }
        }

        // Subtract the true class from the output
        for (int i = 0; i < samples; i++){
            dinputs[i][yTrueint[i]] -= 1; // Subtract the true class from the output
        }

        // Normalize the gradient by the number of samples
        for (int i = 0; i < samples; i++){
            for (int j = 0; j < numClasses; j++){
                dinputs[i][j] /= samples; // Normalize the gradient by the number of samples
            }
        }
    }

    /**
     * Returns the output of the softmax activation.
     * @return The output of the softmax activation (2D array).
     */
    public double[][] getOutput() {
        return this.output; // Return the output from softmax activation
    }
}
