public class ActivationCategoricalCrossentropySoftmaxLoss {

    private ActivationSoftmax activation;
    private LossCategoricalCrossentropy loss;
    private double[][] output;
    double[][] dinputs;

    public ActivationCategoricalCrossentropySoftmaxLoss() {
        // Softmax activation
        this.activation = new ActivationSoftmax();

         // Categorical crossentropy loss
        this.loss = new LossCategoricalCrossentropy();
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
        
        // Get the output (predicted values) from softmax activation
        this.output = this.activation.getOutput(); 

        // Calculate the loss using categorical crossentropy
        return this.loss.calculate(this.output, yTrue);
    }

    /**
     * Backward pass through the loss layer.
     * This method calculates the gradient of the loss with respect to the inputs.
     * @param dvalues The gradient of the loss with respect to the output of the softmax activation (2D array).
     * @param yTrue The true labels (1D array).
     */
    public void backward(double[][] dvalues, Object yTrue) {
        // Initialize dinputs
        this.dinputs = new double[dvalues.length][dvalues[0].length];

        int samples = dvalues.length;

        int numClasses = dvalues[0].length;

        int [] yTrueint = null;

        // Check if labels are binary or categorical
        if (yTrue instanceof int[]) {
            yTrueint = (int[]) yTrue;
        }else if (yTrue instanceof int[][]) {
            System.out.println("yTrue is int[][]");
        } else {
            throw new IllegalArgumentException("yTrue must be an int[] or int[][]");
        }

        // Copy to modify the original array
        for (int i = 0; i < samples; i++){
            for (int j = 0; j < numClasses; j++){
                this.dinputs[i][j] = dvalues[i][j];
            }
        }

        // Subtract the true class from the output
        for (int i = 0; i < samples; i++){
            this.dinputs[i][yTrueint[i]] -= 1;
        }

        // Normalize the gradient by the number of samples
        for (int i = 0; i < samples; i++){
            for (int j = 0; j < numClasses; j++){
                this.dinputs[i][j] /= samples;
            }
        }
    }

    /**
     * Returns the output of the softmax activation.
     * @return The output of the softmax activation (2D array).
     */
    public double[][] getOutput() {
        return this.output;
    }

    /**
     * Returns the gradient of the loss with respect to the inputs.
     * @return The gradient of the loss with respect to the inputs (2D array).
     */
    public double[][] getDInputs() {
        return this.dinputs;
    }
}
