public class ActivationCategoricalCrossentropySoftmaxLoss {

    private ActivationSoftmax activation;
    private LossCategoricalCrossentropy loss;
    private double[][] output; // Output of the softmax activation

    public ActivationCategoricalCrossentropySoftmaxLoss() {
        this.activation = new ActivationSoftmax(); // Softmax activation
        this.loss = new LossCategoricalCrossentropy(); // Categorical crossentropy loss
        super();
    }

    public double forward(double[][] inputs, Object yTrue) {
        // Forward pass through softmax activation
        this.activation.forward(inputs);
        
        // Set the output
        this.output = this.activation.getOutput(); // Get the output from softmax activation

        // Calculate the loss using categorical crossentropy
        return this.loss.calculate(this.output, (double[][]) yTrue); // Calculate the loss using categorical crossentropy
    }

}
