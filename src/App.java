import models.Data;

public class App {
    public static void main(String[] args) throws Exception {
        // Create dataset
        Data dataset = SpiralDataGenerator.createDataset(100, 3); // 100 samples, 3 classes

        double[][] X = dataset.X; // 2D points
        double[] y = dataset.y; // class labels
        
        // Create a DenseLayer with 2 input features and 3 output neurons
        // First DenseLayer with 2 inputs and 3 neurons
        DenseLayer layer1 = new DenseLayer(2, 3);

        // ReLU activation layer
        ActivationReLU activation1 = new ActivationReLU();

        // Second DenseLayer with 3 input features (from the previous layer) and 3 output neurons
        DenseLayer layer2 = new DenseLayer(3, 3);

        // Softmax classifier's combined loss and activation layer
        ActivationCategoricalCrossentropySoftmaxLoss loss = new ActivationCategoricalCrossentropySoftmaxLoss();

        int epochs = 1; // Number of epochs
        for (int i = 0; i < epochs; i++) {
            
            // Perfomance of forward pass through the first layer
            layer1.forward(X); // Forward pass through the first layer

            double[][] layer1Output = layer1.getOutput(); // Get the output from the first layer
            activation1.forward(layer1Output); // Forward pass through ReLU activation

            // Perform a forward pass through the second layer
            layer2.forward(activation1.getOutput()); // Forward pass through the second layer

            // backward pass
        }
    }
}
