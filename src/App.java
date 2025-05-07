import models.Data;

public class App {
    public static void main(String[] args) throws Exception {
        
        // Create dataset
        Data dataset = SpiralDataGenerator.createDataset(100, 3); // 100 samples, 3 classes

        double[][] X = dataset.X; // 2D points
        int[] y = dataset.y; // class labels
        
        // Create a DenseLayer with 2 input features and 3 output neurons
        // First DenseLayer with 2 inputs and 3 neurons
        DenseLayer denseLayer1 = new DenseLayer(2, 3);

        // ReLU activation layer
        ActivationReLU activation1 = new ActivationReLU();

        // Second DenseLayer with 3 input features (from the previous layer) and 3 output neurons
        DenseLayer denseLayer2 = new DenseLayer(3, 3);

        // Softmax classifier's combined loss and activation layer
        ActivationCategoricalCrossentropySoftmaxLoss lossActivation = new ActivationCategoricalCrossentropySoftmaxLoss();

        int epochs = 10001; // Number of epochs
        for (int epoch = 0; epoch < epochs; epoch++) {
            
            // Perfomance of forward pass through the first layer
            denseLayer1.forward(X); // Forward pass through the first layer

            activation1.forward(denseLayer1.getOutput()); // Forward pass through ReLU activation

            // Perform a forward pass through the second layer
            denseLayer2.forward(activation1.getOutput()); // Forward pass through the second layer

            double loss = lossActivation.forward(denseLayer2.getOutput(), y); // Forward pass through the loss layer

            // Get the predicted class labels
            int[] predictions = argMax(lossActivation.getOutput()); // Get the predicted class labels

            // Calculate accuracy
            double accuracy = calculateAccuracy(predictions, y); // Calculate accuracy

            // Backward pass through the loss layer
            lossActivation.backward(lossActivation.getOutput(), y); // Backward pass through the loss layer
            denseLayer2.backward(lossActivation.getDInputs()); // Backward pass through the second layer
            activation1.backward(denseLayer2.getDInputs()); // Backward pass through ReLU activation
            denseLayer1.backward(activation1.getDInputs()); // Backward pass through the

            // Print loss and accuracy every 100 epochs
            if (epoch % 100 != 0) {
                System.out.println("Epoch: " + epoch + ", Loss: " + loss + ", Accuracy: " + accuracy);
            }
            
        }
    }

    public static int[] argMax(double[][] array) {
        int[] maxIndices = new int[array.length];
        double maxValue = array[0][0];

        for (int i = 0; i < array.length; i++) {
            maxValue = array[i][0];
            for (int j = 1; j < array[i].length; j++) {
                if (array[i][j] > maxValue) {
                    maxValue = array[i][j];
                    maxIndices[i] = j;
                }
            }
        }
        return maxIndices; // Return the indices of the maximum values
    }

    public static double calculateAccuracy(int[] predictions, int[] y) {
        int correct = 0; // Counter for correct predictions
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == y[i]) {
                correct++; // Increment the counter for correct predictions
            }
        }
        return (double) correct / predictions.length; // Calculate and return the accuracy
    }
}
