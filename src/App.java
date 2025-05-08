import models.Data;

public class App {
    public static void main(String[] args) throws Exception {
        
        // Create dataset
        // 100 samples, 3 classes
        Data dataset = SpiralDataGenerator.createDataset(100, 3); 
        double[][] X = dataset.X;
        int[] y = dataset.y;
        
        // Create a DenseLayer with 2 input features and 3 output neurons
        // First DenseLayer with 2 inputs and 3 neurons
        DenseLayer denseLayer1 = new DenseLayer(2, 3);

        // ReLU activation layer
        ActivationReLU activation1 = new ActivationReLU();

        // Second DenseLayer with 3 input features (from the previous layer) and 3 output neurons
        DenseLayer denseLayer2 = new DenseLayer(3, 3);

        // Softmax classifier's combined loss and activation layer
        ActivationCategoricalCrossentropySoftmaxLoss lossActivation = new ActivationCategoricalCrossentropySoftmaxLoss();

        // Learning rate and momentum
        OptimizerSGD optimizer = new OptimizerSGD(1.0); 

        int epochs = 10001; // Number of epochs
        for (int epoch = 0; epoch < epochs; epoch++) {
            
            // Perfomance of forward pass through the first layer
            // Forward pass through the first layer
            denseLayer1.forward(X); 

            // Forward pass through ReLU activation
            activation1.forward(denseLayer1.getOutput());

            // Perform a forward pass through the second layer
            // Forward pass through the second layer
            denseLayer2.forward(activation1.getOutput()); 

            // Forward pass through the loss layer
            double loss = lossActivation.forward(denseLayer2.getOutput(), y);

            // Get the predicted class labels
            int[] predictions = argMax(lossActivation.getOutput());

            // Calculate accuracy
            double accuracy = calculateAccuracy(predictions, y);

            // Backward pass 
            lossActivation.backward(lossActivation.getOutput(), y);
            denseLayer2.backward(lossActivation.getDInputs());
            activation1.backward(denseLayer2.getDInputs());
            denseLayer1.backward(activation1.getDInputs());

            // Update parameters using SGD optimizer
            optimizer.update_params(denseLayer1);
            optimizer.update_params(denseLayer2);

            // Print loss and accuracy every 100 epochs
            if (epoch % 100 == 0) {
                System.out.printf("epoch: %d, acc: %.3f, loss: %.3f%n", epoch, accuracy, loss);
            }

        }
    }

    /**
     * Get the index of the maximum value in each row of a 2D array.
     * This method iterates through each row of the 2D array and finds the index of the maximum value in that row.
     * @param array 2D array of doubles
     * @return int array containing the index of the maximum value in each row
     *         of the input 2D array.
     */
    private static int[] argMax(double[][] array) {
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
        return maxIndices;
    }

    /**
     * Calculate the accuracy of predictions against the true labels.
     * This method compares the predicted labels with the true labels and calculates the accuracy.
     * @param predictions int array of predicted class labels
     * @param y int array of true class labels
     * @return double representing the accuracy of the predictions
     */
    private static double calculateAccuracy(int[] predictions, int[] y) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
}
