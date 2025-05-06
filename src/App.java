import models.Data;

public class App {
    public static void main(String[] args) throws Exception {
        // Create dataset
        Data dataset = SpiralDataGenerator.createDataset(100, 3); // 100 samples, 3 classes
        double[][] X = dataset.X; // 2D points
        double[] y = dataset.y; // class labels
        

        
        // Create a DenseLayer with 3 inputs and 2 neurons
        // DenseLayer layer = new DenseLayer(3, 2);

        
    }
}
