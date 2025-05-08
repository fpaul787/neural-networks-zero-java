

public class OptimizerSGD {
    
    private double learningRate; 
    private double decay; 
    private double momentum;
    private int iterations = 0;

    public OptimizerSGD(double learningRate, double decay, double momentum) {
        this.learningRate = learningRate; // Initialize learning rate
        this.decay = decay; // Initialize decay rate
        this.momentum = momentum; // Initialize momentum
        this.iterations = 0; // Initialize iterations
    }

    public void update_params(DenseLayer layer){
        double [][] weights = layer.getWeights();
        double [] biases = layer.getBiases();

        double [][] dWeights = layer.getDWeights();
        double [] dBiases = layer.getDBiases();

        double [][] updatedWeights = new double[weights.length][weights[0].length];
        double [] updatedBiases = new double[biases.length];

        // update weights
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                updatedWeights[i][j] -= this.learningRate * dWeights[i][j];
            }
        }

        // update biases
        for (int i = 0; i < biases.length; i++) {
            updatedBiases[i] -= this.learningRate * dBiases[i];
        }

        layer.setWeights(updatedWeights);
        layer.setBiases(updatedBiases);
    }
}
