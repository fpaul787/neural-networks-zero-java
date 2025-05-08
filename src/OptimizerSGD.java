
public class OptimizerSGD {

    private double learningRate;
    private double momentum;

    public OptimizerSGD(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    public OptimizerSGD(double learningRate) {
        this.learningRate = learningRate;
        this.momentum = 0.0;
    }

    public OptimizerSGD() {
        this.learningRate = 1.0;
        this.momentum = 0.0;
    }

    public void update_params(DenseLayer layer) {
        double[][] weights = layer.getWeights();
        double[] biases = layer.getBiases();

        double[][] dWeights = layer.getDWeights();
        double[] dBiases = layer.getDBiases();

        double[][] updatedWeights = new double[weights.length][weights[0].length];
        double[] updatedBiases = new double[biases.length];

        if (this.momentum > 0) {


        } else {

            // Compute weight updates
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    updatedWeights[i][j] = -this.learningRate * dWeights[i][j];
                }
            }

            // Compute bias updates
            for (int i = 0; i < biases.length; i++) {
                updatedBiases[i] = -this.learningRate * dBiases[i];
            }

            // Apply the updates to the weights and biases
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    updatedWeights[i][j] += weights[i][j];
                }
            }

            for (int i = 0; i < biases.length; i++) {
                updatedBiases[i] += biases[i];
            }

            layer.setWeights(updatedWeights);
            layer.setBiases(updatedBiases);
        }
    }
}
