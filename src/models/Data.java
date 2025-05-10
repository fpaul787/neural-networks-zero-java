package models;

/**
 * Data class to hold the dataset
 */
public class Data {
    public double[][] X;
    public int[] y;

    public Data(double[][] X, int[] y) {
        this.X = X;
        this.y = y;
    }

    /**
     * Get X values
     * 
     * @return
     */
    public double[][] getX() {
        return X;
    }

    /**
     * Get y values
     * 
     * @return
     */
    public int[] getY() {
        return y;
    }
}
