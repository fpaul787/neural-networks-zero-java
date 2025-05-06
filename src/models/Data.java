package models;

public class Data {
    public double[][] X; // 2D points
    public double[] y; // class labels

    public Data(double[][] X, double[] y) {
        this.X = X;
        this.y = y;
    }
}
