package models;

public class Data {
    public double[][] X; // 2D points
    public int[] y; // class labels

    public Data(double[][] X, int[] y) {
        this.X = X;
        this.y = y;
    }
}
