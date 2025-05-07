import java.util.Random;
import models.*;

public class SpiralDataGenerator {

    /**
     * Copyright (c) 2015 Andrej Karpathy
     * License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
     * Source: https://cs231n.github.io/neural-networks-case-study/
     */
    public static Data createDataset(int numSamples, int numClasses) {
        double[][] X = new double[numSamples*numClasses][2]; // 2D points
        int[] y = new int[numSamples*numClasses]; // class labels

        Random rand = new Random(0); // seed for reproducibility

        for (int classNum = 0; classNum < numClasses; classNum++) {
            int start = numSamples * classNum; // 0

            double[] r = linspace(0, 1, numSamples); // 0 to 1 with numSamples points
            double[] t = linspace(classNum * 4.0, (classNum + 1) * 4.0, numSamples); // 0 to 4 with numSamples points

            // Add Guassian noise
            for (int i = 0; i < numSamples; i++) {
                t[i] += rand.nextGaussian() * 0.2; // add noise to t
            }

            for (int i = 0; i < numSamples; i++) {
                double angle = t[i] * 2.5;
                int ix = start + i; // index for the current sample
                X[ix][0] = r[i] * Math.sin(angle); // x coordinate
                X[ix][1] = r[i] * Math.cos(angle); // y coordinate
                y[ix] = classNum; // class label
            }
        }

        return new Data(X, y); // return the dataset
    }

    /**
     * Creates a linearly spaced array of doubles from start to end with num points. Mimics numpy's linspace.
     * @param start The starting value of the array.
     * @param end The ending value of the array.
     * @param num The number of points in the array.
     * @return A double array containing linearly spaced values from start to end.
     */
    private static double[] linspace(double start, double end, int num){
        double[] result = new double[num];
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            result[i] = start + i * step;
        }
        return result;
    }
}
