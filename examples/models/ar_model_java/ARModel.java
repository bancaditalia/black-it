import java.util.Random;

public class ARModel {

    public static void main(String[] args) {

        double constant = Double.parseDouble(args[0]);
        double autoregressiveParameter = Double.parseDouble(args[1]);

        int nPeriods = Integer.parseInt(args[2]);
        long seed = Long.parseLong(args[3]);

        double[] timeSeries = new double[nPeriods];
        timeSeries[0] = 0;

        Random rnd = new Random();
        rnd.setSeed(seed);

        for (int i = 1; i < nPeriods; i++) {
            timeSeries[i] = constant + autoregressiveParameter * timeSeries[i-1] + rnd.nextGaussian();
       }

	   for (int i = 0; i < nPeriods; i++) {
	        System.out.println(timeSeries[i]);
	   }
    }
}