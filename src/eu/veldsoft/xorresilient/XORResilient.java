package eu.veldsoft.xorresilient;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.mathutil.BoundMath;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

class ActivationFadingSin implements ActivationFunction {
	private final ActivationSIN SIN = new ActivationSIN();
	private double period = 1.0D;

	public ActivationFadingSin(double period) {
		this.period = period;
	}

	public void activationFunction(double[] values, int start, int size) {
		for (int i = start; i < (start + size) && i < values.length; i++) {
			double x = values[i] / period;

			if (x < -Math.PI || x > Math.PI) {
				values[i] = BoundMath.sin(x) / Math.abs(x);
			} else {
				values[i] = BoundMath.sin(x);
			}
		}
	}

	public double derivativeFunction(double before, double after) {
		double x = before / period;

		if (x < -Math.PI || x > Math.PI) {
			return BoundMath.cos(x) / Math.abs(x) - BoundMath.sin(x) / (x * Math.abs(x));
		} else {
			return BoundMath.cos(x);
		}
	}

	public ActivationFunction clone() {
		return new ActivationFadingSin(period);
	}

	public String getFactoryCode() {
		return null;
	}

	public String[] getParamNames() {
		return SIN.getParamNames();
	}

	public double[] getParams() {
		return SIN.getParams();
	}

	public boolean hasDerivative() {
		return true;
	}

	public void setParam(int index, double value) {
		SIN.setParam(index, value);
	}

}

public class XORResilient {
	private static final double EPSILON = 0.00001;
	private static final double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 } };
	private static final double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
	private static final NeuralDataSet TRAINING = new BasicNeuralDataSet(XOR_INPUT, XOR_IDEAL);

	/*
	 * It is used for time measurement calibration.
	 */
	static {
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 4));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();

		Train train = new ResilientPropagation(network, TRAINING);
		train.iteration();
	}

	private static void experiment(String title, ActivationFunction activation, double epsilon) {
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(activation, true, 2));
		network.addLayer(new BasicLayer(activation, true, 4));
		network.addLayer(new BasicLayer(activation, false, 1));
		network.getStructure().finalizeStructure();
		network.reset();

		final Train train = new ResilientPropagation(network, TRAINING);

		int epoch = 1;
		long start = System.currentTimeMillis();

		System.out.println(title + " Neural Network Training:");
		System.out.println();

		System.out.println("Time\tEpoch\tError");
		do {
			train.iteration();
			System.out.println((System.currentTimeMillis() - start) + "\t" + epoch + "\t" + train.getError());
			epoch++;
		} while (train.getError() > epsilon);
		System.out.println();

		System.out.println(title + " Neural Network Results:");
		System.out.println();

		System.out.println("Input 1\tInput 2\tIdeal\tActual");
		for (MLDataPair pair : TRAINING) {
			final MLData output = network.compute(pair.getInput());
			System.out.println(pair.getInput().getData(0) + "\t" + pair.getInput().getData(1) + "\t" + output.getData(0)
					+ "\t" + pair.getIdeal().getData(0));
		}
	}

	public static void main(final String args[]) {
		/*
		 * Experiment the neural network with sigmoid function.
		 */
		experiment("Sigmoid", new ActivationSigmoid(), EPSILON);

		/*
		 * Experiment the neural network with fading sine function.
		 */
		experiment("Fading Sine", new ActivationFadingSin(1), EPSILON);
	}

}
