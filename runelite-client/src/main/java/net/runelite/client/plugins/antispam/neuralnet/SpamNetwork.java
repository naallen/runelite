package net.runelite.client.plugins.antispam.neuralnet;

import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;

/**
 * Created on 02-May-19.
 */
public class SpamNetwork
{
	BidirectionalLSTMLayer bidirectionalLSTMLayer;
	DenseLayer dense;
	DenseLayer output;

	public SpamNetwork() throws IOException
	{
		this.bidirectionalLSTMLayer = new BidirectionalLSTMLayer("lstm");
		this.dense = new DenseLayer("dense", DenseLayer.Activation.RELU);
		this.output = new DenseLayer("dense_1", DenseLayer.Activation.SIGMOID);
	}

	public double predict(RealMatrix message, RealMatrix misc)
	{
		RealMatrix X_msg = bidirectionalLSTMLayer.forward_propagate(message);
		RealMatrix X = Util.concatMatrices(X_msg, misc.transpose());
		X = dense.forward_propagate(X);
		X = output.forward_propagate(X);
		return X.getEntry(0,0);
	}
}
