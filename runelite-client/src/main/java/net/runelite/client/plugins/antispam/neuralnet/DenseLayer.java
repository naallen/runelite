package net.runelite.client.plugins.antispam.neuralnet;

import net.runelite.client.plugins.antispam.AntiSpamPlugin;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;

/**
 * Created on 24-Apr-19.
 */
public class DenseLayer implements Layer
{

	private RealMatrix W;
	private RealMatrix b;
	private Activation activation;

	public DenseLayer(String layerName, Activation activation) throws IOException
	{
		this.W = Util.loadMatrixFromInputStream(AntiSpamPlugin.class.getResourceAsStream(layerName + "_params_0.txt"));
		this.b = Util.loadMatrixFromInputStream(AntiSpamPlugin.class.getResourceAsStream(layerName + "_params_1.txt"));
		this.activation = activation;
	}

	@Override
	public RealMatrix forward_propagate(RealMatrix X)
	{
		switch (this.activation)
		{
			case SIGMOID:
				return Util.sigmoid(W.transpose().multiply(X).add(b));
			case RELU:
				return Util.relu(W.transpose().multiply(X).add(b));
			case TANH:
				return Util.tanh(W.transpose().multiply(X).add(b));
			default:
				return null;
		}
	}

	public enum Activation
	{
		SIGMOID,
		RELU,
		TANH
	}
}
