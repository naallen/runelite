package net.runelite.client.plugins.antispam.neuralnet;

import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;

/**
 * Created on 27-Apr-19.
 */
public class BidirectionalLSTMLayer implements Layer
{
	private LSTMLayer forwardLayer, backwardLayer;

	public BidirectionalLSTMLayer(String layerName) throws IOException
	{
		this.forwardLayer = new LSTMLayer("forward_" + layerName);
		this.backwardLayer = new LSTMLayer("backward_" + layerName);
	}


	@Override
	public RealMatrix forward_propagate(RealMatrix X)
	{
		RealMatrix fwd = this.forwardLayer.forward_propagate(X);
		RealMatrix bwd = this.backwardLayer.forward_propagate(Util.reverseMatrix(X));
		return Util.concatMatrices(fwd, bwd);
	}
}
