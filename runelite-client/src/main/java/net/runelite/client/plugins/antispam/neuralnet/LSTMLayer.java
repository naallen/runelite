package net.runelite.client.plugins.antispam.neuralnet;

import net.runelite.client.plugins.antispam.AntiSpamPlugin;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;

/**
 * Created on 24-Apr-19.
 */
public class LSTMLayer implements Layer
{

	private int units;

	private RealMatrix W_i, W_f, W_c, W_o;
	private RealMatrix U_i, U_f, U_c, U_o;
	private RealMatrix b_i, b_f, b_c, b_o;

	public LSTMLayer(String layerName) throws IOException
	{
		RealMatrix W = Util.loadMatrixFromInputStream(AntiSpamPlugin.class.getResourceAsStream(layerName + "_params_0.txt"));
		RealMatrix U = Util.loadMatrixFromInputStream(AntiSpamPlugin.class.getResourceAsStream(layerName + "_params_1.txt"));
		RealMatrix b = Util.loadMatrixFromInputStream(AntiSpamPlugin.class.getResourceAsStream(layerName + "_params_2.txt"));

		this.units = W.getColumnDimension() / 4;

		this.W_i = W.getSubMatrix(0, W.getRowDimension() - 1, 0, this.units - 1);
		this.W_f = W.getSubMatrix(0, W.getRowDimension() - 1, this.units, 2 * this.units - 1);
		this.W_c = W.getSubMatrix(0, W.getRowDimension() - 1, 2 * this.units, 3 * this.units - 1);
		this.W_o = W.getSubMatrix(0, W.getRowDimension() - 1, 3 * this.units, 4 * this.units - 1);

		this.U_i = U.getSubMatrix(0, U.getRowDimension() - 1, 0, this.units - 1);
		this.U_f = U.getSubMatrix(0, U.getRowDimension() - 1, this.units, 2 * this.units - 1);
		this.U_c = U.getSubMatrix(0, U.getRowDimension() - 1, 2 * this.units, 3 * this.units - 1);
		this.U_o = U.getSubMatrix(0, U.getRowDimension() - 1, 3 * this.units, 4 * this.units - 1);

		this.b_i = b.getSubMatrix(0, this.units - 1, 0, 0);
		this.b_f = b.getSubMatrix(this.units, 2 * this.units - 1, 0, 0);
		this.b_c = b.getSubMatrix(2 * this.units, 3 * this.units - 1, 0, 0);
		this.b_o = b.getSubMatrix(3 * this.units, 4 * this.units - 1, 0, 0);
	}

	@Override
	public RealMatrix forward_propagate(RealMatrix X)
	{
		RealMatrix c_t_1 = MatrixUtils.createRealMatrix(units, 1);
		RealMatrix h_t_1 = MatrixUtils.createRealMatrix(units, 1);

		// I might have the wrong dimensions on X and x_t, but it's fine for a 1-feature LSTM
		for (int i = 0; i < X.getColumnDimension(); i++) {
			RealMatrix x_t = X.getSubMatrix(0,0, i, i);
			RealMatrix f_t = Util.hardSigmoid(W_f.transpose().multiply(x_t).add(U_f.transpose().multiply(h_t_1)).add(b_f));
			RealMatrix i_t = Util.hardSigmoid(W_i.transpose().multiply(x_t).add(U_i.transpose().multiply(h_t_1)).add(b_i));
			RealMatrix o_t = Util.hardSigmoid(W_o.transpose().multiply(x_t).add(U_o.transpose().multiply(h_t_1)).add(b_o));
			RealMatrix c_t = Util.tanh(W_c.transpose().multiply(x_t).add(U_c.transpose().multiply(h_t_1)).add(b_c));
			c_t = Util.hadamardProduct(f_t, c_t_1).add(Util.hadamardProduct(i_t, c_t));
			RealMatrix h_t = Util.hadamardProduct(o_t, Util.tanh(c_t));
			c_t_1 = c_t;
			h_t_1 = h_t;
		}

		return h_t_1;
	}
}
