package net.runelite.client.plugins.antispam.neuralnet;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * Created on 24-Apr-19.
 */
public class Util
{
	public static RealMatrix reverseMatrix(RealMatrix X)
	{
		RealMatrix ret = X.copy();
		for (int i = 1; i <= X.getColumnDimension(); i++)
		{
			ret.setColumnMatrix(i - 1, X.getColumnMatrix(X.getColumnDimension() - i));
		}
		return ret;
	}

	public static RealMatrix concatMatrices(RealMatrix top, RealMatrix bottom)
	{
		RealMatrix ret = MatrixUtils.createRealMatrix(new double[top.getRowDimension() + bottom.getRowDimension()][top.getColumnDimension()]);

		for (int i = 0; i < top.getRowDimension(); i++)
		{
			ret.setRowMatrix(i, top.getRowMatrix(i));
		}
		for (int i = 0; i < bottom.getRowDimension(); i++)
		{
			ret.setRowMatrix(top.getRowDimension() + i, bottom.getRowMatrix(i));
		}

		return ret;
	}

	public static RealMatrix loadMatrixFromInputStream(InputStream in) throws IOException
	{
		BufferedReader bf = new BufferedReader(new InputStreamReader(in));

		ArrayList<ArrayList<Double>> inMatrix = new ArrayList<>();

		String line;
		while ((line = bf.readLine()) != null)
		{
			String[] numbers = line.split(" ");
			ArrayList<Double> tmp = new ArrayList<>();
			for (int i = 0; i < numbers.length; i++)
			{
				tmp.add(Double.parseDouble(numbers[i]));
			}
			inMatrix.add(tmp);
		}

		double[][] matrix = new double[inMatrix.size()][inMatrix.get(0).size()];

		for (int i = 0; i < inMatrix.size(); i++)
		{
			ArrayList<Double> row = inMatrix.get(i);
			for (int j = 0; j < row.size(); j++)
			{
				matrix[i][j] = row.get(j);
			}
		}

		return MatrixUtils.createRealMatrix(matrix);
	}

	public static RealMatrix hardSigmoid(RealMatrix X)
	{
		// y = max(0, min(1, x*0.2 + 0.5))
		RealMatrix ret = X.copy();
		for (int i = 0; i < ret.getRowDimension(); i++)
		{
			for (int j = 0; j < ret.getColumnDimension(); j++)
			{
				double x = ret.getEntry(i, j);
				double y = Math.max(0, Math.min(1, x * 0.2 + 0.5));
				ret.setEntry(i, j, y);
			}
		}
		return ret;
	}

	public static RealMatrix sigmoid(RealMatrix X)
	{
		// y = 1/( 1 + e^-x)
		RealMatrix ret = X.copy();
		for (int i = 0; i < ret.getRowDimension(); i++)
		{
			for (int j = 0; j < ret.getColumnDimension(); j++)
			{
				double x = ret.getEntry(i, j);
				double y = 1 / (1 + Math.exp(-x));
				ret.setEntry(i, j, y);
			}
		}
		return ret;
	}

	public static RealMatrix relu(RealMatrix X)
	{
		RealMatrix ret = X.copy();
		for (int i = 0; i < ret.getRowDimension(); i++)
		{
			for (int j = 0; j < ret.getColumnDimension(); j++)
			{
				double x = ret.getEntry(i, j);
				double y = Math.max(0, x);
				ret.setEntry(i, j, y);
			}
		}
		return ret;
	}

	public static RealMatrix tanh(RealMatrix X)
	{
		RealMatrix ret = X.copy();
		for (int i = 0; i < ret.getRowDimension(); i++)
		{
			for (int j = 0; j < ret.getColumnDimension(); j++)
			{
				double x = ret.getEntry(i, j);
				double y = Math.tanh(x);
				ret.setEntry(i, j, y);
			}
		}
		return ret;
	}

	public static RealMatrix hadamardProduct(RealMatrix A, RealMatrix B)
	{
		RealMatrix ret = A.copy();
		for (int i = 0; i < ret.getRowDimension(); i++)
		{
			for (int j = 0; j < ret.getColumnDimension(); j++)
			{
				ret.setEntry(i, j, A.getEntry(i, j) * B.getEntry(i, j));
			}
		}
		return ret;
	}

	public static double[] padByteSeq(byte[] inSeq, int length)
	{
		double[] ret = new double[length];
		for (int i = 0; i < inSeq.length; i++)
		{
			ret[length - inSeq.length + i] = inSeq[i];
		}

		return ret;
	}
}
