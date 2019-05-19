package net.runelite.client.plugins.antispam.neuralnet;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * Created on 24-Apr-19.
 */
public interface Layer
{
	RealMatrix forward_propagate(RealMatrix X);
}
