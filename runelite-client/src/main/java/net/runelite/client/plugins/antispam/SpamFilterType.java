package net.runelite.client.plugins.antispam;

import lombok.RequiredArgsConstructor;

/**
 * Created on 24-Apr-19.
 */
@RequiredArgsConstructor
public enum SpamFilterType
{
	REMOVE_OVERHEAD("Remove overhead"),
	REMOVE_MESSAGE("Remove overhead & chatbox");

	private final String name;

	@Override
	public String toString()
	{
		return name;
	}
}
