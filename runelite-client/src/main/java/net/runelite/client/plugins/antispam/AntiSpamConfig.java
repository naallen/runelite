package net.runelite.client.plugins.antispam;

import net.runelite.client.config.Config;
import net.runelite.client.config.ConfigGroup;
import net.runelite.client.config.ConfigItem;

/**
 * Created on 22-Apr-19.
 */
@ConfigGroup("antiSpam")
public interface AntiSpamConfig extends Config
{
	@ConfigItem(
		keyName = "userBasedFilter",
		name = "Calculate user spamminess",
		description = "Use an alternate algorithm to also calculate spamminess by user, might improve results.",
		position = 1
	)
	default boolean userBasedFilter()
	{
		return true;
	}

	@ConfigItem(
		keyName = "spamThreshold",
		name = "Spam Threshold",
		description = "Sets the sensitivity for blocking spam, lower values will block more spam, but might also block non-spam messages.",
		position = 3
	)
	default int spamThreshold()
	{
		return 50;
	}

	@ConfigItem(
		keyName = "filterType",
		name = "Filter type",
		description = "Configures how spam messages are filtered",
		position = 4
	)
	default SpamFilterType filterType()
	{
		return SpamFilterType.REMOVE_OVERHEAD;
	}
}

