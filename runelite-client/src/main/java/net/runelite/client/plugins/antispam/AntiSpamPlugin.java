package net.runelite.client.plugins.antispam;

import com.google.common.collect.EvictingQueue;
import com.google.inject.Provides;
import lombok.extern.slf4j.Slf4j;
import net.runelite.api.ChatMessageType;
import net.runelite.api.Client;
import net.runelite.api.GameState;
import net.runelite.api.Player;
import net.runelite.api.events.ChatMessage;
import net.runelite.api.events.GameStateChanged;
import net.runelite.api.events.OverheadTextChanged;
import net.runelite.client.config.ConfigManager;
import net.runelite.client.eventbus.Subscribe;
import net.runelite.client.plugins.Plugin;
import net.runelite.client.plugins.PluginDescriptor;
import net.runelite.client.plugins.antispam.neuralnet.SpamNetwork;
import net.runelite.client.plugins.antispam.neuralnet.Util;
import net.runelite.client.ui.JagexColors;
import net.runelite.client.util.ColorUtil;
import net.runelite.client.util.Text;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.text.similarity.LevenshteinDistance;

import javax.inject.Inject;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ScheduledExecutorService;

/**
 * Created on 22-Apr-19.
 */
@PluginDescriptor(
	name = "Anti Spam",
	description = "Anti Spam",
	tags = {"anti", "spam"}
)
@Slf4j
public class AntiSpamPlugin extends Plugin
{
	static private final int MAX_MESSAGE_LEN = 80;
	static private final int LOOKBACK = 10;

	@Inject
	private AntiSpamConfig config;

	@Inject
	private ScheduledExecutorService executor;

	@Inject
	private Client client;

	private SpamNetwork spamNetwork;
	private HashMap<String, Double> userSpamminess = new HashMap<>();
	private HashMap<String, Integer> userNumMessages = new HashMap<>();

	private HashMap<String, EvictingQueue<Double>> timeQueues = new HashMap<>();
	private HashMap<String, EvictingQueue<Double>> lenQueues = new HashMap<>();
	private HashMap<String, String> lastMessages = new HashMap<>();
	private HashMap<String, Integer> lastMsgTick = new HashMap<>();

	private int lastLoginTick;

	@Provides
	AntiSpamConfig getConfig(ConfigManager configManager)
	{
		return configManager.getConfig(AntiSpamConfig.class);
	}

	@Subscribe
	public void onChatMessage(ChatMessage event)
	{
		if (event.getType() == ChatMessageType.PUBLICCHAT)
		{
			EvictingQueue<Double> timeSinceLastQueue = timeQueues.getOrDefault(event.getName(), EvictingQueue.create(LOOKBACK));
			EvictingQueue<Double> lenQueue = lenQueues.getOrDefault(event.getName(), EvictingQueue.create(LOOKBACK));

			if (timeSinceLastQueue.peek() == null)
			{
				timeSinceLastQueue.add((double) (client.getTickCount() - lastLoginTick));
			}
			else
			{
				timeSinceLastQueue.add((double) (client.getTickCount() - lastMsgTick.get(event.getName())));
			}
			lenQueue.add((double) event.getMessage().length());

			timeQueues.put(event.getName(), timeSinceLastQueue);
			lenQueues.put(event.getName(), lenQueue);

			RealMatrix miscFeatures = generateMiscFeatures(event);

			executor.submit(() -> processChatMessage(event, miscFeatures));

			lastMessages.put(event.getName(), event.getMessage());
			lastMsgTick.put(event.getName(), client.getTickCount());
		}
	}

	@Override
	protected void startUp() throws Exception
	{
		spamNetwork = new SpamNetwork();
	}

	@Subscribe
	public void onGameStateChanged(GameStateChanged gameStateChanged)
	{
		if (gameStateChanged.getGameState() == GameState.LOGGED_IN)
		{
			this.timeQueues.clear();
			this.lastMessages.clear();
			this.lastMsgTick.clear();
			this.lastLoginTick = client.getTickCount();
		}
	}

	private void processChatMessage(ChatMessage event, RealMatrix miscFeatures)
	{
		if (isSpam(event.getName(), event.getMessage(), miscFeatures))
		{
			event.getMessageNode().setName(event.getName() + ColorUtil.wrapWithColorTag(" (SPAM)", JagexColors.CHAT_CLAN_TEXT_TRANSPARENT_BACKGROUND));
			client.refreshChat();
		}
	}

	private RealMatrix generateMiscFeatures(ChatMessage event)
	{
		String username = event.getName();
		String message = event.getMessage();

		EvictingQueue<Double> timeSinceLastQueue = timeQueues.get(username);
		double[] timesSinceLast = timeSinceLastQueue.stream().mapToDouble(Double::doubleValue).toArray();

		double timeSinceLast = timesSinceLast[timesSinceLast.length - 1];
		double meanTimeSinceLast = StatUtils.mean(timesSinceLast);
		double stdTimeSinceLast = Math.sqrt(StatUtils.populationVariance(timesSinceLast));
		double msgcount = timesSinceLast.length;

		EvictingQueue<Double> lenQueue = lenQueues.get(username);
		double[] lens = lenQueue.stream().mapToDouble(Double::doubleValue).toArray();

		double len = message.length();
		double meanLen = StatUtils.mean(lens);
		double stdLen = Math.sqrt(StatUtils.populationVariance(lens));

		double lvDist = MAX_MESSAGE_LEN;
		if (lastMessages.containsKey(username))
		{
			lvDist = LevenshteinDistance.getDefaultInstance().apply(lastMessages.get(username), message);
		}

		log.debug(Arrays.toString(timesSinceLast));
		log.debug("Mean time since last: " + meanTimeSinceLast);
		log.debug("Std dev time since last: " + stdTimeSinceLast);
		log.debug("Mean len: " + meanLen);
		log.debug("Std dev len: " + stdLen);
		log.debug("LV dist: " + lvDist);
		log.debug("Time since last: " + timeSinceLast);
		log.debug("Msg count: "+ msgcount);
		log.debug("Length: " + len);

		double[][] miscMatrix = {{meanTimeSinceLast, stdTimeSinceLast, meanLen, stdLen, lvDist, timeSinceLast, msgcount, len}};
		return MatrixUtils.createRealMatrix(miscMatrix);
	}

	private boolean isSpam(String username, String message, RealMatrix miscFeatures)
	{
		String cleanMessage = cleanMessage(message);
		double score = spamNetwork.predict(messageToMatrix(cleanMessage), miscFeatures);

		double priorUserSpamminess = userSpamminess.getOrDefault(username, ((double) config.spamThreshold() / 100.0));
		int priorUserNumMessages = userNumMessages.getOrDefault(username, 1);

		double threshold;
		if (config.userBasedFilter())
		{
			threshold = priorUserSpamminess;
		}
		else
		{
			threshold = ((double) config.spamThreshold() / 100.0);
		}
		userSpamminess.put(username, (threshold * priorUserNumMessages + score) / (priorUserNumMessages + 1));
		userNumMessages.put(username, priorUserNumMessages + 1);

		log.debug(username + ":" + cleanMessage(message) + " score: " + score);

		return score > threshold;
	}

	private String cleanMessage(String msg)
	{
		String convertedString = Normalizer.normalize(Text.removeTags(msg), Normalizer.Form.NFKD);
		convertedString = convertedString.replaceAll("[^\\p{ASCII}]", "");
		return convertedString;
	}

	private RealMatrix messageToMatrix(String msg)
	{
		double[] inSeq = Util.padByteSeq(msg.getBytes(StandardCharsets.US_ASCII), MAX_MESSAGE_LEN);
		double[][] x = new double[][]{inSeq};
		return MatrixUtils.createRealMatrix(x);
	}

	/*private void processOverheadMessage(OverheadTextChanged event)
	{
		if (!(event.getActor() instanceof Player))
		{
			return;
		}
		String originalMessage = event.getOverheadText();
		// Hides the chat message until it is verified to not be spam
		event.getActor().setOverheadText("");
		if (!isSpam(event.getActor().getName(), originalMessage))
		{
			event.getActor().setOverheadText(originalMessage);
		}
	}*/

	/*@Subscribe
	public void onOverheadTextChanged(OverheadTextChanged event)
	{
		executor.submit(() -> processOverheadMessage(event));
	}*/
}
