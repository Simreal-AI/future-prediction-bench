"""Agent interaction terminates at submission; outcome reward arrives separately."""

from .research import ResearchSession
from .analyst import RESEARCH_TOOLS, TOOLKIT_VERSION, tool_definitions
from .schema import public_question


class PredictionEnv:
    """A small dict-action environment, not a Gymnasium or verl integration."""

    def __init__(self, store, provider, *, max_calls=8, market_provider=None):
        self.store = store
        self.provider = provider
        self.max_calls = max_calls
        self.market_provider = market_provider
        self.episode_id = None
        self.session = None

    def reset(self, question_id, policy_id, *, track="benchmark", market_mode="no_consensus", research_mode="self_research", reward_mode="negative_brier"):
        if self.episode_id and self.store.episode(self.episode_id)["status"] == "active":
            raise ValueError("Finish the active episode before resetting")
        self.episode_id = self.store.create_episode(question_id, policy_id, track=track,
                                                   market_mode=market_mode, research_mode=research_mode, reward_mode=reward_mode)
        question = self.store.question(question_id)
        self.session = ResearchSession(question, self.provider, market_mode=market_mode,
                                       max_calls=self.max_calls, clock=self.store.now, market_provider=self.market_provider)
        # Only explicitly public fields are shown; future metadata may be private.
        return {"question": public_question(question),
                "episode_id": self.episode_id, "market_mode": market_mode,
                "research_mode": research_mode, "max_calls": self.max_calls,
                "toolkit_version": TOOLKIT_VERSION,
                "available_tools": [item["function"]["name"] for item in tool_definitions(question, research_mode, market_mode)],
                "output_contract": {"action": "submit", "probabilities": "every option id mapped to a probability; sum = 1"}}

    def step(self, action, *, agent_generated=True):
        if self.episode_id is None:
            raise ValueError("Call reset first")
        episode = self.store.episode(self.episode_id)
        if episode["status"] != "active":
            raise ValueError("Interaction has already terminated")
        if not isinstance(action, dict):
            raise ValueError("Action must be an object")
        kind = action.get("action")
        if kind == "submit":
            receipt = self.store.submit(self.episode_id, action.get("probabilities"), agent_generated=agent_generated)
            return {"observation": {"status": receipt["status"], "receipt_sha256": receipt["receipt_sha256"], "error": receipt["error"]},
                    "reward": receipt["reward"], "terminated": True,
                    "info": {"episode_id": self.episode_id, "awaiting_outcome": receipt["status"] == "pending_reward"}}
        if kind not in RESEARCH_TOOLS:
            raise ValueError("Unsupported analyst action")
        if not agent_generated:
            raise ValueError("Only submission may be synthesized by the host")
        if episode["research_mode"] == "no_search":
            raise ValueError("This is a no-search ablation")
        count = len(self.session.events)
        observation = self.session.dispatch(kind, {key: value for key, value in action.items() if key != "action"})
        self.store.append_events(self.episode_id, self.session.events[count:])
        expired = self.store.episode(self.episode_id)["status"] == "missed"
        return {"observation": observation, "reward": None, "terminated": expired,
                "info": {"episode_id": self.episode_id, "missed_deadline": expired}}

    def reward_status(self):
        if self.episode_id is None:
            raise ValueError("Call reset first")
        episode = self.store.episode(self.episode_id)
        return {"episode_id": self.episode_id, "status": episode["status"],
                "reward": episode["reward"], "scores": episode["scores"]}
