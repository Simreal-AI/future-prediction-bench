"""Deterministic adapter/agent tests; no network calls or paid inference."""

import copy
import json
import socket
import unittest
from unittest.mock import patch

from future_prediction_bench.demo import FixtureClock, FixtureProvider, fixture_questions
from future_prediction_bench.providers import (BraveResearchProvider, ChatCompletionModel, ProviderError,
                                               PublicPageReader, _resolve, strict_json_loads)
from future_prediction_bench.runner import RunLimits, run_questions, runner_config
from future_prediction_bench.store import Store, digest


class ResearchFixture(FixtureProvider):
    def public_config(self):
        return {"adapter": "offline_test"}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def public_config(self):
        return {"adapter": "offline_test", "model": "scripted-test-model"}

    def complete(self, messages, tools, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools, **kwargs}))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def final(probabilities):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps({"probabilities": probabilities})}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40}}


def call(name, arguments, identifier="call_1"):
    return {"choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": identifier, "type": "function", "function": {
                                         "name": name, "arguments": json.dumps(arguments)}}]}}]}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FixtureClock()
        self.store = Store(":memory:", mode="fixture", clock=self.clock)
        self.question = fixture_questions()[0]
        self.question["metadata"] = {"secret_baseline": {"yes": .999}, "private_resolution": "secret answer"}
        self.store.add_question(self.question)

    def tearDown(self):
        self.store.close()

    def run_agent(self, model, **kwargs):
        kwargs.setdefault("reward_mode", "negative_brier")
        return run_questions(self.store, [self.question["question_id"]], model=model,
                             provider=ResearchFixture(), **kwargs)[0]

    def test_default_reward_skips_missing_baseline_before_inference(self):
        report = self.run_agent(ScriptedModel([]), research_mode="no_search", reward_mode="baseline_improvement")
        self.assertEqual(report["status"], "skipped_missing_baseline")
        self.assertEqual(report["model_steps"], 0)
        self.assertEqual(self.store.summary()["groups"], [])

    def test_agent_uses_notebook_calculator_and_draft_then_submits(self):
        class AnalystModel(ScriptedModel):
            def complete(self, messages, tools, **kwargs):
                self.requests.append(copy.deepcopy({"messages": messages, "tools": tools, **kwargs}))
                step = len(self.requests)
                if step == 1:
                    return call("search", {"query": "weather bulletin"})
                if step == 2:
                    source = json.loads(messages[-1]["content"])["results"][0]["sha256"]
                    return call("notebook", {"claim": "Forecast evidence from the bulletin", "source_hashes": [source]})
                if step == 3:
                    return call("calculator", {"expression": "0.5 + 0.2"})
                if step == 4:
                    return call("draft", {"probabilities": {"yes": .7, "no": .3}, "rationale": "Updated estimate", "source_hashes": []})
                return final({"yes": .7, "no": .3})
        model = AnalystModel([])
        report = self.run_agent(model)
        self.assertEqual(report["status"], "pending_reward")
        self.assertEqual(report["research_calls"], 4)
        names = {tool["function"]["name"] for tool in model.requests[0]["tools"]}
        self.assertTrue({"search", "open", "notebook", "calculator", "draft", "submit"} <= names)
        self.assertNotIn("market_snapshot", names)
        events = self.store.events(report["episode_id"])
        self.assertEqual([event["tool"] for event in events if event["type"] == "action"],
                         ["search", "notebook", "calculator", "draft", "submit"])
        self.assertIsNone(self.store.episode(report["episode_id"])["reward"])

    def test_new_tool_arguments_cannot_override_action_and_are_schema_checked(self):
        model = ScriptedModel([call("calculator", {"expression": "2", "unexpected": True}),
                               call("calculator", {"expression": "2", "action": "submit"}),
                               call("search", {"query": "weather"}), final({"yes": .5, "no": .5})])
        report = self.run_agent(model)
        self.assertEqual(report["status"], "pending_reward")
        self.assertEqual(report["research_calls"], 2)
        observation = json.loads(model.requests[1]["messages"][-1]["content"])
        self.assertEqual(observation["reason"], "invalid_tool_arguments")
        events = self.store.events(report["episode_id"])
        self.assertEqual(len([event for event in events if event["tool"] == "submit"]), 1)

    def test_host_budget_submission_is_not_marked_as_policy_generated(self):
        report = self.run_agent(ScriptedModel([call("search", {"query": "weather"})]), max_steps=1)
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(self.store.events(report["episode_id"])[-1]["loss_mask"], 0)

    def test_private_baseline_never_enters_model_messages(self):
        self.store.seal_baseline(self.question["question_id"], {"yes": .98765, "no": .01235},
                                 kind="internal_model", identity="private-baseline-model")
        model = ScriptedModel([final({"yes": .5, "no": .5})])
        report = self.run_agent(model, research_mode="no_search", reward_mode="baseline_improvement")
        self.assertEqual(report["status"], "pending_reward")
        transcript = json.dumps(model.requests)
        self.assertNotIn("private-baseline-model", transcript)
        self.assertNotIn("0.98765", transcript)
        self.assertNotIn("secret_baseline", transcript)

    def test_research_then_submit_and_exact_public_transcript(self):
        model = ScriptedModel([call("search", {"query": "weather bulletin"}), final({"yes": .7, "no": .3})])
        report = self.run_agent(model)
        self.assertEqual(report["status"], "pending_reward")
        self.assertEqual(report["research_calls"], 1)
        turns = self.store.model_turns(report["episode_id"])
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[1]["request"]["messages"], model.requests[1]["messages"])
        self.assertEqual(model.requests[1]["messages"][-1]["role"], "tool")
        self.assertNotIn("secret_baseline", json.dumps(model.requests))
        self.assertNotIn("secret answer", json.dumps(turns))
        self.assertEqual(report["usage"]["completion_tokens"], 15)
        self.assertIsNone(self.store.episode(report["episode_id"])["reward"])

    def test_categorical_submission_outputs_all_option_probabilities(self):
        question = fixture_questions()[1]
        self.store.add_question(question)
        probabilities = {item["id"]: 1 / len(question["options"]) for item in question["options"]}
        reports = run_questions(self.store, [question["question_id"]], model=ScriptedModel([final(probabilities)]),
                                research_mode="no_search", reward_mode="negative_brier")
        self.assertEqual(reports[0]["status"], "pending_reward")
        stored = self.store.episode(reports[0]["episode_id"])
        self.assertEqual(set(stored["prediction"]), set(probabilities))

    def test_final_invalid_probabilities_are_not_normalized(self):
        report = self.run_agent(ScriptedModel([final({"yes": .7, "no": .5})]), research_mode="no_search")
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(self.store.episode(report["episode_id"])["raw_response"], {"yes": .7, "no": .5})

    def test_malformed_final_json_is_audited_and_invalid(self):
        response = {"choices": [{"message": {"content": '{"probabilities":{"yes":0.6,"yes":0.5,"no":0.5}}'}}]}
        report = self.run_agent(ScriptedModel([response]), research_mode="no_search")
        self.assertEqual(report["status"], "invalid")
        self.assertIsInstance(self.store.episode(report["episode_id"])["raw_response"], str)

    def test_model_failures_do_not_produce_fake_forecasts_or_retry(self):
        report = self.run_agent(ScriptedModel([ProviderError("secret transport diagnostics")]))
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["episode_status"], "active")
        self.assertIsNone(self.store.episode(report["episode_id"])["prediction"])
        self.assertNotIn("secret transport", json.dumps(self.store.model_turns(report["episode_id"])))
        again = self.run_agent(ScriptedModel([]))
        self.assertEqual(again["status"], "skipped")
        self.assertEqual(again["model_steps"], 0)

    def test_missing_search_is_invalid(self):
        report = self.run_agent(ScriptedModel([final({"yes": .5, "no": .5})]))
        self.assertEqual(report["status"], "invalid")

    def test_no_search_exposes_only_submit_tool(self):
        model = ScriptedModel([call("submit", {"probabilities": {"yes": .7, "no": .3}})])
        report = self.run_agent(model, research_mode="no_search")
        self.assertEqual(report["status"], "pending_reward")
        self.assertEqual([tool["function"]["name"] for tool in model.requests[0]["tools"]], ["submit"])

    def test_output_budget_reserves_requested_allowance_even_without_usage(self):
        model = ScriptedModel([call("search", {"query": "weather bulletin"})] * 3)
        report = self.run_agent(model, max_tokens_per_step=100, max_output_tokens=150)
        self.assertEqual(report["model_steps"], 2)
        self.assertEqual([request["max_tokens"] for request in model.requests], [100, 50])
        self.assertEqual(report["requested_output_tokens"], 150)
        self.assertEqual(report["reason"], "output_token_budget_exhausted")
        self.assertEqual(report["status"], "invalid")

    def test_tool_budget_is_enforced_and_visible_to_model(self):
        model = ScriptedModel([call("search", {"query": "one"}), call("search", {"query": "two"}),
                               final({"yes": .6, "no": .4})])
        report = self.run_agent(model, max_calls=1)
        self.assertEqual(report["research_calls"], 1)
        self.assertIn("call_budget_exhausted", model.requests[2]["messages"][-1]["content"])
        self.assertEqual(report["status"], "pending_reward")

    def test_invalid_tool_arguments_do_not_execute_arbitrary_actions(self):
        model = ScriptedModel([call("shell", {"command": "write secrets"}),
                               final({"yes": .5, "no": .5})])
        report = self.run_agent(model, research_mode="no_search")
        self.assertEqual(report["status"], "pending_reward")
        self.assertEqual(report["research_calls"], 0)
        self.assertIn("invalid_tool_arguments", model.requests[1]["messages"][-1]["content"])

    def test_prompt_limit_prevents_model_api_call(self):
        report = self.run_agent(ScriptedModel([]), research_mode="no_search", max_prompt_chars=10)
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["model_steps"], 0)

    def test_deadline_crossing_during_model_call_is_not_submitted(self):
        fixture_clock = self.clock
        from datetime import datetime, timezone

        class SlowModel(ScriptedModel):
            def complete(self, *args, **kwargs):
                result = super().complete(*args, **kwargs)
                fixture_clock.value = datetime(2030, 1, 2, 1, tzinfo=timezone.utc)
                return result

        report = self.run_agent(SlowModel([final({"yes": .5, "no": .5})]), research_mode="no_search")
        self.assertEqual(report["status"], "missed")
        self.assertIsNone(self.store.episode(report["episode_id"])["prediction"])
        self.assertEqual(len(self.store.model_turns(report["episode_id"])), 1)

    def test_configs_separate_budgets_and_reward_modes(self):
        model = ScriptedModel([])
        common = {"research_mode": "no_search"}
        a = runner_config(model, None, RunLimits(), **common)
        b = runner_config(model, None, RunLimits(max_calls=9), **common)
        c = runner_config(model, None, RunLimits(), reward_mode="negative_brier", **common)
        self.assertNotEqual(digest(a), digest(b))
        self.assertNotEqual(digest(a), digest(c))

    def test_invalid_limits_rejected_before_assignments(self):
        for kwargs in ({"max_steps": True}, {"max_calls": 0}, {"max_wall_seconds": float("inf")},
                       {"max_output_tokens": -1}, {"max_prompt_chars": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_agent(ScriptedModel([]), **kwargs)
        self.assertEqual(self.store.summary()["groups"], [])

    def test_rl_cannot_use_test_questions(self):
        question = fixture_questions()[2]
        self.store.add_question(question)
        report = run_questions(self.store, [question["question_id"]], model=ScriptedModel([]), track="rl",
                               research_mode="no_search", reward_mode="negative_brier")[0]
        self.assertEqual(report["status"], "skipped")


class ProviderTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys_and_nonfinite(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json_loads(text)

    def test_model_request_has_limits_and_secret_only_in_header(self):
        requests = []
        def request(url, **kwargs):
            requests.append((url, kwargs))
            return final({"yes": .5, "no": .5})
        model = ChatCompletionModel("https://model.example/v1", "secret-test-token", "forecast-model", request_json=request)
        model.complete([{"role": "user", "content": "predict"}], [{"type": "function"}], max_tokens=55, timeout=2)
        self.assertEqual(requests[0][0], "https://model.example/v1/chat/completions")
        self.assertEqual(requests[0][1]["payload"]["max_tokens"], 55)
        self.assertEqual(requests[0][1]["timeout"], 2)
        self.assertNotIn("secret-test-token", json.dumps(model.public_config()))
        self.assertNotIn("secret-test-token", json.dumps(requests[0][1]["payload"]))
        self.assertFalse(requests[0][1]["payload"]["parallel_tool_calls"])

    def test_missing_model_environment_is_an_actionable_configuration_error(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "FPB_MODEL_BASE_URL"):
            ChatCompletionModel.from_env()

    def test_model_rejects_insecure_remote_or_embedded_credentials(self):
        for url in ("http://api.example/v1", "https://user:secret@api.example/v1", "https://api.example/v1?key=secret"):
            with self.subTest(url=url), self.assertRaises((ValueError, ProviderError)):
                ChatCompletionModel(url, "key", "model")
        self.assertEqual(ChatCompletionModel("http://127.0.0.1:8000/v1", "dummy", "model").model_name, "model")

    def test_brave_search_maps_bounded_snippets_without_fabricated_timestamps(self):
        requests = []
        def request(url, **kwargs):
            requests.append((url, kwargs))
            return {"web": {"results": [{"url": "https://example.org/source", "title": "A <b>title</b>",
                                          "description": "Current <strong>evidence</strong>", "age": "2 hours ago",
                                          "extra_snippets": ["More evidence"]}]}}
        provider = BraveResearchProvider("search-secret", request_json=request)
        results = provider.search("forecast evidence")
        self.assertEqual(results[0]["title"], "A title")
        self.assertIn("More evidence", results[0]["text"])
        self.assertNotIn("published_at", results[0])
        self.assertNotIn("search-secret", json.dumps(provider.public_config()))
        self.assertIn("q=forecast+evidence", requests[0][0])

    def test_open_strips_active_content_and_caps_output(self):
        def transport(url, **kwargs):
            return {"status": 200, "headers": {"content-type": "text/html"},
                    "body": b'<title>Evidence</title><script>steal()</script><style>hide</style><p>Observed facts</p>'}
        reader = PublicPageReader(transport=transport, max_text_chars=100)
        page = reader.open("https://example.org/article")
        self.assertEqual(page["title"], "Evidence")
        self.assertIn("Observed facts", page["text"])
        self.assertNotIn("steal", page["text"])
        self.assertNotIn("hide", page["text"])

    def test_open_rejects_nonpublic_urls_before_transport(self):
        def forbidden(*args, **kwargs):
            self.fail("Transport must not be called")
        reader = PublicPageReader(transport=forbidden)
        for url in ("file:///etc/passwd", "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/",
                    "https://[::1]/", "http://192.168.1.2/", "https://user:password@example.org/",
                    "https://example.org:22/", "http://224.0.0.1/", "https://[ff02::1]/", "http://[64:ff9b::7f00:1]/", "https://%31%32%37.0.0.1/"):
            with self.subTest(url=url), self.assertRaises(ProviderError):
                reader.open(url)

    def test_redirect_to_private_address_is_blocked(self):
        requests = []
        def redirect(url, **kwargs):
            requests.append(url)
            return {"status": 302, "headers": {"location": "http://127.0.0.1/admin"}, "body": b""}
        with self.assertRaises(ProviderError):
            PublicPageReader(transport=redirect).open("https://example.org/source")
        self.assertEqual(len(requests), 1)

    def test_public_redirect_final_url_is_preserved_and_bounded(self):
        def redirect(url, **kwargs):
            if url.endswith("first"):
                return {"status": 302, "headers": {"location": "/last"}, "body": b""}
            return {"status": 200, "headers": {"content-type": "text/plain"}, "body": b"facts"}
        result = PublicPageReader(transport=redirect).open("https://example.org/first")
        self.assertEqual(result["url"], "https://example.org/last")
        with self.assertRaises(ProviderError):
            PublicPageReader(max_redirects=0, transport=redirect).open("https://example.org/first")

    def test_dns_rejects_mixed_public_private_answers(self):
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ("8.8.8.8", "127.0.0.1")]
        with patch("future_prediction_bench.providers.socket.getaddrinfo", return_value=answers), self.assertRaises(ProviderError):
            _resolve("example.org", 443, 1, True)

    def test_dns_pins_result_and_avoids_second_resolution(self):
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
        with patch("future_prediction_bench.providers.socket.getaddrinfo", return_value=answers) as resolve:
            self.assertEqual(_resolve("example.org", 443, 1, True), answers[0])
            self.assertEqual(resolve.call_count, 1)

    def test_unsupported_documents_and_oversized_bodies_are_rejected(self):
        for mime, data in (("application/pdf", b"document"), ("text/plain", b"x" * 11)):
            def response(url, **kwargs):
                return {"status": 200, "headers": {"content-type": mime}, "body": data}
            with self.subTest(mime=mime), self.assertRaises(ProviderError):
                PublicPageReader(max_bytes=10, transport=response).open("https://example.org/source")


if __name__ == "__main__":
    unittest.main()
