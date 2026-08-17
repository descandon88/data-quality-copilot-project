"""
Unit tests for retrieval/query_rewrite.py.

The one property that actually matters here is the fail-open contract:
rewrite_query() must never raise and must never return something worse
than the original question when the Groq call fails — a broken rewrite
step should degrade retrieval back to "as if rewriting didn't happen," not
break search entirely. All tests inject a mocked client (rewrite_query's
own client= parameter), so none of this touches a real GROQ_API_KEY or
makes a network call.
"""
from unittest.mock import MagicMock

from retrieval.query_rewrite import rewrite_query


def _client_returning(content):
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = response
    return client


class TestSuccessPath:
    def test_returns_rewritten_query(self):
        client = _client_returning("points_balance negative RULE-004 hard-stop")
        result = rewrite_query("why would loyalty point balances be wrong", client=client)
        assert result == "points_balance negative RULE-004 hard-stop"

    def test_strips_surrounding_whitespace_and_quotes(self):
        client = _client_returning('  "points_balance negative"  ')
        result = rewrite_query("why is the balance wrong", client=client)
        assert result == "points_balance negative"

    def test_passes_the_original_question_into_the_prompt(self):
        client = _client_returning("rewritten")
        rewrite_query("are we violating RULE-001 right now", client=client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        prompt_sent = call_kwargs["messages"][0]["content"]
        assert "are we violating RULE-001 right now" in prompt_sent

    def test_uses_the_configured_rewrite_model(self):
        from retrieval.settings import QUERY_REWRITE_MODEL_NAME

        client = _client_returning("rewritten")
        rewrite_query("some question", client=client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == QUERY_REWRITE_MODEL_NAME


class TestFailsOpen:
    def test_falls_back_to_original_on_empty_response(self):
        client = _client_returning("")
        original = "why would loyalty point balances be wrong"
        assert rewrite_query(original, client=client) == original

    def test_falls_back_to_original_on_whitespace_only_response(self):
        client = _client_returning("   ")
        original = "why would loyalty point balances be wrong"
        assert rewrite_query(original, client=client) == original

    def test_falls_back_to_original_on_client_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("rate_limit_exceeded")
        original = "are we currently violating RULE-002"
        result = rewrite_query(original, client=client)
        assert result == original

    def test_never_raises_even_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        # Should not raise — this is the whole point of the fail-open design.
        rewrite_query("any question", client=client)

    def test_verbose_prints_the_fallback_reason(self, capsys):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("tokens per day")
        rewrite_query("any question", client=client, verbose=True)
        captured = capsys.readouterr()
        assert "falling back to original query" in captured.out
        assert "tokens per day" in captured.out

    def test_silent_by_default_on_fallback(self, capsys):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("boom")
        rewrite_query("any question", client=client)  # verbose defaults False
        captured = capsys.readouterr()
        assert captured.out == ""
