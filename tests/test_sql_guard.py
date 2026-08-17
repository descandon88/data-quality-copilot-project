"""
Unit tests for agent/tools.py's query_warehouse() SQL guard — the
regex-based denylist that's the only thing standing between the agent's
query_warehouse tool and arbitrary SQL against the warehouse (see
agent/tools.py's own docstring: "a demo-appropriate guard, not a production
one"). Worth testing precisely because it's a security-relevant boundary
the agent's own generated SQL has to pass through, not just an internal
implementation detail.

All reject-path tests run with no Postgres connection at all — a rejected
query returns before agent.tools ever calls get_connection(). The
accept-path tests mock get_connection so they don't need a live database
either; this file never touches real Postgres.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.tools import query_warehouse


class TestRejectsNonSelect:
    @pytest.mark.parametrize("sql", [
        "DROP TABLE bronze.loyalty_accounts",
        "DELETE FROM bronze.loyalty_transactions",
        "UPDATE bronze.loyalty_accounts SET points_balance = 0",
        "INSERT INTO bronze.loyalty_accounts VALUES (1, 2, 3)",
        "TRUNCATE bronze.loyalty_transactions",
        "ALTER TABLE bronze.loyalty_accounts ADD COLUMN x int",
        "",
        "   ",
        "not sql at all",
    ])
    def test_rejects_non_select_statement(self, sql):
        result = query_warehouse(sql)
        assert result == "Rejected: only SELECT statements are allowed."

    def test_select_prefix_is_case_insensitive_and_whitespace_tolerant(self):
        # A SELECT that WOULD pass the prefix check should not be rejected
        # at this stage (it may still be rejected by the keyword denylist,
        # or would proceed to a real DB call — covered by other tests).
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [("loyalty_id",)]
            mock_cursor.fetchall.return_value = [(1,)]
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            result = query_warehouse("   select   loyalty_id from bronze.loyalty_accounts")
            assert result != "Rejected: only SELECT statements are allowed."


class TestRejectsForbiddenKeywords:
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM bronze.loyalty_accounts; DROP TABLE bronze.loyalty_accounts",
        "SELECT * FROM bronze.loyalty_accounts WHERE 1=1; DELETE FROM bronze.loyalty_accounts",
        "SELECT * FROM bronze.loyalty_accounts UNION SELECT * FROM information_schema.tables; GRANT ALL ON bronze.loyalty_accounts TO public",
    ])
    def test_rejects_select_with_forbidden_keyword_smuggled_in(self, sql):
        result = query_warehouse(sql)
        assert result == ("Rejected: query contains a disallowed keyword "
                           "(only read-only SELECTs are permitted).")

    def test_known_gap_select_into_is_not_caught_by_the_denylist(self):
        # "SELECT ... INTO new_table" is a real Postgres footgun (it
        # creates a table) that this regex-based guard does NOT catch —
        # none of insert/update/delete/drop/alter/truncate/grant/revoke/
        # create/... appear in "SELECT * INTO new_table FROM ...". Asserted
        # here (with the DB call mocked, since this SQL is NOT rejected and
        # falls through to a real query) so this is a documented, known gap
        # instead of a silent assumption — see agent/tools.py's own
        # docstring note that this guard is "a demo-appropriate guard, not
        # a production one."
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [("loyalty_id",)]
            mock_cursor.fetchall.return_value = [(1,)]
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            result = query_warehouse("SELECT * INTO new_table FROM bronze.loyalty_accounts")
            assert result != "Rejected: only SELECT statements are allowed."
            assert result != ("Rejected: query contains a disallowed keyword "
                               "(only read-only SELECTs are permitted).")

    def test_forbidden_keyword_matches_whole_words_only(self):
        # "dropped_at" contains "drop" as a substring but is a column name,
        # not the DROP keyword — the guard uses \b word boundaries
        # specifically so this is NOT a false-positive rejection.
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [("dropped_at",)]
            mock_cursor.fetchall.return_value = []
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            result = query_warehouse("SELECT dropped_at FROM bronze.loyalty_accounts")
            assert result != ("Rejected: query contains a disallowed keyword "
                               "(only read-only SELECTs are permitted).")


class TestLimitInjection:
    def _run_with_mock(self, sql):
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [("loyalty_id",)]
            mock_cursor.fetchall.return_value = [(1,)]
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            query_warehouse(sql)
            executed_sql = mock_cursor.execute.call_args[0][0]
            return executed_sql

    def test_appends_limit_when_missing(self):
        executed = self._run_with_mock("SELECT loyalty_id FROM bronze.loyalty_accounts")
        assert "LIMIT 50" in executed

    def test_does_not_append_limit_when_already_present(self):
        executed = self._run_with_mock(
            "SELECT loyalty_id FROM bronze.loyalty_accounts LIMIT 10"
        )
        assert executed.count("LIMIT") == 1
        assert "LIMIT 10" in executed

    def test_strips_trailing_semicolon_before_appending_limit(self):
        executed = self._run_with_mock("SELECT loyalty_id FROM bronze.loyalty_accounts;")
        assert executed == "SELECT loyalty_id FROM bronze.loyalty_accounts LIMIT 50"


class TestResultFormatting:
    def test_no_rows_returns_explicit_message(self):
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [("loyalty_id",)]
            mock_cursor.fetchall.return_value = []
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            result = query_warehouse("SELECT loyalty_id FROM bronze.loyalty_accounts")
            assert result == "Query returned no rows."

    def test_formats_header_and_rows(self):
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [("loyalty_id",), ("points_balance",)]
            mock_cursor.fetchall.return_value = [(1, 100), (2, -5)]
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            result = query_warehouse("SELECT loyalty_id, points_balance FROM bronze.loyalty_accounts")
            lines = result.split("\n")
            assert lines[0] == "loyalty_id | points_balance"
            assert "1 | 100" in lines
            assert "2 | -5" in lines

    def test_query_failure_is_caught_and_reported_not_raised(self):
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = Exception("relation does not exist")
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            result = query_warehouse("SELECT * FROM bronze.nonexistent_table")
            assert result.startswith("Query failed:")
            assert "relation does not exist" in result

    def test_connection_is_always_closed(self):
        with patch("agent.tools.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = Exception("boom")
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            query_warehouse("SELECT * FROM bronze.loyalty_accounts")
            mock_conn.close.assert_called_once()
