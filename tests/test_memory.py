import redis
from langchain_core.messages import AIMessage, HumanMessage

from marketing_crew.memory.store import (
    CheckpointStore,
    MemoryManager,
    RedisChatMessageHistory,
    connect_redis,
)


def test_history_roundtrip_ttl_and_trim(redis_client):
    history = RedisChatMessageHistory(redis_client, "s1", ttl_seconds=60, max_messages=3)
    history.add_messages([HumanMessage("a"), AIMessage("b"), HumanMessage("c"), AIMessage("d")])
    assert [m.content for m in history.messages] == ["b", "c", "d"]
    assert isinstance(history.messages[0], AIMessage)
    assert 0 < redis_client.ttl(history.key) <= 60


def test_history_survives_new_instance(redis_client):
    RedisChatMessageHistory(redis_client, "s2").add_user_message("remember me")
    assert RedisChatMessageHistory(redis_client, "s2").messages[0].content == "remember me"


class FlakyRedis:
    """Wraps a client and fails the next N pipeline executions."""

    def __init__(self, inner, failures):
        self.inner, self.failures = inner, failures

    def pipeline(self):
        pipe = self.inner.pipeline()
        outer = self

        class Pipe:
            def __getattr__(self, name):
                return getattr(pipe, name)

            def execute(self):
                if outer.failures > 0:
                    outer.failures -= 1
                    raise redis.exceptions.ConnectionError("down")
                return pipe.execute()

        return Pipe()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_writes_are_buffered_while_redis_is_down(redis_client):
    flaky = FlakyRedis(redis_client, failures=3)  # outlasts the 3 retries
    history = RedisChatMessageHistory(flaky, "s3")
    history.add_user_message("queued while down")
    assert history.pending_writes == 1
    # Redis is back: the next access flushes the buffer, nothing was lost.
    assert [m.content for m in history.messages] == ["queued while down"]
    assert history.pending_writes == 0
    assert redis_client.llen(history.key) == 1


def test_connect_falls_back_when_redis_unreachable():
    client, durable = connect_redis("redis://127.0.0.1:1/0", allow_fallback=True)
    assert durable is False
    client.set("k", "v")
    assert client.get("k") == "v"


def test_recall_combines_session_and_agent_memory(redis_client):
    memory = MemoryManager(redis_client, ttl_seconds=60, max_messages=50)
    memory.session("s").add_user_message("brief for Q1")
    memory.agent("s", "writer").add_ai_message("draft v1")
    recalled = memory.recall("s", "writer")
    assert "brief for Q1" in recalled and "draft v1" in recalled
    assert memory.recall("empty") == "No prior memory for this session."


def test_evidence_ids_dedupe_and_search(evidence):
    first = evidence.add(source="geo", content="Mission District coordinates 37.75")
    again = evidence.add(source="geo", content="Mission District coordinates 37.75")
    other = evidence.add(source="wiki", content="Cold brew steeped 12 to 24 hours")
    assert (first.id, again.id, other.id) == ("E1", "E1", "E2")
    assert evidence.exists("e2") and not evidence.exists("E9")
    assert evidence.search("cold brew hours")[0].id == "E2"


def test_checkpoints(redis_client):
    store = CheckpointStore(redis_client, "r1")
    store.save("market_research", "brief", owner="researcher")
    store.set_meta(status="running")
    assert store.completed() == {"market_research": "brief"}
    assert store.meta()["status"] == "running"
