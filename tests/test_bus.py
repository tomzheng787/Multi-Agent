from marketing_crew.messaging.bus import AgentMessage, MessageBus, MessageType


def msg(recipient="writer", **payload):
    return AgentMessage(run_id="r", sender="critic", recipient=recipient, type=MessageType.FEEDBACK, payload=payload)


def test_publish_consume_ack(redis_client):
    bus = MessageBus(redis_client, "bus:r")
    bus.publish(msg(note="tighten KPIs"))
    deliveries = bus.consume("writer", "g", "c1")
    assert [d.message.payload["note"] for d in deliveries] == ["tighten KPIs"]
    bus.ack("writer", "g", deliveries[0].entry_id)
    assert bus.consume("writer", "g", "c1") == []
    assert len(bus.history(MessageBus.EVENTS)) == 1  # mirrored to the firehose


def test_failed_messages_retry_then_dead_letter(redis_client):
    bus = MessageBus(redis_client, "bus:r", max_attempts=2)
    bus.publish(msg(n=1))
    seen = []

    def handler(message):
        seen.append(message.attempt)
        raise RuntimeError("boom")

    first = bus.process("writer", "g", "c", handler, reclaim_idle_ms=None)
    second = bus.process("writer", "g", "c", handler, reclaim_idle_ms=None)
    assert (first.retried, second.dead_lettered) == (1, 1)
    assert seen == [1, 2]
    dead = bus.dead_letters()
    assert len(dead) == 1 and "boom" in dead[0].payload["_error"]
    assert bus.process("writer", "g", "c", handler, reclaim_idle_ms=None).succeeded == 0


def test_reclaims_messages_from_crashed_consumer(redis_client):
    bus = MessageBus(redis_client, "bus:r")
    bus.publish(msg(n=1))
    assert len(bus.consume("writer", "g", "crashed-worker")) == 1  # read, never acked
    handled = []
    stats = bus.process("writer", "g", "rescuer", lambda m: handled.append(m.payload["n"]), reclaim_idle_ms=0)
    assert stats.succeeded == 1 and handled == [1]
