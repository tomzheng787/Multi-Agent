import fakeredis
import pytest

from marketing_crew.memory.store import EvidenceStore
from marketing_crew.settings import Settings


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env(offline=True, redis_url="fakeredis://", output_dir=tmp_path / "outputs")


@pytest.fixture
def evidence(redis_client):
    return EvidenceStore(redis_client, "test-run")
