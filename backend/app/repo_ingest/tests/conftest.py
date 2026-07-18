import pytest

from app.repo_ingest.tests.fakes import FakeRedis


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()
