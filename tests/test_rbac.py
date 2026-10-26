import pytest

from marketing_crew.settings import CONFIG_DIR
from marketing_crew.security.rbac import AccessController, PermissionDenied, RBACPolicy


@pytest.fixture
def access():
    return AccessController.from_yaml(CONFIG_DIR / "rbac.yaml")


def test_inherited_and_direct_permissions(access):
    assert access.is_allowed("researcher", "search:local")
    assert access.is_allowed("researcher", "bus:publish")  # inherited from base
    assert not access.is_allowed("writer", "artifact:write")


def test_wildcard_permissions():
    policy = RBACPolicy.from_dict({"roles": {"auditor": {"permissions": ["evidence:*"]}}, "tools": {}})
    access = AccessController(policy)
    assert access.is_allowed("auditor", "evidence:read")
    assert not access.is_allowed("auditor", "artifact:write")


def test_tools_for_role_is_least_privilege(access):
    requested = ["local_market_search", "save_artifact", "evidence_lookup"]
    assert access.tools_for("researcher", requested) == ["local_market_search", "evidence_lookup"]
    assert access.tools_for("executor", requested) == ["save_artifact", "evidence_lookup"]


def test_check_tool_denies_and_audits(access):
    with pytest.raises(PermissionDenied):
        access.check_tool("writer", "schedule_campaign")
    access.check_tool("executor", "schedule_campaign")
    decisions = [(e.role, e.resource, e.allowed) for e in access.audit_log]
    assert decisions == [("writer", "schedule_campaign", False), ("executor", "schedule_campaign", True)]


def test_unregistered_tools_are_denied(access):
    with pytest.raises(PermissionDenied):
        access.check_tool("executor", "drop_database")


def test_delegation_policy(access):
    assert access.can_delegate("manager", "writer")
    assert not access.can_delegate("writer", "executor")
    assert not access.can_delegate("manager", "manager")
    with pytest.raises(PermissionDenied):
        access.check_delegation("critic", "executor")


def test_manager_has_no_tool_permissions(access):
    assert access.tools_for("manager", list(access.policy.tool_permissions)) == []


def test_inheritance_cycle_rejected():
    with pytest.raises(ValueError, match="cycle"):
        RBACPolicy.from_dict({"roles": {"a": {"inherits": ["b"]}, "b": {"inherits": ["a"]}}})
