"""Role-based access control for agent tools and delegation.

The policy lives in ``config/rbac.yaml``. Two enforcement points use it:

* build time - :meth:`AccessController.tools_for` decides which tools an agent
  is handed at all (least privilege);
* run time - every guarded tool calls :meth:`AccessController.check` before it
  executes, so a tool that leaks to the wrong agent still cannot run.

Every decision is written to an audit sink.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import yaml


class PermissionDenied(PermissionError):
    def __init__(self, role: str, permission: str, resource: str = ""):
        self.role = role
        self.permission = permission
        self.resource = resource
        target = f" on '{resource}'" if resource else ""
        super().__init__(f"role '{role}' lacks permission '{permission}'{target}")


@dataclass(frozen=True)
class RoleDef:
    name: str
    permissions: frozenset[str]
    inherits: tuple[str, ...] = ()
    can_delegate_to: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class AuditEvent:
    ts: float
    role: str
    permission: str
    resource: str
    allowed: bool


@dataclass
class RBACPolicy:
    roles: dict[str, RoleDef]
    tool_permissions: dict[str, str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RBACPolicy":
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "RBACPolicy":
        roles = {
            name: RoleDef(
                name=name,
                permissions=frozenset(spec.get("permissions", [])),
                inherits=tuple(spec.get("inherits", [])),
                can_delegate_to=tuple(spec.get("can_delegate_to", [])),
                description=(spec.get("description") or "").strip(),
            )
            for name, spec in (data.get("roles") or {}).items()
        }
        policy = cls(roles=roles, tool_permissions=dict(data.get("tools") or {}))
        policy.validate()
        return policy

    def validate(self) -> None:
        for role in self.roles.values():
            for parent in role.inherits + role.can_delegate_to:
                if parent not in self.roles:
                    raise ValueError(f"role '{role.name}' references unknown role '{parent}'")
        # Reject inheritance cycles up front instead of recursing forever later.
        for name in self.roles:
            self._resolve(name, stack=())

    def _resolve(self, name: str, stack: tuple[str, ...]) -> frozenset[str]:
        if name in stack:
            raise ValueError(f"inheritance cycle: {' -> '.join(stack + (name,))}")
        role = self.roles[name]
        perms = set(role.permissions)
        for parent in role.inherits:
            perms |= self._resolve(parent, stack + (name,))
        return frozenset(perms)


@dataclass
class AccessController:
    policy: RBACPolicy
    audit_sink: Callable[[AuditEvent], None] | None = None
    _cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    _log: list[AuditEvent] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> "AccessController":
        return cls(RBACPolicy.from_yaml(path), **kwargs)

    def permissions_for(self, role: str) -> frozenset[str]:
        if role not in self.policy.roles:
            return frozenset()
        if role not in self._cache:
            self._cache[role] = self.policy._resolve(role, stack=())
        return self._cache[role]

    def is_allowed(self, role: str, permission: str) -> bool:
        return any(fnmatch.fnmatchcase(permission, granted) for granted in self.permissions_for(role))

    def check(self, role: str, permission: str, *, resource: str = "") -> None:
        allowed = self.is_allowed(role, permission)
        self._record(AuditEvent(time.time(), role, permission, resource, allowed))
        if not allowed:
            raise PermissionDenied(role, permission, resource)

    def required_permission(self, tool_name: str) -> str:
        try:
            return self.policy.tool_permissions[tool_name]
        except KeyError:
            # Unknown tools are denied by default rather than silently allowed.
            raise PermissionDenied("*", f"tool:{tool_name}", "unregistered tool") from None

    def check_tool(self, role: str, tool_name: str) -> None:
        self.check(role, self.required_permission(tool_name), resource=tool_name)

    def tools_for(self, role: str, available: Iterable[str]) -> list[str]:
        return [
            name
            for name in available
            if name in self.policy.tool_permissions
            and self.is_allowed(role, self.policy.tool_permissions[name])
        ]

    def can_delegate(self, from_role: str, to_role: str) -> bool:
        role = self.policy.roles.get(from_role)
        return bool(role) and self.is_allowed(from_role, "delegate") and to_role in role.can_delegate_to

    def check_delegation(self, from_role: str, to_role: str) -> None:
        allowed = self.can_delegate(from_role, to_role)
        self._record(AuditEvent(time.time(), from_role, "delegate", to_role, allowed))
        if not allowed:
            raise PermissionDenied(from_role, "delegate", to_role)

    @property
    def audit_log(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._log)

    def _record(self, event: AuditEvent) -> None:
        with self._lock:
            self._log.append(event)
        if self.audit_sink is not None:
            try:
                self.audit_sink(event)
            except Exception:  # auditing must never break the request path
                pass
