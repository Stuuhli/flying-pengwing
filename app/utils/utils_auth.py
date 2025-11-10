"""Utility helpers for user authentication and authorization."""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import password_verify
from app.db import (
    Collection,
    Permission,
    User,
    Workspace,
)
from app.db.database import get_db
from app.utils.utils_logging import initialize_logging, logger
from app.config import BACKEND_FASTAPI_LOG

initialize_logging(BACKEND_FASTAPI_LOG)


class CollectionProfile(BaseModel):
    id: int
    name: str
    description: str | None = None
    document_count: int


class WorkspaceProfile(BaseModel):
    id: int
    name: str
    description: str | None = None
    collections: list[CollectionProfile] = Field(default_factory=list)


class UserProfile(BaseModel):
    id: int
    email: EmailStr
    is_admin: bool
    permissions: list[str]
    workspaces: list[WorkspaceProfile] = Field(default_factory=list)


class user_auth_format(BaseModel):
    """Request payload for creating a new user account."""

    email: EmailStr
    password: str = Field(min_length=6)
    is_admin: bool = False
    rag_type: str = Field(pattern=r"^(rag|graphrag)$")
    workspace_ids: list[int] = Field(default_factory=list)


class user_auth_validate(BaseModel):
    """Payload for validating user credentials and issuing a JWT."""

    email: EmailStr
    password: str


def create_user_record(
    db: Session,
    *,
    email: str,
    hashed_password: str,
    is_admin: bool,
    rag_type: str,
    workspace_ids: Iterable[int] | None = None,
) -> User:
    """Persist a new user together with permission and workspace mappings."""

    normalized_rag = rag_type.lower()
    if normalized_rag not in {"rag", "graphrag"}:
        raise ValueError("Invalid rag_type. Allowed values are 'rag' or 'graphrag'.")

    existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if existing:
        raise ValueError("User with this e-mail already exists.")

    user = User(email=email, hashed_password=hashed_password, is_admin=is_admin)
    db.add(user)
    db.flush()

    permission_names = ["admin" if is_admin else "user", normalized_rag]
    permissions = (
        db.execute(select(Permission).where(Permission.name.in_(permission_names))).scalars().all()
    )
    if len(permissions) != len(set(permission_names)):
        raise ValueError("Required permissions are not configured in the database.")
    user.permissions.extend(permissions)

    workspace_ids = list(workspace_ids or [])
    if workspace_ids:
        workspaces = (
            db.execute(select(Workspace).where(Workspace.id.in_(workspace_ids))).scalars().all()
        )
        if len(workspaces) != len(set(workspace_ids)):
            raise ValueError("One or more workspaces are invalid.")
        user.workspaces.extend(workspaces)

    db.commit()
    db.refresh(user)
    logger.info("User %s created with %s permissions", email, ", ".join(permission_names))
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    """Retrieve a user by e-mail address."""

    return db.execute(select(User).where(User.email == email)).scalar_one_or_none()


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """Validate credentials and return the matching user if valid."""

    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not password_verify(password=password, hashed=user.hashed_password):
        return None
    return user


def build_user_profile(user: User) -> UserProfile:
    """Convert a user ORM object into a serialisable profile."""

    workspaces: list[WorkspaceProfile] = []
    for workspace in sorted(user.workspaces, key=lambda ws: ws.name.lower()):
        collections = [
            CollectionProfile(
                id=collection.id,
                name=collection.name,
                description=collection.description,
                document_count=collection.document_count,
            )
            for collection in sorted(workspace.collections, key=lambda col: col.name.lower())
        ]
        workspaces.append(
            WorkspaceProfile(
                id=workspace.id,
                name=workspace.name,
                description=workspace.description,
                collections=collections,
            )
        )

    permissions = sorted({permission.name for permission in user.permissions})
    return UserProfile(
        id=user.id,
        email=user.email,
        is_admin=user.is_admin,
        permissions=permissions,
        workspaces=workspaces,
    )


__all__ = [
    "user_auth_format",
    "user_auth_validate",
    "UserProfile",
    "WorkspaceProfile",
    "CollectionProfile",
    "create_user_record",
    "get_user_by_email",
    "authenticate_user",
    "build_user_profile",
    "get_db",
]
