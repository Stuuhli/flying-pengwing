"""SQLAlchemy models for authentication, workspaces and collections."""

from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class User(Base):
    """Application user with role and workspace assignments."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)

    permissions: Mapped[list[Permission]] = relationship(
        "Permission",
        secondary="user_permissions",
        back_populates="users",
        cascade="all",
    )
    workspaces: Mapped[list[Workspace]] = relationship(
        "Workspace",
        secondary="user_workspaces",
        back_populates="users",
    )


class Workspace(Base):
    """Logical grouping for collections and access control."""

    __tablename__ = "workspaces"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default="")

    users: Mapped[list[User]] = relationship(
        "User",
        secondary="user_workspaces",
        back_populates="workspaces",
    )
    collections: Mapped[list[Collection]] = relationship(
        "Collection",
        secondary="workspace_collections",
        back_populates="workspaces",
    )


class Permission(Base):
    """Discrete permission that can be attached to users."""

    __tablename__ = "permissions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default="")

    users: Mapped[list[User]] = relationship(
        "User",
        secondary="user_permissions",
        back_populates="permissions",
    )


class Collection(Base):
    """Collection of documents bundled for retrieval."""

    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default="")
    document_count: Mapped[int] = mapped_column(Integer, default=0)

    workspaces: Mapped[list[Workspace]] = relationship(
        "Workspace",
        secondary="workspace_collections",
        back_populates="collections",
    )


class UserPermission(Base):
    """Association table mapping users to permissions."""

    __tablename__ = "user_permissions"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    permission_id: Mapped[int] = mapped_column(
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (UniqueConstraint("user_id", "permission_id", name="uq_user_permission"),)


class UserWorkspace(Base):
    """Association table mapping users to workspaces."""

    __tablename__ = "user_workspaces"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (UniqueConstraint("user_id", "workspace_id", name="uq_user_workspace"),)


class WorkspaceCollection(Base):
    """Association table mapping workspaces to collections."""

    __tablename__ = "workspace_collections"

    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    )
    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (UniqueConstraint("workspace_id", "collection_id", name="uq_workspace_collection"),)
