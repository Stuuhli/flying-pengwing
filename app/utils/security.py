"""Security helpers for JWT based authentication."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field, ValidationError
from sqlalchemy.orm import Session

from app.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)
from app.db.database import get_db
from app.utils.utils_auth import UserProfile, build_user_profile, get_user_by_email


class TokenPayload(BaseModel):
    """Structure of the JWT claims used by the backend."""

    sub: EmailStr
    user_id: int
    permissions: list[str]
    is_admin: bool
    rag_type: str
    workspaces: list[int] = Field(default_factory=list)
    exp: datetime


class CurrentUser(BaseModel):
    """Container returned by the authentication dependency."""

    profile: UserProfile
    claims: TokenPayload


def create_access_token(
    *,
    subject: EmailStr,
    user_id: int,
    permissions: Iterable[str],
    is_admin: bool,
    rag_type: str,
    workspaces: Iterable[int],
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT for the authenticated user."""

    expire = datetime.now(tz=timezone.utc) + (
        expires_delta if expires_delta is not None else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "user_id": user_id,
        "permissions": list(permissions),
        "is_admin": is_admin,
        "rag_type": rag_type,
        "workspaces": list(workspaces),
        "exp": expire,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except (JWTError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from exc


def get_current_user(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
) -> CurrentUser:
    """Dependency that resolves the current authenticated user."""

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    token = authorization.split(" ", 1)[1].strip()
    claims = _decode_token(token)

    user = get_user_by_email(db, email=str(claims.sub))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    profile = build_user_profile(user)
    return CurrentUser(profile=profile, claims=claims)


__all__ = [
    "TokenPayload",
    "CurrentUser",
    "create_access_token",
    "get_current_user",
]
