"""Database session and initialization helpers."""

from collections.abc import Generator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


# Configure engine and session factory
_connect_args: dict[str, object] = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator:
    """Yield a database session and ensure it is closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _seed_permissions() -> None:
    """Ensure the static permission matrix exists."""
    from app.db.models import Permission  # Avoid circular imports

    default_permissions: dict[str, str] = {
        "user": "Standardbenutzer mit Leseberechtigungen.",
        "admin": "Administrationsberechtigungen für die Plattform.",
        "rag": "Zugriff auf Retrieval-Augmented-Generation Workflows.",
        "graphrag": "Zugriff auf GraphRAG Workflows.",
    }

    with SessionLocal() as session:
        existing = {
            name
            for (name,) in session.execute(select(Permission.name)).all()
        }
        for name, description in default_permissions.items():
            if name not in existing:
                session.add(Permission(name=name, description=description))
        session.commit()


def init_db() -> None:
    """Create database tables and seed reference data if required."""
    from app.db import models  # noqa: F401  # Ensure models are imported

    Base.metadata.create_all(bind=engine)
    _seed_permissions()
