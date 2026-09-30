"""Alembic environment: metadata from app.db, URL from DATABASE_URL."""
from alembic import context
from app.config import Settings
from app.db import Base, build_engine

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=Settings().database_url, target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = build_engine(Settings().database_url)
    with engine.connect() as connection:
        # render_as_batch lets ALTERs work on SQLite (local dev) as well as Postgres.
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
