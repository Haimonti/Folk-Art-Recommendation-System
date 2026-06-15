"""GeMi preference-capture data layer. One schema, any engine, selected by DATABASE_URL."""
import os
import json
import uuid
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, Integer, BigInteger, DateTime, Text,
    ForeignKey, Index, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import TypeDecorator

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./gemi_local.db")

_is_sqlite = DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


class JSONEncodedDict(TypeDecorator):
    """JSON stored as portable TEXT so behaviour is identical on SQLite and MySQL."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None or value == "":
            return None
        return json.loads(value)


def new_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    user_id = Column(String(36), primary_key=True, default=new_uuid)
    handle = Column(String(64), nullable=True, index=True)
    email = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSONEncodedDict, nullable=True)


class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=True, index=True)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    user_agent = Column(String(512), nullable=True)
    referrer = Column(String(512), nullable=True)


class Event(Base):
    __tablename__ = "events"
    event_id = Column(BigIntPK, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id"), nullable=False, index=True)
    panel_index = Column(Integer, nullable=True, index=True)
    panel_id = Column(String(64), nullable=True, index=True)
    event_type = Column(String(32), nullable=False, index=True)
    value = Column(Integer, nullable=True)
    context = Column(JSONEncodedDict, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Recommendation(Base):
    __tablename__ = "recommendations"
    rec_id = Column(BigIntPK, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id"), nullable=False, index=True)
    model_name = Column(String(64), nullable=True, index=True)
    seed_panel_indices = Column(JSONEncodedDict, nullable=True)
    served_panel_indices = Column(JSONEncodedDict, nullable=True)
    query_text = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


Index("ix_events_subject", Event.session_id, Event.panel_index, Event.event_type)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
