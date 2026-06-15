"""Optional preference-capture endpoints. Failures here never affect core recommendation."""
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .db import SessionLocal
from . import store

router = APIRouter(prefix="/api", tags=["capture"])


class RegisterRequest(BaseModel):
    handle: Optional[str] = None
    email: Optional[str] = None
    meta: Optional[dict] = None


class EventRequest(BaseModel):
    session_id: str
    event_type: str
    panel_index: Optional[int] = None
    panel_id: Optional[str] = None
    value: Optional[int] = None
    user_id: Optional[str] = None
    context: Optional[dict] = None


class RecoLogRequest(BaseModel):
    session_id: str
    model_name: Optional[str] = None
    seed_panel_indices: Optional[list[int]] = None
    served_panel_indices: Optional[list[int]] = None
    query_text: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/register")
def register(req: RegisterRequest):
    db = SessionLocal()
    try:
        if req.handle:
            existing = store.get_user_by_handle(db, req.handle)
            if existing is not None:
                return {"user_id": existing.user_id, "handle": existing.handle, "returning": True}
        user = store.create_user(db, handle=req.handle, email=req.email, meta=req.meta)
        return {"user_id": user.user_id, "handle": user.handle, "returning": False}
    except Exception as e:
        return {"user_id": None, "error": str(e)}
    finally:
        db.close()


@router.post("/events")
def log_event(req: EventRequest):
    db = SessionLocal()
    try:
        store.get_or_create_session(db, session_id=req.session_id, user_id=req.user_id)
        store.record_event(db, req.session_id, req.event_type,
                           panel_index=req.panel_index, panel_id=req.panel_id,
                           value=req.value, context=req.context)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


@router.post("/recommendations/log")
def log_recommendation(req: RecoLogRequest):
    db = SessionLocal()
    try:
        store.get_or_create_session(db, session_id=req.session_id, user_id=req.user_id)
        store.record_recommendation(db, req.session_id, model_name=req.model_name,
                                    seed_panel_indices=req.seed_panel_indices,
                                    served_panel_indices=req.served_panel_indices,
                                    query_text=req.query_text)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


@router.get("/preferences/matrix")
def preference_matrix(by: str = "user"):
    db = SessionLocal()
    try:
        rows = store.build_preference_matrix(db, by=by)
        return {"by": by, "count": len(rows), "rows": rows}
    except Exception as e:
        return {"by": by, "count": 0, "rows": [], "error": str(e)}
    finally:
        db.close()
