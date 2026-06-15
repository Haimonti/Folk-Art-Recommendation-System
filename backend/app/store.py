"""Capture operations and the user-preference matrix derived from logged events."""
from sqlalchemy import select

from .db import User, Session as VisitSession, Event, Recommendation

DEFAULT_WEIGHTS = {
    "like": 3.0,
    "open_scroll": 2.0,
    "click_reco": 1.0,
    "dwell": 1.0,
    "view": 0.1,
    "unlike": -3.0,
}

DWELL_MS_THRESHOLD = 3000


def create_user(db, handle=None, email=None, meta=None):
    user = User(handle=handle, email=email, meta=meta)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_handle(db, handle):
    return db.execute(select(User).where(User.handle == handle)).scalar_one_or_none()


def get_or_create_session(db, session_id, user_id=None, user_agent=None, referrer=None):
    existing = db.get(VisitSession, session_id)
    if existing is not None:
        if user_id and not existing.user_id:
            existing.user_id = user_id
            db.commit()
        return existing
    visit = VisitSession(
        session_id=session_id,
        user_id=user_id,
        user_agent=user_agent,
        referrer=referrer,
    )
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit


def record_event(db, session_id, event_type, panel_index=None, panel_id=None,
                 value=None, context=None):
    ev = Event(
        session_id=session_id,
        panel_index=panel_index,
        panel_id=panel_id,
        event_type=event_type,
        value=value,
        context=context,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


def record_recommendation(db, session_id, model_name=None, seed_panel_indices=None,
                          served_panel_indices=None, query_text=None):
    rec = Recommendation(
        session_id=session_id,
        model_name=model_name,
        seed_panel_indices=seed_panel_indices,
        served_panel_indices=served_panel_indices,
        query_text=query_text,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def build_preference_matrix(db, weights=None, by="user"):
    if weights is None:
        weights = DEFAULT_WEIGHTS
    rows = db.execute(
        select(Event.session_id, Event.panel_index, Event.event_type, Event.value,
               VisitSession.user_id)
        .join(VisitSession, Event.session_id == VisitSession.session_id)
        .where(Event.panel_index.isnot(None))
    ).all()
    scores = {}
    for session_id, panel_index, event_type, value, user_id in rows:
        if by == "user":
            subject = user_id if user_id else "anon:" + session_id
        else:
            subject = session_id
        weight = weights.get(event_type, 0.0)
        if event_type == "dwell" and (value is None or value < DWELL_MS_THRESHOLD):
            weight = 0.0
        key = (subject, panel_index)
        scores[key] = scores.get(key, 0.0) + weight
    result = [
        {"subject_id": subject, "panel_index": panel_index, "score": round(score, 3)}
        for (subject, panel_index), score in scores.items()
        if score != 0.0
    ]
    result.sort(key=lambda r: (str(r["subject_id"]), -r["score"]))
    return result


def pivot_matrix(long_rows):
    subjects = sorted({str(r["subject_id"]) for r in long_rows})
    panels = sorted({r["panel_index"] for r in long_rows})
    index = {(str(r["subject_id"]), r["panel_index"]): r["score"] for r in long_rows}
    matrix = [[index.get((s, p), 0.0) for p in panels] for s in subjects]
    return subjects, panels, matrix
