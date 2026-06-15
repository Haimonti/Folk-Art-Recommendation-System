"""Local smoke test for the GeMi preference-capture layer. No server required."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///./gemi_local.db")

from app.db import init_db, SessionLocal
from app import store


def main():
    init_db()
    db = SessionLocal()

    user = store.create_user(db, handle="visitor_one")
    visit = store.get_or_create_session(db, session_id="ab12cd34",
                                        user_id=user.user_id, user_agent="test-agent")

    store.record_event(db, visit.session_id, "view", panel_index=17, panel_id="17",
                       context={"position": 1, "model_name": "llamasigclip_gcn"})
    store.record_event(db, visit.session_id, "dwell", panel_index=17, panel_id="17", value=8200)
    store.record_event(db, visit.session_id, "like", panel_index=17, panel_id="17")
    store.record_event(db, visit.session_id, "view", panel_index=42, panel_id="42")
    store.record_event(db, visit.session_id, "dwell", panel_index=42, panel_id="42", value=900)

    store.record_recommendation(db, visit.session_id, model_name="llamasigclip_gcn",
                                seed_panel_indices=[17],
                                served_panel_indices=[42, 55, 88])

    anon = store.get_or_create_session(db, session_id="ff99ee88", user_agent="anon-test")
    store.record_event(db, anon.session_id, "like", panel_index=55, panel_id="55")

    long_rows = store.build_preference_matrix(db, by="user")
    subjects, panels, matrix = store.pivot_matrix(long_rows)

    print("Preference rows (subject, panel_index, score):")
    for r in long_rows:
        print("   ", r)
    print()
    print("Matrix panels:", panels)
    for s, row in zip(subjects, matrix):
        print("   ", s[:20], row)
    print()

    p17 = next(r for r in long_rows if r["panel_index"] == 17)
    assert abs(p17["score"] - 4.1) < 1e-6, p17
    p42 = next(r for r in long_rows if r["panel_index"] == 42)
    assert abs(p42["score"] - 0.1) < 1e-6, p42
    assert any(r["panel_index"] == 55 and str(r["subject_id"]).startswith("anon:")
               for r in long_rows)
    print("OK: schema, capture, and preference-matrix build all working.")


if __name__ == "__main__":
    main()
