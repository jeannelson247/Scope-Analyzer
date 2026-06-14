from obsidian_notes import session_note_markdown


def test_session_note_includes_ai_trace_metadata():
    md = session_note_markdown(
        shot_name="T0000",
        source_path="/tmp/T0000.CSV",
        source_hash="abcdef1234567890",
        channels=["CH1", "CH2"],
        tool_events=["detect_anomalies: ok"],
        ai_events=[
            "AI trace: app=0.1.0; backend=mlx; model=test; "
            "prompt_sha256=abc; system_sha256=def; max_tokens=768; source=123"
        ],
    )

    assert "## AI annotation trace" in md
    assert "prompt_sha256=abc" in md
    assert "backend=mlx" in md
