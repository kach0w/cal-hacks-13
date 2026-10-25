"""End to end on the simulated table (tools/simulate.py, scripted mode): video file -> VideoSource ->
Pipeline thread -> change gate -> Memory -> Store -> Query, with ground-truth boxes instead of a model."""
import sys

from recall.store import Store


def test_scripted_simulation(tmp_path, monkeypatch, capsys):
    from tools import simulate
    monkeypatch.setattr(sys, "argv", ["simulate", "--set", f"DATA_DIR={tmp_path}"])
    simulate.main()
    out = tmp_path / "sim" / "scripted"
    events = [(e.kind, e.object, e.from_surface, e.to_surface) for e in Store(out).events()]
    assert events == [
        ("appeared", "keys", None, "kitchen counter"),
        ("appeared", "glasses", None, "kitchen counter"),
        ("appeared", "remote", None, "table"),
        ("appeared", "wallet", None, "table"),
        ("appeared", "bag", None, "table"),
        ("moved", "keys", "kitchen counter", "table"),             # via mid-air: one move, not two
        # the hand over the remote at t=9 commits nothing
        ("disappeared", "glasses", "kitchen counter", None),
        ("reappeared", "glasses", "kitchen counter", "table"),
        ("inferred_contained", "glasses", "table", "table"),
        ("moved", "bag", "table", "kitchen counter"),
        ("moved", "glasses", "table", "kitchen counter"),          # carried inside the bag
    ]
    printed = capsys.readouterr().out
    assert "Your glasses are probably in the bag, on the kitchen counter — inferred" in printed
    assert len(list((out / "screens").glob("*.jpg"))) > 20 and (out / "contact_sheet.jpg").exists()
