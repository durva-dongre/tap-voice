import sys; sys.path.insert(0, "pod")
from app.route import plan
J = lambda *ls: [{"id": i, "language": l} for i, l in enumerate(ls)]
def test_empty():          assert plan([]) == ([], [], "none")
def test_en_only():        g, o, m = plan(J("en", "en")); assert [x[0] for x in g] == ["kokoro"] and m == "kokoro"
def test_indic_only():     g, o, m = plan(J("hi", "mr", "pa", "kn")); assert [x[0] for x in g] == ["indic"] and m == "indic"
def test_both_bigger_first(): g, o, m = plan(J("en", "hi", "hi")); assert [x[0] for x in g] == ["indic", "kokoro"] and m == "both"
def test_unsupported():    g, o, m = plan(J("en", "fr")); assert len(o) == 1 and m == "kokoro"
