EN, INDIC = {"en"}, {"hi", "mr", "pa", "kn"}

def plan(jobs):
    en = [j for j in jobs if j["language"] in EN]
    ind = [j for j in jobs if j["language"] in INDIC]
    other = [j for j in jobs if j["language"] not in EN | INDIC]
    groups = [g for g in sorted([("kokoro", en), ("indic", ind)], key=lambda g: -len(g[1])) if g[1]]
    models = "both" if len(groups) == 2 else (groups[0][0] if groups else "none")
    return groups, other, models
