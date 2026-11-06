"""Guard: the repository must not contain any reference to persons, examinations or local data locations."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "pseudonym": re.compile(r"\b[PS]\d{3,4}(_E\d+)?\b"),
    "drive path": re.compile(r"\b[A-Z]:[\\/]"),
    "user folder": re.compile(r"Users[\\/]\w+", re.I),
    "date": re.compile(r"\b\d{1,2}\.\d{1,2}\.(19|20)\d\d\b|\b(19|20)\d\d-\d\d-\d\d\b"),
    "patient tag": re.compile(r"Patient(Name|ID|BirthDate|Sex|Age|Weight|Size)|AccessionNumber|StudyInstanceUID|InstitutionName|ReferringPhysician"),
}


def test_no_identifiers():
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.name == Path(__file__).name or any(s in p.parts for s in (".git", "__pycache__", ".pytest_cache")):
            continue
        if p.suffix.lower() not in (".py", ".md", ".json", ".toml", ".yml", ".txt", ".csv", ".cmd", ".cfg", ".cff", ""):
            hits.append(f"{p.relative_to(ROOT)}: unexpected file type")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                hits.append(f"{p.relative_to(ROOT)}: {name} '{m.group(0)}'")
    assert not hits, "\n".join(hits)
