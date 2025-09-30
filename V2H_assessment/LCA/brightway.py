# -*- coding: utf-8 -*-
import os, sys, zipfile, tempfile, shutil, importlib
import bw2data as bd
import bw2io as bi
import bw2calc as bc
from bw2data import Method
from bw2io.importers import Ecospold1Importer


# ======= ANPASSEN =======
PROJECT  = "my_lca_project"
DB_NAME  = "ecoinvent_full_es1"
ES1_PATH = r"C:\Users\Philipp Thunshirn\Desktop\PhD\openLCA\openlcaexportsingle\EcoSpold01"  # Ordner ODER .zip ODER einzelne .xml

# ------- Importer-Finder (verschiedene bw2io-Versionen) -------
def get_es1_importer_class():
    """
    Versucht, den Ecospold1Importer aus verschiedenen bw2io-Namespace-Varianten zu laden.
    Wirft einen klaren Fehler, wenn er nicht vorhanden ist.
    """
    candidates = [
        ("bw2io.importers.ecospold1", "Ecospold1Importer"),
        ("bw2io.ecospold1", "Ecospold1Importer"),
    ]
    for modname, clsname in candidates:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, clsname):
                return getattr(mod, clsname)
        except Exception:
            pass
    raise ImportError(
        "Kein Ecospold1Importer gefunden.\n"
        "→ In deiner venv updaten:\n"
        "   python -m pip install --upgrade pip\n"
        "   pip install --upgrade bw2io bw2data bw2calc lxml"
    )

# ------- Pfad vorbereiten: Ordner / .zip / einzelne .xml -------
def ensure_folder_with_xml(path: str):
    """
    Liefert (folder, tmpdir).
    - Wenn path Ordner: prüft, ob darin .xml liegen.
    - Wenn path .zip: entpackt in temp, sucht Unterordner mit .xml.
    - Wenn path einzelne .xml: gibt deren Ordner zurück.
    tmpdir != None nur wenn aus .zip entpackt (am Ende aufräumen).
    """
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            if any(f.lower().endswith(".xml") for f in files):
                return path, None
        raise FileNotFoundError("Im Ordner wurden keine .xml-Dateien gefunden.")
    if os.path.isfile(path):
        low = path.lower()
        if low.endswith(".zip"):
            tmpdir = tempfile.mkdtemp(prefix="es1_")
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(tmpdir)
            candidate = tmpdir
            # Häufige Unterordnernamen
            for sub in ("datasets", "dataset", "export", "ecospold", "EcoSpold01", "ecospold01"):
                p = os.path.join(tmpdir, sub)
                if os.path.isdir(p):
                    candidate = p
                    break
            for root, _, files in os.walk(candidate):
                if any(f.lower().endswith(".xml") for f in files):
                    return candidate, tmpdir
            raise FileNotFoundError("In der .zip keine .xml-Dateien gefunden.")
        elif low.endswith(".xml"):
            folder = os.path.dirname(path)
            if not folder:
                folder = "."
            return folder, None
    raise FileNotFoundError("ES1_PATH ist weder Ordner mit .xml, eine .xml, noch eine .zip.")

# ======= START =======
bd.projects.set_current(PROJECT)

# Basis bereitstellen
if "biosphere3" not in bd.databases:
    bi.create_default_biosphere3()
if len(bd.methods) == 0:
    bi.create_default_lcia_methods()

# Ziel-DB sauber neu
if DB_NAME in bd.databases:
    print(f"⚠ DB '{DB_NAME}' existiert bereits – wird gelöscht.")
    del bd.databases[DB_NAME]

# Quelle prüfen/aufbereiten
folder, tmpdir = ensure_folder_with_xml(ES1_PATH)
print("Importquelle (EcoSpold01):", folder)

# Importer laden
Importer = get_es1_importer_class()

# Import
imp = Importer(folder, DB_NAME)
imp.apply_strategies()
imp.statistics()
imp.write_database()
count = len(list(bd.Database(DB_NAME)))
print(f"✔ ES1-Import abgeschlossen: '{DB_NAME}' mit {count} Datensätzen")

# --- Smoke-Test: 1 Prozess (LCI + LCIA) ---
db = bd.Database(DB_NAME)
procs = [ds for ds in db if ds.get("type") == "process"]
if not procs:
    print("⚠ Keine Prozesse gefunden – Export prüfen (enthält der Ordner wirklich Prozesse?).")
else:
    act = procs[0]
    prod = next((e.input for e in act.exchanges() if e["type"] == "production"), None)
    if prod:
        fu = {prod: 1.0}
        try:
            lca = bc.LCA(fu); lca.lci()
            print("✔ LCI ok für:", act.get("name"), "| Unit:", prod.get("unit"))
        except Exception as e:
            print("⚠ LCI fehlgeschlagen:", e)

        # robuste Methode wählen & Cache bauen (gegen „processed … .zip not found“)
        def pick_method():
            for m in bd.methods:
                s = " ".join(m).lower()
                if ("ipcc" in s and "2013" in s and "100a" in s):
                    return m
            for m in bd.methods:
                s = " ".join(m).lower()
                if ("ef 3.1" in s and "climate change" in s and "midpoint" in s):
                    return m
            return next(iter(bd.methods))

        try:
            method = pick_method()
            try:
                Method(method).process()
            except Exception:
                pass
            lca = bc.LCA(fu, method); lca.lci(); lca.lcia()
            print("✔ LCIA-Score:", lca.score, "| Methode:", method)
        except Exception as e:
            print("⚠ LCIA nicht möglich:", e)

# tmp aufräumen (falls .zip)
if tmpdir:
    shutil.rmtree(tmpdir, ignore_errors=True)
