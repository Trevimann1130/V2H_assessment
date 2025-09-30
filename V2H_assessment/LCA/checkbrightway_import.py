import bw2data as bd, bw2io as bi, bw2calc as bc

PROJECT  = "my_lca_project"
DB_NAME  = "ecoinvent_full_es2"
ES2_DIR  = r"C:\Users\Philipp Thunshirn\Desktop\PhD\openLCA\export_es2"  # dein Ordner mit ES2-Dateien

bd.projects.set_current(PROJECT)
if "biosphere3" not in bd.databases: bi.create_default_biosphere3()
if len(bd.methods) == 0: bi.create_default_lcia_methods()
if DB_NAME in bd.databases: del bd.databases[DB_NAME]

imp = Ecospold2Importer(ES2_DIR, DB_NAME)
imp.apply_strategies()
imp.statistics()
imp.write_database()

print("Fertig:", len(bd.Database(DB_NAME)), "Datensätze")
