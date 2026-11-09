# Original data
The unchanged CSV, SQLite snapshot and exact saved train/test ID files are in raw/.
The experiment loader reads them directly and never regenerates them.
original_sha256.txt verifies all four data files plus the historical baseline reference in src/.
The dataset is synthetic and its labels are heuristic.
Original Midterm sources and the old presentation were removed from the working tree;
a local backup is kept in archive/midterm_originals.zip, excluded from Git but included
in the submission ZIP when available.
