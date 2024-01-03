import pandas as pd
import numpy
from IPython.display import display

# Caricamento e conversione dei file CSV da ',' a ';'
converted_files = []

for file_path in ["Testing_2_with_Antibiotics.csv", 
                  "Testing_2-NoTarget.csv", 
                  "Testing_2.csv"]:
    # Leggi il file con delimitatore ','
    df = pd.read_csv(file_path, delimiter=',', index_col='ID Strain')

    # Crea un nuovo nome file con suffisso "_semicolon"
    new_file_path = file_path.replace(".csv", "_.csv")
    display(df)
    # Salva il file con delimitatore ';'
    df.to_csv(file_path, sep=';', index=True)
    converted_files.append(new_file_path)