# Thesis-Streptococcus-Classification
L'obiettivo del progetto è creare un classificatore basato sui dati MALDI (macro-gruppo 3) che possa predire la sottospecie di Streptococcus (colonna H) e l'antibiotico-sensibilità (macro-gruppi 4 e 5). Qualora i dati MALDI non fossero sufficienti per la classificazione, sarebbe possibile inserire nel modello anche le informazioni delle colonne D, I e J. Infine altre informazioni che sarebbe interessante poter predire sono i sequence type (ST- colonna F) ed i fattori di virulenza (macro-gruppo 6).

Come vedrai, i dati sono organizzati in sei macro-gruppi:

 1. Metadati origine isolati. Si tratta di informazioni relative all'anno, la regione, la specie animale e l'organo di provenienza dell'isolato. La colonna ID Strain rappresenta l'identificativo di ogni isolato di Streptoccus e quindi è quella che ci consente di rintracciare le informazioni.

 2. Informazioni principali isolati. Sono riportati i dati principali di ogni isolato in termini microbiologici. La colonna H è quella più importante per la finalità del progetto: avremmo bisogno di poter predire la sottospecie a cui appartiene ciascun isolato. L'identificazione che trovi alla colonna H è stata ottenuta tramite genomica e rappresenta il valore di riferimento. Un'altra colonna che ci piacerebbe poter predire è la colonna F. Questa colonna rappresenta gli ST, identificativi genomici di un ceppo batterico. Le colonne I e J (lancefield group e haemolysis) sono dati che possono essere integrati nel classificatore per predire la colonna H.

 3. Picchi MALDI m/z individuati e loro intensità. Sono i dati principali da utilizzare nel classificatore, derivanti dai test in MALDI-TOF. Nelle colonne sono descritti i picchi m/z (massa/carica). I valori rappresentano l'intensità di ciascun picco relativamente ad un campione.

 4. Sensibilità agli antibiotici (S: sensibile; NS: non sensibile). Sono i dati relativi alla sensibilità agli antibiotici. Sarebbe interessante verificare se il classificatore basato sui picchi MALDI riesca a predire la sensibilità agli antibiotici.

 5. Geni di antibiotico-resistenza. Presenza (1) o assenza (0) di geni di resistenza per gli antibiotici. Sarebbe interessante verificare se il classificatore basato sui picchi MALDI riesca a predire la presenza di questi geni, i dati potrebbero esser simili a quelli del punto 4.

 6. Fattori di virulenza. Presenza (1) o assenza (0) di fattori di virulenza. Si potrebbe verificare se si possano predire dai picchi MALDI.
