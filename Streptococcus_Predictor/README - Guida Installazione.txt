CONFIGURAZIONE PER ESECUZIONE PROGRAMMA:
1. Installare python tramite il file SETUP/python-3.11.7-installer
2. Dalla cartella Streptococcus_Predictor, fare click dx e selezionare "apri nel Terminale"
3. Creare un ambiente virtuale se non si vuole intaccare le librerie di sistema (OPZIONALE)
4. Eseguire il comando: 
pip install -r SETUP/requirements.txt
5. Aggiungere nella cartella Streptococcus_Predictor il file contenente i campioni in cui va prevista la sottospecie di Streptococco, questo può contenere uno o più campioni 
6. Il file deve contenere solo: [ID_Strain, Specie animale, Haemolysis, Picchi maldi]
Esattamente in questo ordine, aprire il file tabella_input.csv per visualizzare il formato e la struttura che deve avere la tabella, usare come separatore la ','
7. Eseguire il comando: 
python Streptococcus_Predictor.py
8. Inserire il nome del file contenente i campioni da prevedere
9. Inserire quante colonne rappresentanti i picchi Maldi sono presenti nella tabella di input
10. Visualizzazione risultati direttamente dal terminale oppure tramite il file results.csv
11. Rinominare e spostare il file results.csv così da non rischiare di perdere le previsioni fatte una volta che viene rieseguito lo script


Per qualsiasi problema nella configurazione o esecuzione, scrivere a gabriele.panciotti@studenti.unipg.it