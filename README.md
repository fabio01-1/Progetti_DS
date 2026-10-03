# Spotify Popularity Analysis

Le caratteristiche audio di un brano e il suo genere musicale permettono di prevedere quanto sarà popolare su Spotify?

Il progetto trasforma `popularity` (indice 0-100) in 4 classi di popolarità (quartili) e confronta una Regressione Logistica con LightGBM, valutando anche le probabilità predette (Brier Score, log-loss, calibrazione) e l'interpretabilità (importanza per guadagno e per permutazione, SHAP).

## Dataset
[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle, maharshipandya): 114.000 tracce, 1.000 per ciascuno dei 114 generi. Scarica `dataset.csv` e salvalo in `data/dataset.csv` (il file non è incluso nel repository: controlla la licenza su Kaggle).

## Metodo
1. **Pulizia**: valori mancanti, duplicati esatti (la stessa registrazione compare in più generi) e quasi duplicati (stesso titolo e artista, audio entro tolleranze definite per ogni feature e stesse variabili categoriche), rimossi in tre fasi: (A) tracce a `popularity = 0` con una copia positiva, (B) duplicati tra tracce a 0, (C) duplicati tra tracce positive con la stessa `popularity`. Poi filtro sulla durata (percentili 0.5-99.5) e sul tempo (>= 50 BPM). Ogni passaggio è registrato in una tabella riepilogativa (114.000 → 84.709 righe) e un controllo finale verifica che non restino duplicati né brani spariti del tutto.
2. **EDA**: distribuzioni, correlazioni di Spearman (anche escludendo le tracce a 0), profilo audio per classe, associazione delle variabili categoriche con le classi (V di Cramér).
3. **Modelli**: split train/test **per brano** (titolo + artista normalizzati, `GroupShuffleSplit`, 80/20), preprocessing dentro la `Pipeline`. Gli `assert` verificano che nessuna riga, `track_id` o brano compaia sia nel train sia nel test.
4. **Valutazione**: accuracy, F1 macro, errore ordinale medio, Brier Score (e Brier skill score), log-loss, ECE, calibration curve; confronto con un riferimento banale; confronto "solo audio / solo genere / audio + genere / tutte le feature".
5. **Interpretabilità (LightGBM)**: importanza per guadagno (`gain`, colonne one-hot sommate per variabile) e per permutazione; SHAP globale, per genere e su singole tracce.

Una cella finale di controllo verifica che i valori citati nelle conclusioni coincidano con quelli calcolati.

## Risultati principali (test set, circa 16.900 tracce)
| | Accuracy | Brier Score |
|---|---|---|
| Classe più frequente | 0.27 | 0.75 |
| Regressione Logistica | 0.61 | 0.50 |
| LightGBM | circa 0.64 | circa 0.48 |

- Il genere è la variabile dominante (V di Cramér circa 0.57 con le classi; le altre variabili categoriche non superano 0.07). Con la Regressione Logistica le sole feature audio raggiungono 0.35 di accuracy (0.60 con il solo genere). Con LightGBM aggiungere le feature audio al genere porta un guadagno maggiore.
- Le correlazioni di Spearman tra feature audio e `popularity` sono tutte deboli (al massimo circa 0.18 in valore assoluto, `instrumentalness`).
- Permutation importance e SHAP indicano il genere come variabile dominante, con un distacco netto dalle feature audio; descrivono il comportamento del modello, non cause della popolarità.
- Le probabilità sono ben calibrate sul test set senza ricalibrazione (ECE circa 0.01-0.02).
- Limiti (campione costruito per genere, genere dei duplicati scelto a caso, `popularity` come indice istantaneo con zeri spesso dovuti ad assenza di dato, fase A della deduplica che seleziona in base al target, una sola partizione, nessun tuning) sono discussi nelle conclusioni del notebook.

## Come eseguirlo
```bash
pip install -r requirements.txt
# scarica il CSV in data/dataset.csv, poi:
jupyter notebook Spotify_Analysis_Project.ipynb
```
I grafici vengono salvati in `figures/`. Seed fissato (`RANDOM_STATE = 42`).

## Struttura
```
Spotify_Analysis_Project.ipynb
requirements.txt
data/dataset.csv   (da scaricare)
figures/           (generata dal notebook)
```
