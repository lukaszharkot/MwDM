# Klasyfikacja Sentymentu na Twitterze

Ten projekt implementuje model głębokiego uczenia do klasyfikacji sentymentu tweetów, wykorzystując podejście oparte na LSTM. Notebook przetwarza surowe dane tekstowe, trenuje model i ocenia jego wydajność.

## Przebieg Projektu

1. **Importowanie Wymaganych Bibliotek**
   - Projekt wykorzystuje biblioteki Python, takie jak TensorFlow, datasets, numpy, pandas i matplotlib.

2. **Przygotowanie Danych**
   - Pobiera zestawy danych do trenowania i testowania z określonego URL.
   - Dzieli dane na zbiory: treningowy, walidacyjny i testowy.
   - Zapisuje przetworzone dane w formacie JSONL do dalszego wykorzystania.

3. **Budowa Modelu**
   - Implementuje model sekwencyjny LSTM z warstwami embedding, LSTM i dense.
   - Optymalizuje model za pomocą optymalizatora Adam.

4. **Trenowanie i Ewaluacja**
   - Trenuje model na przygotowanym zbiorze danych.
   - Ocenia wydajność za pomocą takich metryk jak dokładność (accuracy).

5. **Wizualizacja**
   - Zawiera wykresy ilustrujące metryki, takie jak strata i dokładność w czasie epok treningowych.

## Wyniki
- Notebook przedstawia wnioski dotyczące wydajności modelu na danych testowych oraz wizualizuje postęp podczas treningu.

## Wymagania
- Python 3.x
- Biblioteki: TensorFlow, datasets, numpy, pandas, matplotlib

## Użycie
1. Sklonuj repozytorium i przejdź do katalogu projektu.
2. Zainstaluj wymagane zależności, używając polecenia `pip install -r requirements.txt`.
3. Uruchom notebook w Jupyter lub innym kompatybilnym środowisku.

## Opis Plików
- **TwitterSentimentClassification.ipynb**: Główny notebook zawierający kod, objaśnienia i wyniki.
- **Przetworzone Pliki JSONL**: Tworzone podczas przygotowania danych do treningu i testowania.

## Podziękowania
- Źródło danych: [GitHub Repository](https://github.com/cblancac/SentimentAnalysisBert)

---
