# Analiza Sentymentu na Twitterze - Projekt z użyciem LSTM

## Opis

Ten projekt ma na celu analizę sentymentów na podstawie tweetów z Twittera. Model oparty jest na sieci neuronowej LSTM (Long Short-Term Memory), która klasyfikuje teksty jako pozytywne lub negatywne na podstawie ich treści.

### Cele Projektu:
- Pobranie i przetworzenie danych z Twittera.
- Zbudowanie i wytrenowanie modelu LSTM do analizy sentymentu.
- Ocena dokładności modelu na zbiorach treningowym, walidacyjnym i testowym.
- Wizualizacja wyników w postaci wykresów oraz przykładowych predykcji.

## Struktura Projektu

1. **Importowanie bibliotek**  
   W tym kroku zaimportowano niezbędne biblioteki, takie jak `tensorflow`, `pandas`, `numpy` i inne, które są wymagane do przetwarzania danych i budowy modelu.

2. **Pobieranie i przetwarzanie danych**  
   Datasets zostały pobrane z linków GitHub i przygotowane do analizy. Dane są przetwarzane i dzielone na zbiory: treningowy, walidacyjny i testowy.

3. **Przygotowanie danych**  
   Przekształcone dane zostały tokenizowane, a teksty zostały zamienione na sekwencje liczb, które są następnie uzupełniane do stałej długości.

4. **Budowanie i trenowanie modelu**  
   Zbudowano model LSTM, który jest trenowany na zbiorze treningowym. Model składa się z warstw `Embedding`, `LSTM`, oraz `Dense`.

5. **Ocena i wizualizacja wyników**  
   Po przetrenowaniu modelu, dokładność jest oceniana na zbiorze testowym. Dodatkowo, generowane są wykresy ilustrujące dokładność modelu na różnych etapach treningu.

6. **Predykcje**  
   Na końcu przedstawiono przykłady predykcji na nowych danych, porównując je z rzeczywistymi etykietami.

7. **Usuwanie plików**  
   Na koniec, po zakończeniu analizy, pliki JSONL utworzone podczas przetwarzania danych są usuwane.

## Wymagania

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Datasets (Hugging Face)
