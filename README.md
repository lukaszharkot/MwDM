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

## Wyniki
**Dokładność treningowa i walidacyjna podczas trenowania:**  

![Dokładność treningowa i walidacyjna](images/Epochs.png "Dokładność modelu na różnych etapach treningu")

- Przy występowaniu większej liczby epok w modelu występuje przeuczenie. Z tego powodu zastosowana została funkcja EarlyStopping, która przywraca najlepszą wagę modelu podczas jego uczenia.

**Wykres dokładności:**  

![Wykres dokładności](images/Chart.png "Wykres dokładności modelu")

- Mimo wrostu dokładności przy uczeniu na danych treningowych dokładność walidacyjna po pewnym czasie maleje.

**Dokładność testowa:**  

accuracy: 0.7902 - loss: 0.4486  
Dokładność na zbiorze testowym: 0.79

**Wyświetlamy przykładowe predykcje i rzeczywiste etykiety:**  

Przykładowe predykcje:  

 - Tweet: @justineville ...yeahhh. i'm 39 tweets from 1,600!  
   Predykcja: Negatywna, Rzeczywista etykieta: Negatywna

 - Tweet: @ApplesnFeathers aww. Poor baby! On your only REAL day off.  
   Predykcja: Pozytywna, Rzeczywista etykieta: Pozytywna

 - Tweet: @joeymcintyre With my refunded $225 (Australian ticket price) I bought me a hot pair of brown boots  Woulda rathered seeing U any day  
   Predykcja: Pozytywna, Rzeczywista etykieta: Negatywna

 - Tweet: It's fine. Today sucks just because me those things. i dunno if i can see you 
   Predykcja: Pozytywna, Rzeczywista etykieta: Pozytywna

 - Tweet: Im just chilling on psp and stuff, but sitting on pc now, also watching wimledon, getting ready for holiday @WhiteTigerNora Ahh poor you  
   Predykcja: Negatywna, Rzeczywista etykieta: Negatywna

 - Tweet: @lisarinna very sad Lisa...she is freeeeeeeeeeee an Angel in Heaven xoxo  
   Predykcja: Negatywna, Rzeczywista etykieta: Negatywna

 - Tweet: Comfortablity has won out  
   Predykcja: Negatywna, Rzeczywista etykieta: Pozytywna

 - Tweet: blaaah. I don't feel good aagain  
   Predykcja: Negatywna, Rzeczywista etykieta: Pozytywna

**W projekcie przeprowadzono także unit testy, które sprawdzają różne etapy przetwarzania danych oraz trenowania modelu:**

1. Tokenizacja i dopełnianie sekwencji: Testy sprawdzają, czy sekwencje mają odpowiedni kształt i długość oraz czy każdy token jest liczbą całkowitą.
2. Podział danych: Testy weryfikują poprawność podziału danych na zbiory treningowy i walidacyjny.
3. Struktura plików JSONL: Testy sprawdzają, czy pliki JSONL są tworzone z odpowiednią strukturą.

Wszystkie testy przeszły pomyślnie, co potwierdza poprawność przetwarzania danych oraz trenowania modelu.


## Wymagania

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Datasets (Hugging Face)
