# OLX_Projects

# Project 1. System Rekomendacyjny dla OLX Praca

Projekt ten polega na budowie i ewaluacji systemu rekomendacyjnego w oparciu o dane OLX Jobs. Głównym celem jest zastosowanie różnych podejść rekomendacyjnych, takich jak filtrowanie kolaboracyjne i techniki zaawansowane (Graph Neural Networks, ALS, KNN), w celu poprawy trafności rekomendacji.

## Dane

Zestaw danych pochodzi z Kaggle:  
[OLX Jobs Interactions Dataset](https://www.kaggle.com/datasets/olxdatascience/olx-jobs-interactions)  
Zawiera **65 milionów interakcji użytkowników** z ogłoszeniami o pracę.

### Przetwarzanie danych
1. **Wybrany typ zdarzeń**: Uwzględniamy tylko interakcje z `event = 'click'`.
2. **Podzbiór danych**:
   - Wybrano **10% danych** na podstawie użytkowników o największej liczbie interakcji.
   - Dodatkowe próbkowanie x% użytkowników i przedmiotów (np. x=1%) w celu dostosowania rozmiaru danych do możliwości obliczeniowych.
3. **Podział na zbiór treningowy i testowy**:
   - **Temporal Split**: 80% wcześniejszych interakcji w zbiorze treningowym i 20% ostatnich interakcji w zbiorze testowym (sortowanie po `timestamp`).

### Specyfikacja sprzętu
Testy przeprowadzono na maszynach wyposażonych w:
- **RAM**: 32 GB
- **Procesor**: Wysokowydajne CPU (konfiguracja zależna od metody).

## Metody rekomendacyjne

### Podejścia zastosowane
1. **Collaborative Filtering**:
   - **Item-Item KNN**: Rekomedacje na podstawie podobieństwa między przedmiotami.
   - **ALS (Alternating Least Squares)**: Analiza ukrytych czynników użytkowników i przedmiotów.
2. **Graph Neural Networks (GraphSAGE)**:
   - Model wykorzystujący strukturę grafową relacji między użytkownikami i przedmiotami.

### Wyniki i analizy
| Metoda         | Rozmiar danych | Precision@k | Recall@k | Czas obliczeń | Uwagi                                   |
|----------------|----------------|-------------|----------|---------------|-----------------------------------------|
| **GraphSAGE**  | 25% danych     | 0.0004      | 0.0004   | ~15-20 min    | Bardzo szybkie obliczenia, trudna implementacja. |
| **ALS**        | 10% danych     | 0.0344      | 0.0642   | ~7 godzin     | Dobre wyniki jakościowe, długi czas obliczeń. |
| **Item-Item KNN** | 100% danych | 0.024       | 0.012    | ~6 godzin     | Skuteczność niższa niż oczekiwana na pełnych danych. |

### Szczegółowe obserwacje
1. **GraphSAGE**:
   - Pomimo niskich wyników metrycznych, metoda wyróżnia się szybkim czasem obliczeń.
   - Wymaga dalszego zrozumienia teoretycznego i dopracowania implementacji.
2. **ALS**:
   - Zapewnia lepszą jakość rekomendacji w porównaniu z GraphSAGE.
   - Wysoki koszt czasowy obliczeń przy dużych zbiorach danych.
3. **KNN**:
   - Wyniki na małych próbkach (np. 0.1% danych) były znacznie lepsze niż na pełnym zbiorze (0.02 vs. 0.00055 Precision@k).
   - Skalowanie na większych zbiorach danych wymaga dalszych analiz i dostrojenia parametrów.

## Ewaluacja

### Metryki
- **Precision@k**: Odsetek rekomendacji, które są trafne.
- **Recall@k**: Odsetek trafnych przedmiotów z całkowitej liczby dostępnych trafnych.
- **RMSE**: Dla algorytmów predykcji ocen, różnica między rzeczywistą a przewidywaną oceną.
- **Inne**:
  - **NDCG (Normalized Discounted Cumulative Gain)**: Mierzy jakość rankingu.
  - **MAP (Mean Average Precision)**: Średnia precyzja na całej liście rekomendacji.

## Wnioski
1. GraphSAGE wymaga dalszego dopracowania teoretycznego i implementacyjnego.
2. ALS jest obecnie najbardziej obiecującą metodą dla tego zbioru danych, mimo dużego czasu obliczeń.
3. Skalowanie metody KNN na większych zbiorach nie przyniosło oczekiwanych wyników, ale poprawione implementacje mogą zwiększyć skuteczność w przyszłości.

# Project 2. Analiza Cen Nieruchomości

Celem jest analiza cen nieruchomości w Poznaniu oraz opracowanie modeli regresyjnych do przewidywania cen mieszkań. Projekt zawiera kroki od analizy eksploracyjnej danych (EDA) po budowę i ocenę modeli predykcyjnych.

## Zawartość

### Dane
- Dane pierwotne dotyczące cen mieszkań na rynku wtórnym w Poznaniu.
- Pliki CSV przygotowane na różnych etapach preprocessingu:
  - **real_estate_imputed**: Dane przefiltrowane i uzupełnione brakujące wartości.
  - **real_estate_imputed_geo**: Dane wzbogacone o cechy geograficzne (np. odległości do obiektów takich jak szkoły czy sklepy).

### Skrypty
Projekt składa się z następujących plików:
1. **Przygotowanie_zbioru**:
   - Analiza eksploracyjna danych (EDA) z wykorzystaniem Ydata profiling.
   - Generowanie przetworzonych danych wejściowych (`real_estate_imputed.csv`).
2. **Dane_geograficzne**:
   - Analiza cen na mapie Poznania.
   - Generowanie danych z cechami geograficznymi (`real_estate_imputed_geo.csv`).
3. **Model_AutoGluon**:
   - Model AutoGluon na danych surowych (`real_estate_imputed`).
4. **Model_AutoGluon_geo**:
   - Model AutoGluon z cechami geograficznymi (`real_estate_imputed_geo`).
5. **Model_bazowy_liniowy_RF**:
   - Modele regresji liniowej (Lasso, Ridge) i lasów losowych.
6. **GEO_model_bazowy_liniowy_RF**:
   - Modele regresji liniowej i lasów losowych z cechami geograficznymi.
7. **RandomForest**:
   - Modele lasów losowych z cechami tekstowymi i geograficznymi.
8. **XGBoost**:
   - Modele XGBoost z cechami tekstowymi i geograficznymi.

### Dostęp do projektu
Pliki projektowe zostały umieszczone na prywatnej chmurze:  
[Link do projektu](https://drive.google.com/file/d/13DsQJJMLKhdn3ORfEe4qxIbGDftVXBlW/view?usp=sharing)  

## Metodologia

1. **Analiza Eksploracyjna Danych (EDA)**:
   - Wykorzystano Ydata profiling do wstępnej analizy danych.
2. **Modelowanie**:
   - Model bazowy: dane z podstawowymi parametrami, np. lokalizacja geograficzna (GeoPy).
   - Modele z cechami tekstowymi: przetwarzanie tytułu i opisu za pomocą metod:
     - TF-IDF
     - FastText
     - BERT
     - BertTopic
     - GPT
3. **Podział danych**:
   - 20% zbioru przeznaczono na zbiór testowy.
   - Stratyfikowany podział z koszykami wartości zmiennej objaśnianej.

## Wyniki

| Model                                | Dane wejściowe             | RMSE na zbiorze testowym | Uwagi                                      |
|--------------------------------------|----------------------------|--------------------------|--------------------------------------------|
| **Model bazowy (RandomForest)**      | real_estate_imputed        | 740.12                  | Brak cech geograficznych.                 |
| **Model bazowy z cechami geograficznymi** | real_estate_imputed_geo   | 710.78                  | Wzbogacenie danych o cechy geograficzne.  |
| **XGBoost (pełny model)**            | Wszystkie dane             | **670.57**              | Najlepszy wynik dla modeli manualnych.    |
| **AutoGluon**                        | Wszystkie dane             | **271.76**              | Automatyczna optymalizacja, najlepszy wynik. |

### Dodatkowe informacje
- **Średnia wartość zmiennej objaśnianej**:
  - Zbiór treningowy: 7020.02
  - Zbiór testowy: 7021.93
- **Rozkład danych**:
  - Losowy podział z zachowaniem proporcji cen poprzez koszyki.

## Wnioski

1. **Najlepszy model manualny**: XGBoost (RMSE: 670.57), szczególnie skuteczny w analizie danych wzbogaconych o cechy tekstowe i geograficzne.
2. **Model automatyczny**: AutoGluon (RMSE: 271.76), który przewyższył wszystkie modele manualne pod względem dokładności.
3. **Zalecenia**:
   - W dalszych pracach warto skupić się na optymalizacji cech geograficznych i wykorzystaniu modeli automatycznych.
