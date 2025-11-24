# Opis Projektu: AMD-DBSCAN vs DBSCAN

## Opis problemu

Klasteryzacja to podstawowe zadanie w analizie danych, polegające na grupowaniu podobnych punktów. Algorytmy oparte na gęstości, szczególnie DBSCAN, są popularne, ponieważ potrafią wykrywać klastry o dowolnym kształcie oraz identyfikować punkty szumowe. 
Jednak klasyczny DBSCAN ma poważne ograniczenia:


1. **Trudność doboru parametrów**: DBSCAN wymaga dwóch parametrów (Eps i MinPts), które są trudne do określenia, szczególnie w przestrzeniach o wysokiej wymiarowości, gdzie wizualizacja jest niemożliwa.

2. **Ograniczenie do jednej gęstości**: Tradycyjny DBSCAN używa jednej pary parametrów (Eps, MinPts) dla całego zbioru danych. Działa to dobrze dla danych o jednolitej gęstości, ale zawodzi na **zbiorach wielogęstościowych**, gdzie różne klastry mają znacznie różne gęstości.

   - Jeśli parametry dobrane są pod klastry gęste — rzadkie klastry błędnie uznawane są za szum  
   - Jeśli parametry dobrane są pod klastry rzadkie — gęste klastry łączą się w jeden

## Istniejące rozwiązania

W literaturze zaproponowano kilka podejść do rozwiązania tych ograniczeń:

### Metody adaptacji parametrów
- **ROCKA** i **PDDBSCAN**: Metody adaptujące automatycznie Eps lub MinPts, ale nie oba jednocześnie  
- **ISB-DBSCAN**: Redukuje liczbę parametrów wejściowych, lecz nadal wymaga ręcznego strojenia  
- **AA-DBSCAN**: Zapewnia przybliżone adaptacyjne Eps dla danych wielogęstościowych  

### Metody wielogęstościowe
- **YADING**: Iteracyjnie klasteryzuje najpierw obszary o wysokiej gęstości, potem pozostałe punkty. Wymaga jednak stałego k i dodatkowych hiperparametrów  
- **AEDBSCAN**: Ulepsza analizę krzywej k-distance za pomocą różnic drugiego rzędu  
- **VDBSCAN**: Automatycznie dostosowuje parametry do różnych regionów gęstości  
- **HDBSCAN**: Podejście hierarchiczne działające na danych wielogęstościowych, ale o długim czasie działania  

**Ograniczenia istniejących metod:**
- Większość adaptuje tylko jeden parametr (Eps LUB MinPts)  
- Wymagają dodatkowych hiperparametrów ustawianych ręcznie  
- Niektóre mają wysoką złożoność obliczeniową  
- Dokładność na ekstremalnych zbiorach wielogęstościowych jest nadal ograniczona  

## Nowe podejście: AMD-DBSCAN

AMD-DBSCAN (Adaptive Multi-density DBSCAN) rozwiązuje te problemy dzięki trzem kluczowym innowacjom:

### 1. Adaptacyjny dobór parametrów (Eps i MinPts jednocześnie)

Zamiast ręcznego dobierania parametrów, AMD-DBSCAN:
- Generuje listy kandydatów Eps i MinPts na podstawie rozkładu przestrzennego danych  
- Używa wyszukiwania binarnego do znalezienia optymalnej pary parametrów  
- Automatycznie wyznacza adaptacyjną wartość `k`, która steruje procesem klasteryzacji  

**Kluczowa idea**: algorytm znajduje pierwszą „stabilną” strefę, gdzie liczba klastrów pozostaje stała, a następnie wyszukiwanie binarne lokalizuje najlepszą parę parametrów w tej strefie.

### 2. Odkrywanie kandydatów Eps dla danych wielogęstościowych

Dla zbiorów danych o wielu gęstościach AMD-DBSCAN:
- Używa adaptacyjnego `k` do obliczenia wartości k-distance dla wszystkich punktów  
- Tworzy histogram częstości tych wartości  
- Wykrywa lokalne maksima (różne poziomy gęstości)  
- Stosuje K-means na wartościach k-distance, aby automatycznie znaleźć wartości Eps dla różnych poziomów gęstości  

**Zaleta**: wystarczy obserwacja liczby pików w histogramie (lub może być wykryta automatycznie), bez skomplikowanego strojenia hiperparametrów.

### 3. Iteracyjna klasteryzacja wielogęstościowa

Algorytm klasteryzuje dane warstwowo:
- Sortuje kandydatów Eps rosnąco  
- Dla każdego Eps oblicza adaptacyjne MinPts na podstawie aktualnych punktów nieprzypisanych  
- Uruchamia DBSCAN tylko dla niezaklasteryzowanych punktów  
- Usuwa przypisane punkty i przechodzi do kolejnego poziomu gęstości  
- Pozostałe punkty oznacza jako szum  

**Efekt**: Każdy poziom gęstości otrzymuje własną zoptymalizowaną parę parametrów, co umożliwia prawidłową identyfikację zarówno gęstych, jak i rzadkich regionów.

## Kluczowe zalety

1. **W pełni automatyczny**: wymaga jedynie danych wejściowych (opcjonalnie liczby pików)  
2. **Adaptuje oba parametry**: w przeciwieństwie do większości metod  
3. **Wydajny**: wyszukiwanie binarne skraca czas strojenia parametrów o ~75% względem metod wyczerpujących  
4. **Wysoka dokładność**: średnio +24.7% na zbiorach wielogęstościowych, przy zachowaniu jakości na zbiorach jednorodnych  
5. **Odporny**: działa dobrze na danych o ekstremalnie zmiennej gęstości (VNN > 100)  

## Szczegóły techniczne

### Metryka VNN

Artykuł wprowadza VNN (Variance of Number of Neighbors) do pomiaru zróżnicowania gęstości:
- Niskie VNN (< 10): zbiór jednogęstościowy  
- Wysokie VNN (> 10): zbiór wielogęstościowy  
- Bardzo wysokie VNN (> 100): ekstremalny zbiór wielogęstościowy  

### Złożoność algorytmu

- Adaptacja parametrów: O(log(n) × f(n)), gdzie f(n) to złożoność DBSCAN  
- Odkrywanie kandydatów Eps: O(n)  
- Klasteryzacja wielogęstościowa: O(f(n))  
- Całość: O(log(n) × f(n))  

## Porównanie z klasycznym DBSCAN

| Aspekt | DBSCAN | AMD-DBSCAN |
|--------|--------|------------|
| Parametry | Ręczne (Eps, MinPts) | Automatyczne (opcjonalnie: liczba pików) |
| Jedna gęstość | Dobre wyniki | Dobre (porównywalne) |
| Wiele gęstości | Słabo | Doskonale (+24.7%) |
| Strojenie parametrów | Wymagane | Niepotrzebne |
| Czas wykonania | Szybki | Umiarkowany (adaptacja parametrów) |

## Wnioski

AMD-DBSCAN to znaczący krok naprzód w klasteryzacji opartej na gęstości, dzięki automatycznej adaptacji do danych o różnych poziomach gęstości. Usuwa potrzebę ręcznego doboru parametrów i zapewnia lepsze wyniki na trudnych zbiorach wielogęstościowych, co czyni go bardziej praktycznym w zastosowaniach rzeczywistych, gdzie charakterystyka danych nie jest znana z góry.
