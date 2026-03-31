# Artykuł 1. – Sanity Ckecks for Saliency Maps

- link do artykułu: [https://arxiv.org/pdf/1810.03292](https://arxiv.org/pdf/1810.03292),
- link do kodu eksperymentów: [https://github.com/adebayoj/sanity_checks_saliency](https://github.com/adebayoj/sanity_checks_saliency).

## Problem
* istnieje wiele metod atrybucji, saliency – wskazują one które fragmenty danych wpłynęły na decyzję algorytmu AI,
* wyjaśnienia mogą pomóc w debugowaniu modelu, wykryć bias lub niezamierzone zachowanie wyuczone przez model,
* jednak nie wiadomo dokładnie w jaki sposób mierzyć jakość tych wyjaśnień – czy są poprawne, czy tylko dobrze wyglądają:
    * niektóre popularne metody XAI działają w sposób niezależny od tego czego nauczył się model oraz niezależny od samych danych,
    * niektóre mapy atrybucji wyglądają niemal identycznie jak wynik zwykłego detektora krawędzi – model zamiast faktycznie identyfikować obiekt, może po prostu wyciągać z obrazka jego strukturę.

## Metodologia zaproponowana przez autorów
* autorzy proponują metodologię opartą na testach randomizacyjnych, a dokładnie dwie instancje testów:
    - Model Parameter Randomization Test (MPRT),
    - Data Randomization Test (DRT),
* jeżeli okaże się, że jakaś metoda jest niezmienna względem wag, to zastosowanie tej metody w celu np. debugowania modelu będzie bez sensu.

### Model Parameter Randomization Test

- porównywane są wyniki metod saliency na wytrenowanym modelu, z wynikami tych metod na modelach tej samej architektury o zrandomizowanych parametrach,
- jeżeli dana metoda saliency zależy od wyuczonych parametrów modelu, wynik powinien się znacząco różnić,
- jeżeli wyniki będą podobne – można wnioskować, że dana metoda nie jest zależna od parametrów modelu, że nie nada się do zadań wyjaśniania modelu,
- autorzy przeprowadzają dwa rodzaje randomizacji:
    - kaskadową:
        - zaczynają od wytrenowanego modelu i reinicjalizują wagi kaskadowo od góry do dołu, od pierwszej warstwy do ostatniej,
        - na każdym kroku obliczają mapę atrybucji i porównują ją z oryginałem,
        - otrzymują w ten sposób krzywą degradacji – jak mapa się zmienia w miarę niszczenia coraz większej części modelu,
        - jeśli metoda jest wrażliwa, krzywa powinna stopniowo spadać do 0,
    - niezależną:
        - reinicjalizują jedną warstwę na raz, reszta pozostaje nietknięta,
        - po randomizacji danej warstwy wyznaczają mapę i porównują ją z oryginałem,
        - pozwala to wyizolować wpływ poszczególnych warstw, można zobaczyć, czy mapa atrybucji zależy bardziej od warstw niższych, czy wyższych,
- stosowane metryki do porównywania map:
    - korelacja rangowa Spearmana,
    - HOGs – histogram of oriented gradient,
    - SSIM – structural similarity index,
- wyniki eksperymentów pokazały, że Guided BackProp i Guided GradCAM nie zmieniają się niezależnie od degradacji modelu,


### Data Randomization Test
- porównuje daną metodę saliency wykorzystaną na modelu wytrenowanym na poprawnie zaetykietowanych danych z tą samą metodą wykorzystaną na modelu o tej samej architekturze, ale wyuczonym na kopii zbioru treningowego z losowo pozamienianymi etykietami,
- jeżeli dana metoda zależy od poprawnego etykietowania – wyniki powinny się znacząco różnić niż w przypadku gdy etykiety będą pozamieniane,
- eksperyment ten ewaluuje czułość danej metody XAI na relację pomiędzy przykładami a ich etykietami,
- autorzy losowo permutują etykiety w zbiorze treningowym i trenują model od zera aż zacznie uzyskiwać 95% dokładności na zbiorze treningowym, aż model na pamięć zapamięta większość błędnych etykiet, 
- następnie porównują uzyskane mapy wyjaśnień na poprawnie wytrenowanym modelu i tym wytrenowanym na losowych etykietach.

## Modele użyte w eksperymentach
- Inception v3 wytrenowany na ImageNet,
- CNN wytrenowany na MNIST i Fashion MNIST,
- MLP wytrenowany na MNIST,
- Inception v4 wytrenowany ma Skeletal Radiograms.

## Najważniejsze wnioski
- architektura sieci neuronowej ma duży wpływ na reprezentacje pochodzące z sieci:
    - sama architektura może wymuszać na modelu szukania lokalnych wzorców jak np. CNN,
    - nawet jeśli wagi są losowe CNN działa jak filtr, który przepuszcza określone kształty,
- losowo zainicjalizowana sieć nie jest całkowicie głupia:
    - posiada pewne wstepne założenia / predyspozycje (priors) do rozpoznawania określonych struktur,
    - jest w stanie generować nietrywialne reprezentacje tego co widzi, a nie tylko losowy szum,
    - można jej użyć do pewnych zadań takich jak odszumianie czy super resolution,
- wyjaśnienia które nie zależą od parametrów modelu czy danych treningowych nadal mogą być przydatne do zrozumienia priors wynikających z samej architektury modelu,
- wiele metod z rodziny saliency sprowadza się do przemnożenia wejściowego obrazu przez gradient:
    - okazuje się, że obraz wejściowy dominuje gradient w tym mnożeniu,
    - metody typu LRP mogą w rzeczywistości zwracać po prostu lekko zmodyfikowany obrazek wejściowy, a nie wyjaśnienie decyzji modelu.

# Artykuł 2. - A Fresh Look at Sanity Checks for Saliency Maps

- link do artykułu: [https://arxiv.org/pdf/2405.02383](https://arxiv.org/pdf/2405.02383),
- link do kodu eksperymentów: [https://github.com/annahedstroem/sanity-checks-revisited](https://github.com/annahedstroem/sanity-checks-revisited).

## Motywacja powstania artykułu

- od czasu zaproponowania MPRT pojawiło się wiele prac podważających nieco metodologię testów:
   - miary podobieństwa użyte do porównywania,
   - kolejność randomizacji kolejnych warstw,
   - wstępne przetwarzanie wyników metod saliency,
   - wpływ konkretnego modelu i zadania na wynik testu.
- te techniczne detale mogą całkowicie zaburzać wyniki badania,
- autorzy proponują więc dwa nowe warianty MPRT:
    - Smooth MPRT,
    - Efficient MPRT.

## Problemy oryginalnej metodologii

### Pre-processing
- w oryginalnej implementacji, każda mapa atrybucji normalizowana była do przedziału [0, 1] przez min-max, 
jeżeli więc na danym obrazie jakiś jeden piksel miałby przypadkowo bardzo dużą wartość, to cała reszta mapy po normalizacji stanie się prawie zerowa
- różne metody saliency generują wartości w zupełnie innych skalach, ściskanie ich na siłę do przedziału [0, 1] sprawia, że traci się informacje o tym jak silna była pierwotna atrybucja, utrudnia to porównywanie różnych metod saliency
- w oryginalnych testach na mapach używana jest też wartość bezwzględna, co może usuwać istotne informacje – informacja o znaku może być kluczowa

### Kolejność randomizacji wag
- adebayo założył, że jeśli zepsuje się ostatnią warstwę, to całe wyjaśnienie uzyskane przez metodę saliency powinno być zepsute,  jednak wiele badań dowiodło że niekoniecznie,
- dolne warstwy, z poprawnymi wagami, dalej będą ekstrahować silne cechy, mimo że góra jest zrandomizowana, to te przetworzone informacje i tak będą w jakimś stopniu narzucać strukturę w wyższych losowych warstwach,
- jeżeli pierwsza warstwa bardzo silnie zareaguje na dany fragment, to ta aktywacja przejdzie przez zrandomizowane warstwy jak taran, losowe wagi mogą ją ewentualnie trochę osłabić, ale ona dalej będzie dominować,
- jeżeli w danej architekturze są połączenia rezydualne, to nawet mimo zrandomizowania ostatnich warstw, informacja z początku przejdzie, przez co oczekiwana różnica pomiędzy mapami nie będzie tak widoczna, co sugerowałoby że metoda nie jest wierna, a nie musi tak być.

### Wykorzystane metryki podobieństwa
- w głębokich sieciach gradienty mają tendencję do bycia bardzo poszarpanymi i chaotycznymi, wyglądają jak drobny, gęsty szum,
- metryki użyte w oryginalnej metodzie: korelacja Spearmana i SSIM są bardzo czułe na drobne, nieskorelowane zmiany,
- jeżeli metoda XAI naturalnie generuje dużo szumu, to po zrandomizowaniu wag modelu ten szum staje się jeszcze bardziej chaotyczny, przez co metryki natychmiast spadają do zera,
- daje to błędne wnioski, że dana metoda faktycznie jest dobra skoro korelacja spadła, a to po prostu dana metoda jest z natury zaszumiona, co daje nieuczciwą przewagę metod gradientowych nad innymi.

## Proponowane przez autorów usprawnienia

### Smooth Model Parameter Randomization Test

- rozwiązuje problem w poszarpanym szumem w oryginalnej metodyce MPRT, która faworyzowała metody z szumem przez użycie metryk SSIM i korelacji Spearmana,
- metoda sMPRT wprowadza krok wstępnego przetwarzania, który oczyszcza wyjaśnienia przed ich porównaniem,
- dla jednego konkretnego wejścia, algorytm tworzy N dodatkowych kopii, do których dodaje mały, losowy szum gaussowski (eksperymentalnie wyszło, że N=50 daje najlepsze wyniki),
- dla każdej z tych N zaszumionych kopii obliczana jest osobna mapa atrybucji,
- wszystkie te mapy są sumowane i dzielone przez N, co pozwala uzyskać jedną, wygładzoną mapę atrybucji, pozbawioną przypadkowego szumu,
- następnie postępuje się podobnie jak w oryginalnym MPRT – porównuje średnią atrybucję z modelu bazowego ze średnią atrybucją z modelu, którego wagi są stopniowo niszczone, ale z **dołu do góry** – skutecznie przerywa to przepływ informacji strukturalnej z oryginalnego sygnału.

### Efficient Model Parameter Randomization Test

- metoda zamiast porównywać dwie mapy atrybucji oryginalną i zrandomizowaną za pomocą zawodnych miar podobieństwa, bada zmianę złożoności samego wyjaśnienia, mierząc strukturę mapy
- głównym narzędziem w eMPRT jest entropia dyskretna – służy jako miara złożoności wyjaśnienia:
    - wiarygodne wyjaśnienie oryginalnego modelu powinno być skupione i ustrukturyzowane – niska entropia
    - po randomizacji model traci zdolność sensownej argumentacji, więc jego wyjaśnienia powinny stać się chaotyczne i informacyjnie puste – wysoka entropia

$$\hat{q}^{eMPRT} = \frac{\xi(\hat{e}) - \xi(e)}{\xi(e)}$$

- interpretacja wyrażenia:
    - wartość dodatnia oznacza wzrost złożoności wyjaśnienia po randomizacji – metoda jest czuła na parametry modelu,
    - wartość ujemna – spadek złożoności,
    - wartość równa 0 – brak zmiany złożoności wyjaśnienia, niewrażliwość metody na stan modelu.
- autorzy używają entropii dyskretnej opartej na histogramie:
    - wartości atrybucji z mapy wyjaśnienia dzielone są na B równych przedziałów zwanych binami (autorzy użyli B = 100),
    - dla każdego binu wyliczana jest częstotliwość występowania danej wartości ci, następnie jest ona normalizowana aby uzyskać rozkład prawdopodobieństwa dla każdego przedziału.
- zalety tej metody obliczeń:
    - pozwala zachować informację o znaku – czy fragment miał wpływ pozytywny czy negatywny,
    - metoda sama w sobie skaluje dane, eliminuje błędy ręcznego ustawiania zakresów min-max,
    - uniwersalność – można tą metodę stosować do różnych architektur.



