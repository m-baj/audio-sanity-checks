# Design proposal

## Cel projektu

Głównym celem projektu jest systematyczna weryfikacja wiarygodności metod atrybucji (takich jak Grad-CAM, Integrated Gradients, SmoothGrad, LRP oraz Guided Backpropagation) w kontekście klasyfikacji sygnałów audio reprezentowanych jako spektrogramy. Projekt dąży do odpowiedzi na pytanie, czy popularne techniki wyjaśnialnej sztucznej inteligencji faktycznie obrazują wyuczoną wiedzę modelu, czy jedynie powielają silną strukturę czasowo-częstotliwościową sygnału wejściowego

## Bibliografia

* [Adebayo et al. (2018) Sanity Checks for Saliency Maps, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf),
* [Hedström et al. (2024) A Fresh Look at Sanity Checks for Saliency Maps](https://arxiv.org/pdf/2405.02383),
* [Paissan et al. (2024) Listenable Maps for Audio Classifiers, ICML](https://arxiv.org/pdf/2403.13086),
* [Kong et al., (2020) PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211)

TODO: poszukać więcej artykułów

## Planowany zakres eksperymentów

TODO: Zweryfikować / dodać

* przygotowanie danych wejściowych,
* trenowanie wybranych modeli klasyfikacyjnych dla przygotowanych danych wejściowych,
* analiza map atrybucji (Grad-CAM, Integrated Gradients, SmoothGrad, LRP Guided Backpropagation) dla modeli klasyfikujących audio na podstawie spektrogramów,
* porównanie zachowania metod dla modelu poprawnie wytrenowanego, zrandomizowanego oraz uczonego na losowych etykietach,
* ocena wyników zarówno jakościowo, jak i ilościowo z użyciem metryk podobieństwa map atrybucji,
* weryfikacja, czy metody wyjaśniające odzwierciedlają decyzje modelu

## Planowany stack technologiczny

* **Język:** Python 3.12+
* **Menedżer pakietów:** `uv`
* **Przetwarzanie audio:** `torchaudio`, `librosa`,
* **Deep Learning:** `PyTorch`, `PyTorch Lightning`, `timm` (Hugging Face)
* **Metody XAI:** `Captum`, `pytorch-grad-cam`
* **CLI & Logi:** `typer`, `loguru`, `tqdm`
* **Wizualizacja:** `matplotlib`, `seaborn`

## Funkcjonwalności

TODO: Zweryfikować

Przetwarzanie danych i Pipeline
- Automatyczny preprocessing audio: Konwersja surowych plików dźwiękowych (WAV/MP3) na znormalizowane spektrogramy melowe o stałych wymiarach.
- Wieloklasowa klasyfikacja sygnałów: Rozpoznawanie kategorii dźwięków przy użyciu zaawansowanych sieci neuronowych.
- Trening wariantowy: Możliwość uczenia modelu w trzech trybach: na poprawnych danych, na danych z losowymi etykietami oraz praca na wagach całkowicie zrandomizowanych.

Weryfikacja i Diagnostyka (Sanity Checks)
- Kaskadowe niszczenie wag: Funkcja stopniowego resetowania parametrów sieci (od końca do początku) w celu sprawdzenia, czy mapy XAI faktycznie przestają być czytelne.
- Automatyczne porównywanie map: Ilościowe wyliczanie podobieństwa między wyjaśnieniami (np. czy mapa dla modelu „pustego” różni się od mapy modelu wyuczonego).

Raportowanie i Obsługa
- Zarządzanie z poziomu terminala (CLI): Pełna kontrola nad eksperymentami (trening, generowanie map, testy Adebayo) za pomocą prostych komend tekstowych.
- Eksport wyników: Zapisywanie przetworzonych map, metryk i logów w ustrukturyzowanych formatach (obrazy, pliki CSV/JSON).
## Harmonogram prac

TODO: Zweryfikować

### Tydzień 1: 23 marca – 29 marca
* **Cel:** Przygotowanie i dostarczenie *Design Proposal*.
* Finalizacja opisu celu projektu i zakresu eksperymentów.
* Dobór ostatecznej bibliografii.
* Zdefiniowanie stacku technologicznego.
* Dostarczenie pliku `design_proposal.md` do repozytorium.
* Analiza literatury.

### Tydzień 2: 30 marca – 5 kwietnia
* **Cel:** Konfiguracja środowiska i **Deadline na prototyp (02.04.2026, czwartek)**.
* Konfiguracja środowiska eksperymentalnego przy użyciu menedżera `uv`.
* Przygotowanie skryptów do pobierania i preprocessingu danych (ESC-50, Speech Commands).
* Demonstracja funkcjonalnego prototypu: wczytywanie audio, generowanie spektrogramu i przejście danych przez model.
* Udokumentowanie postępu analizy literaturowej w formie tabeli `.md`.

### Tydzień 3: 6 kwietnia – 12 kwietnia
* **Cel:** Implementacja potoku klasyfikacji i baseline XAI.
* Implementacja pętli trenującej dla klasyfikatorów audio (lub skryptów fine-tuningu dla modeli SOTA z HuggingFace).
* Uruchomienie pierwszej metody atrybucji (np. Saliency) na poprawnie działającym modelu.
* Wstępne testy na mniejszych fragmentach danych.

### Tydzień 4: 13 kwietnia – 19 kwietnia
* **Cel:** Rozpoczęcie testów Adebayo – Random Parameter Test.
* Implementacja skryptu do kaskadowej randomizacji wag modelu (od ostatniej warstwy do wejścia).
* Generowanie map atrybucji dla modelu zrandomizowanego.

### Tydzień 5: 20 kwietnia – 26 kwietnia
* **Cel:** Trening modeli do testu Random Label Test.
* Przygotowanie wariantu datasetu z losowo pomieszanymi etykietami.
* **Trenowanie modeli:** Rozpoczęcie procesu uczenia modeli na "szumiących" etykietach (wymaga znacznych zasobów GPU – planowane wykorzystanie klastra/Google Colab).

### Tydzień 6: 27 kwietnia – 3 maja
* **Cel:** Generowanie i gromadzenie wyników atrybucji.
* Uruchomienie wszystkich 5 metod atrybucji na 3 stanach modeli (Trained, Randomized, Label-shuffled).
* Zbiór wyników wizualnych i metryk ilościowych (np. korelacja między mapami).
* **Zasoby obliczeniowe:** Intensywne generowanie map atrybucji (wymaga dużej pamięci VRAM).

### Tydzień 7: 4 maja – 10 maja
* **Cel:** Analiza porównawcza i ranking metod.
* Opracowanie rankingu wiarygodności metod specyficznego dla audio.
* Analiza wpływu struktury spektrogramu na "oszukiwanie" sanity checks.
* **Standup 2 (zgodnie z obrazkiem – tydzień 8 semestru):** Gotowość do zreferowania wyników pośrednich.

### Tydzień 8: 11 maja – 17 maja
* **Cel:** Finalizacja eksperymentów i dokumentacja techniczna.
* Ostatnie poprawki w kodzie i testy jednostkowe skryptów.
* Przygotowanie instrukcji użytkowania (README.md) oraz dokumentacji końcowej.
* Weryfikacja, czy wszystkie wymagane elementy są kompletne.

### Tydzień 9: 18 maja – 24 maja
* **Cel:** Przygotowanie materiałów końcowych i **Termin zwolnienia projektów (21.05.2026, czwartek)**.
* Nagranie 3-5 minutowego filmu demonstrującego projekt (screencast z wynikami i omówieniem rankingu).
* Przygotowanie prezentacji na finałowy wykład.

### Tydzień 10: 25 maja – 2 czerwca
* **Cel:** Finalna prezentacja i **Deadline finalny (02.06.2026, wtorek)**.
* Ostateczne sprawdzenie repozytorium.
* Publiczna prezentacja projektu na ostatnim wykładzie.
* Oddanie kompletu plików (kod, dokumentacja, film, prezentacja).

---

<!-- ### **Przewidywane zapotrzebowanie na zasoby obliczeniowe:**
1.  **Trenowanie modeli (Random Label Test):** Ok. 20-30 godzin pracy procesora GPU (klasy RTX 3060 lub lepszego) na każdy zbiór danych, aby osiągnąć zbieżność przy losowych etykietach.
2.  **Generowanie map atrybucji:** Ok. 10-15 GB wolnej pamięci VRAM dla modeli typu Transformer (AST) oraz ok. 5 GB dla modeli typu ResNet.
3.  **Przestrzeń dyskowa:** Ok. 20 GB na dane surowe (RAW), spektrogramy oraz checkpointy modeli. -->
