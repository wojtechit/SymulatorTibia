# Symulator Tibia – polowanie Druida

Jednoplikowa aplikacja PyQt5: rysujesz trasę na mapie, a Druid chodzi po niej w pętli,
walczy z Giant Spiderami, leczy się, pije mikstury i (opcjonalnie) trzyma Utamo Vita.

## Uruchomienie

```bash
pip install -r requirements.txt
python3 tibia_simulator.py
```

Opcjonalnie połóż `map.png` obok skryptu (albo użyj „Wczytaj mapę…”). Bez mapy jest ciemnoszare tło z siatką 32 px = 1 sqm.

## Sterowanie mapą

| Akcja | Efekt |
|---|---|
| LPM | dodaj punkt trasy |
| PPM | usuń ostatni punkt |
| ŚPM + przeciągnij | przesuwanie mapy |
| Kółko | zoom |

Trasa jest pętlą (przerywana linia ostatni → pierwszy). Edycja jest zablokowana w trakcie symulacji – użyj „Reset”.

## Mechanika (tick 500 ms)

- **HP** = `145 + (lvl-8)*5`, **Mana** = `90 + (lvl-8)*30`, prędkość = `110 + 2*(lvl-1)`.
- **Utamo Vita** (50 many, 200 s): obrażenia idą najpierw w manę; gdy mana spadnie do 0, tarcza pęka, a reszta trafia w HP. Automatycznie odnawiana.
- **Exura Gran** (70 many, CD 1 s) gdy HP < 60%.
- **Great Mana Potion** (120–160, CD 1 s) gdy mana < 50% (z aktywną tarczą < 85%).
- **Exori Frigo** (20 many, CD 2 s) w najsłabszego potwora w zasięgu 5 sqm.
- **Giant Spider**: 1300 HP, obrażenia 0–300 co 2 s, aggro 7 sqm, 900 exp, respawn po 60 s.
- Regeneracja: +1 HP i +4 many co 2 s.

Silnik (`Druid`, `Monster`, `HuntSimulation`) nie zależy od Qt – można go testować bez GUI.

---

# Cyfrowy bliźniak RL (`rl_twin/`) — faza 2: mechanika Tibii

```bash
python3 rl_twin_app.py                                                        # GUI
python3 -m rl_twin.trainer scenario.json --episodes 5000 --out druid_policy   # trening bez GUI
python3 brainstem_example.py druid_policy.npz                                 # test wyeksportowanej polityki
```

## Moduły

| Plik | Rola |
|---|---|
| `rl_twin/data/monsters.json` | baza potworów (HP, exp, prędkość, pancerz, melee, modyfikatory żywiołów) |
| `rl_twin/monsters.py` | `MonsterDatabase` – ładowanie i walidacja JSON |
| `rl_twin/combat.py` | żywioły, księga czarów (`SPELLBOOK`), pancerz, modyfikatory, cooldowny (czar + grupa) |
| `rl_twin/physics.py` | czas kroku z prędkości (wzór Tibii), koszt ukosu, A*, linia wzroku, zasięg ekranu 15×11 |
| `rl_twin/mapparser.py` | `MapParser`: PNG → siatka kolizji numpy |
| `rl_twin/entities.py` | `Entity(x, y, z)`, `Druid`, `Monster` |
| `rl_twin/env.py` | `DruidHuntEnv` (`reset`/`step`), zegar w ms, ticki serwera 50 ms |
| `rl_twin/observation.py` | `GameSnapshot` → wektor 190 cech (wspólny z `brainstem.py`) |
| `rl_twin/agent.py`, `trainer.py`, `gui.py` | Double DQN (numpy), pętla treningowa, GUI PyQt5 |

## Mechanika

- **Czas:** każda decyzja agenta to jeden `step()`. Krok zużywa realny czas kroku postaci, inna akcja 100 ms. Świat (ruch i ataki potworów, regeneracja, Utamo) liczy się w tickach 50 ms.
- **Prędkość → czas kroku:** `eff = 857.36·ln(speed + 261.29) − 4795.01`, `krok = ceil(1000·podłoże / eff / 50)·50 ms`. Ukos ×1.5 (konfigurowalne). Przykład: Druid lvl 100 – 250 ms / ukos 350 ms, Giant Spider – 300 / 450 ms.
- **A\*:** 8 kierunków, koszt ukosu = mnożnik; potwory szukają ścieżki do pola obok gracza, omijając przeszkody i inne potwory.
- **Obrażenia:** fizyczne → pancerz `random(armor/2, armor − (armor%2 + 1))` → modyfikator żywiołu (%) → Utamo Vita → HP. Pancerz działa tylko na obrażenia fizyczne (tak jak w Tibii); czary żywiołów omijają go i liczą się wyłącznie z modyfikatorem.
- **Czary:** Exori Frigo (lód), Exori Tera (ziemia) – zasięg 3, wymagają linii wzroku, CD 2 s + grupa ataku 2 s; Exura Gran – CD 1 s; Utamo Vita – 200 s; Great Mana Potion – 1 s exhaust.
- **Potwory** widzą gracza w obszarze ekranu 15×11 (±7, ±5) na tym samym piętrze; tracą cel po wyjściu z ekranu.
- **Akcje (14):** 4 kierunki, 4 ukosy, Exori Frigo, Exori Tera, Exura Gran, mikstura many, czekaj, runa Sudden Death.
- **Nagrody:** +100 zabicie, −5 za każde trafienie, −1000 śmierć, −1 za każde 500 ms czasu gry, −0.5 akcja niemożliwa, +2 punkt trasy.

## Ekwipunek i runy (faza 3)

- **Różdżki** (`rl_twin/data/items.json`, `ItemDatabase`): żywioł, obrażenia min–max, mana za atak, zasięg, wymagany poziom. Auto-atak co 2000 ms na własnym zegarze w pętli świata (działa też w trakcie kroku): cel = potwór w zasięgu i linii wzroku, pobiera manę (również z Utamo Vita), obrażenia × modyfikator żywiołu potwora. Za mało many = brak ataku.
- **Sudden Death** (akcja `use_sd`): śmierć, `poziom·0.2 + ML·[4.3..6.8] + [32..44]`, zasięg 7 (na ekranie, z linią wzroku), wymaga ML 15, zużywa runę, wspólna grupa cooldownu ataku 2000 ms z Exori Frigo/Tera.
- **Obserwacja (+8 cech):** cooldown SD, liczba run, czy jest różdżka, cooldown auto-ataku, czy cel jest w zasięgu różdżki/runy, modyfikator śmierci i żywiołu różdżki najbliższego potwora.
- Wartości różdżek i wzór SD są przybliżone — zweryfikuj z TibiaWiki.

## Czary potworów (faza 4)

- `monsters.json` → `spells`: `beam` (linia, `length` pól), `wave` (stożek, szerokość 1→3→5 co `spread` pól), `ball` (koło o promieniu `radius` wokół celu w zasięgu `range`). Dragon: Fire Wave + Great Fireball; Dragon Lord: Fire Wave + Fire Beam + Great Fireball.
- Potwory patrzą w jednym z 4 kierunków (ruch po ukosie ustawia je na wschód/zachód). Co `cooldown_ms` każdy czar losuje `chance`; rzucany jest tylko, gdy druid jest w obszarze (po obróceniu się potwora w jego stronę) i w linii wzroku. Obrażenia żywiołu idą bez pancerza: modyfikator → Utamo Vita → HP, −5 nagrody za trafienie.
- Geometria w `rl_twin/aoe.py`: kształty liczone raz (cache), test trafienia = jedno sprawdzenie w zbiorze; pole dokładnie po ukosie od smoka jest poza belką i falą.
- **Obserwacja (+8 cech, +81 pól):** kierunek patrzenia najbliższego potwora, wyrównanie poziome/pionowe/ukośne, czy ma czary obszarowe, czas do próby rzutu, „stoję w zagrożeniu”, oraz trzeci kanał okna 9×9 — mapa zagrożenia: 1 = pole, na którym fala/belka któregoś potwora by trafiła.
- GUI: żółta kreska = kierunek patrzenia potwora, kolorowy błysk = obszar rzuconego czaru, checkbox „Strefy zagrożenia” pokazuje mapę z obserwacji agenta.

## Ekonomia, Avalanche, teren (faza 5)

- **Nagroda** (`rl_twin/rewards.py`): `exp_scale · exp za obrażenia` (proporcjonalnie do zadanych obrażeń; suma przy zabiciu = exp potwora × mnożnik serwera) `+ gold_scale · loot` `− gold_scale · koszt zapasów` (Great Mana Potion, SD, Avalanche — ceny w `items.json`, średni loot w `monsters.json`) + dotychczasowe kary (trafienia, śmierć, czas, akcje niemożliwe). Stały bonus za zabicie = 0.
- **Statystyki**: exp/h, zysk/h, loot, koszt zapasów, HP min, % czasu z potworem obok, mikstury/h, runy/h, leczenia — w GUI i w trenerze (średnie ze 100 epizodów).
- **Avalanche** (akcja `use_avalanche`, klawisz 6): runa lodowa, koło 37 pól (jak GFB w Tibii), zasięg 7, wspólny cooldown ataku 2 s z SD/Exori, cel = pole trafiające najwięcej potworów (linia wzroku). Obserwacja: cooldown, liczba run, ile potworów trafiłaby teraz Avalanche.
- **Teren**: prędkość podłoża per pole (krok liczony z podłoża pola), schody/dziury/drabiny (żółte pola minimapy) łączone automatycznie z sąsiednim piętrem — wejście na pole zmienia piętro; nawigacja do punktu na innym piętrze prowadzi do właściwych schodów. Import folderu minimapy klienta (`Minimap_Color_*` + `Minimap_WaypointCost_*`, przycisk „Minimapa klienta…”): ściany, koszt pól (wolne pola = niebieski odcień), schody, tło każdego piętra. **Dekodowanie WaypointCost jest założeniem** (żółty = nieprzechodnie, magenta = nieodkryte, czerwony kanał = koszt względem najczęstszego) — sprawdź na swoich plikach.
- **Edytor**: PgUp/PgDn = piętro, Ctrl+Z / Ctrl+Y = cofnij / ponów, Delete = usuń ostatni punkt trasy.

## Taktyka hybrydowa (faza 7)

`rl_twin/tactics.py`: automat stanów obok polityki RL. Sieć steruje **tylko w stanie WALKA** (mikro: pozycja, kiting, rotacja czarów/run); resztę robi automat, a trener uczy się z akcji faktycznie wykonanej (kroki automatu działają jak demonstracje).

| Stan | Kto | Co |
|---|---|---|
| LURE | automat | idzie trasą; w strefie Lure zbiera min–max potworów (poza strefą walczy od 1); po 12 s bez pełnej paczki walczy i tak |
| PULL | automat | prowadzi paczkę na najbliższą Arenę, czeka gdy wolne potwory zostają > 3 pola z tyłu („smycz”) |
| WALKA | RL | walka w arenie; obserwacja: stan, wektor do areny, „w arenie”, styczna Orbity |
| UCIECZKA | automat (nadpisuje RL) | HP < próg (40%) albo potworów > limit → najbliższe schody/dziura/drabina → zmiana piętra |
| REGENERACJA | automat | leczy się i czeka do HP ≥ 90% i min. czasu (30 s) |
| POWRÓT | automat | schodami na piętro polowania → LURE |

Poza WALKĄ działa warstwa przetrwania (Exura Gran < 70% HP, mikstura < 60% many). Strefy rysujesz narzędziami **Strefa Lure / Arena walki / Orbita** (kółko myszy nad środkiem strefy = promień, PPM = usuń, Przesuń = przenieś).

Nagrody: `lure_bonus` za każdy potwór doprowadzony do walki, `aoe_missed` za pojedynczy cel (Exori/SD), gdy gotowa Avalanche trafiłaby ≥ 3, `escape_success` za udaną ucieczkę. Metryki: ucieczki, pulle, średnia paczka, marnotrawstwo zapasów % (przelane mikstury, nadmiarowe obrażenia run w dobijanych potworach), ocena bezpieczeństwa (min HP, 0 po śmierci), wartość netto/h (`exp_gold_value` gp za 1 exp + zysk).

Skrypty (`rl_twin/scripts.py`, „Eksport/Import skryptu”): `.json` (trasa z typami + strefy) albo `.txt` z liniami `Node (x, y, z)` / `Ladder` / `Hole` we współrzędnych świata (`Świat X/Y pola [0,0]`, ustawiane automatycznie z minimapy klienta). Import przyjmuje każdą linię z trzema liczbami.

## Faza 10: zachowanie potworów z Canary

Importer (`rl_twin.canary_monsters`) czyta z plików Lua Canary, poza statystykami, także
zachowanie. Środowisko je symuluje:

| Zachowanie | Źródło w Canary | W symulatorze | Ile potworów |
|---|---|---|---|
| **Dystans** | `flags.targetDistance` > 1 | potwór dystansowy podchodzi do zadanej odległości z linią wzroku i cofa się, gdy druid się zbliża; strzela (strzały, włócznie, czary) | 166 |
| **Ucieczka** | `flags.runHealth` | przy HP ≤ progu ucieka (oddala się, nie bije wręcz), dalej rzuca czary; gdy się wyleczy powyżej progu, wraca do walki | 427 |
| **Leczenie** | `defenses` z COMBAT_HEALING | co `interval` rzut na `chance`, leczy min–max (np. Hydra 260–407 z szansą 25%) | 606 |
| **Efekt z ataku wręcz** | `condition = {type, totalDamage, interval}` w ataku wręcz | trafienie nakłada obrażenia w czasie (trucizna, podpalenie, zamrożenie); suma rozłożona na 8 ticków | 174 |
| **Czary DoT** | atak `condition` (chmury trucizny itp.) | obszar jak fala/kula, nakłada obrażenia w czasie zamiast natychmiastowych | 129 czarów |
| **Spowolnienie / paraliż** | atak `speed` z ujemnym `speedChange` | prędkość druida spada (×2 skala, minimum 40) na `duration` | 161 czarów |
| **Powrót na spawn** | (zachowanie silnika) | potwór bez celu wraca w promień 2 pól od swojego spawnu | wszystkie |

Silniejszy efekt tego samego żywiołu zastępuje słabszy. Utamo Vita przyjmuje też obrażenia z
efektów.

W obserwacji (362 cechy) doszło:
- dla każdego z 4 najbliższych potworów: `ranged` i `fleeing`;
- dla druida: `condition_damage` (ile obrażeń z efektów jeszcze przyjdzie) i `slowed`.

Reguły nie gonią uciekających potworów: dobijają je SD, jeśli są w zasięgu, a jeśli nie,
polują dalej. Ręczna baza (9 gatunków) dostała te same pola z Canary; Hydra i Demon dostały też
brakujące czary obszarowe.

**Jeszcze nie modelowane:**
- pola na ziemi (fire field, poison field);
- przyzwania, niewidzialność, zmiana wyglądu, drunk;
- leczenie efektów przez druida (Exana Pox / Exana Flam);
- haste druida (Utani Hur / Gran Hur), który w grze zdejmuje paraliż.

## Faza 9: model uczy się trasy, lure'owania i ucieczek

Wcześniej trasę wyznaczałeś Ty, a lure, pull i ucieczkę robił automat. Teraz, po wyłączeniu
automatu taktycznego (albo z `--no-tactics` w treningu), decyduje o tym polityka.

**Respawn.** Zabity potwór wraca na swój spawn po `respawn_ms` (domyślnie 60 s; 5. pole spawnu
nadpisuje czas w sekundach). Jak w Tibii: potwór nie odradza się, gdy druid widzi jego pole.
Przy włączonym respawnie epizod kończy się śmiercią albo limitem decyzji, nie wybiciem
wszystkiego.

**Grupy spawnów i pamięć łowiska.** Spawny bliżej niż 6 pól tworzą grupę. Środowisko pamięta
to, co wiedziałby gracz:
- które grupy i kiedy odwiedził;
- które potwory zabił i kiedy wrócą;
- ile potworów powinno teraz być w każdej grupie.

Do obserwacji (dziś 362 cechy) doszło:

| Cechy | Znaczenie |
|---|---|
| `m0..m3_*` | 4 najbliższe potwory osobno: kierunek, HP, czy stoi obok, czy grozi falą/beamem z tej pozycji |
| `t0..t3_*` | 4 najbliższe grupy spawnów (koszt dojścia): kierunek, koszt drogi, ile potworów powinno tam być, czas od ostatniej wizyty |
| `hp_trend`, `mana_trend`, `damage_rate`, `respawn_soon` | krótka pamięć: zmiana HP/many w 2 s, obrażenia w 5 s, ile zabitych wraca w ciągu 10 s |
| `escape_*` | pierwszy krok i koszt drogi do najbliższych schodów/dziury |

Pozycja bezwzględna x/y jest domyślnie zerowana (`absolute_position=False`), więc polityka nie
uczy się na pamięć jednej mapy.

**Makro-akcje (21 akcji):**
- `goto_target_0..3`: krok w stronę k-tej grupy spawnów;
- `goto_arena`: krok w stronę najbliższej areny;
- `escape_stairs`: krok w stronę schodów.

Trasa po łowisku to sekwencja wyborów grup, lure to decyzja, kiedy przestać zbierać i zacząć
walczyć, a ucieczka to `escape_stairs` w niebezpieczeństwie.

**Nagrody za nowe decyzje** (tylko bez automatu):
- `aggro_gain`: +1 za każdego nowo zebranego potwora; każdy spawn liczy się raz, więc
  gubienie i ponowne łapanie aggro nic nie daje;
- `escape_success`: zmiana piętra, gdy HP < progu ucieczki albo potworów > limitu.

Resztę robi ekonomia: exp za obrażenia, Avalanche na paczkę, koszt zapasów, kara za trafienia
i śmierć.

**Polowanie bez trasy.** Bez punktów trasy automat (LURE) i reguły idą do grupy z największą
spodziewaną liczbą potworów na koszt dojścia i trzymają ten cel do skutku. Gdy wszystko jest
wybite, patrolują grupę odwiedzoną najdawniej, bo stanie przy spawnie blokuje respawn. Po
ucieczce na inne piętro regenerują się i wracają schodami.

**Nauczyciel.** `python -m rl_twin.trainer scen.json --no-tactics --teacher rules`, w GUI pole
„Nauczyciel (reguły)”. Na początku treningu część decyzji (malejąco do 0) podejmują reguły,
a DQN uczy się z tych przejść off-policy.

**Potwory z Canary.** `rl_twin/data/monsters_canary.json` zawiera 1592 potwory z plików Lua
Canary. Generuje go `python -m rl_twin.canary_monsters <canary>/data-otservbr-global/monster`.
- Prędkość ×2 (nowa skala Canary → stara).
- Odporności: 100 − percent.
- Ataki: fala, beam, kula, strzał.
- Loot to tylko spodziewane monety, czyli dolna granica zysku.

Domyślna baza = ręcznie dopracowane `monsters.json` (pierwszeństwo) + import. Łowiska mają
teraz wszystkie potwory ze spawnów, nie tylko 9 gatunków. Lista potworów w edytorze ma
wyszukiwanie.

## Faza 8: edytor all-in-one, mapa świata i natywne spawny

**Łowiska z listy.** Wybierasz łowisko z listy i klikasz **Wczytaj**. Symulator pobiera
prawdziwy wycinek mapy świata: ściany z pathfindingu Tibii, prędkość podłoża, schody i
piętra wyżej/niżej. Do tego stawia potwory z globalnej bazy spawnów Canary dokładnie na ich
współrzędnych [x, y, z]. Lista jest w `rl_twin/data/hunting_grounds.json`. Środki łowisk
pochodzą ze znaczników tibiamaps.io, a zasięg z rozkładu spawnów. Wpisy można dopisywać
ręcznie.

**Źródła mapy:**

| Źródło | Co daje | Moduł |
|---|---|---|
| Łowisko z listy / „Prawdziwe kolizje (tibiamaps)” | ściany, prędkość podłoża, schody, sąsiednie piętra, współrzędne świata | `tibiamaps.py`, `hunting.py` |
| Minimapa klienta (`Minimap_Color/WaypointCost_X_Y_Z.png`) | jak wyżej, z plików Twojego klienta | `terrain.py` |
| Mapa OTBM (OTServ / RME) + items.otb + items.xml | ściany (BLOCK_SOLID), prędkość podłoża, kolory minimapy, schody (floorchange), plik spawnów mapy | `otbm.py` |
| PNG minimapy | ściany z kolorów (przybliżenie) | `mapparser.py` |

**Spawny.** Obsługiwane formaty:
- XML w formacie TFS (`<spawns><spawn>`);
- XML w formacie Canary (`<monsters><monster centerx…>`);
- JSON (`[{name, x, y, z}]`).

Baza Canary jest pobierana raz do `data/cache/spawns/`. Potwory spoza
`rl_twin/data/monsters.json` są pomijane z ostrzeżeniem, które wymienia ich nazwy. Spawn
leżący na ścianie przesuwam na najbliższe wolne pole w promieniu 2.

**Wczytywanie w tle.** Każdy loader (łowisko, tibiamaps, minimapa klienta, OTBM, spawny,
teren projektu) działa w wątku `BackgroundJob`. GUI nie zamarza, a pasek w stopce pokazuje
postęp.

**Edytor:**
- mapa w centrum;
- doki: *Panel edytora*, *Podgląd skryptu (na żywo)*, *Dziennik*. Można je zwijać,
  przenosić i odpinać (menu Widok);
- warstwy jako przełączniki nad mapą i w menu Warstwy: Ściany, Spawny, Trasa, Strefy,
  Heatmapa. Heatmapa pokazuje, gdzie druid spędzał czas, w kolorach od fioletu do cyjanu;
- podgląd skryptu odświeża się po każdym kliknięciu. Formaty: `rl_twin.script` JSON,
  cavebot TXT albo pełny projekt;
- Widok → *Tryb streamera* powiększa czcionki.

**Projekt (Plik → Zapisz/Wczytaj projekt, Ctrl+S / Ctrl+O).** Jeden plik
`rl_twin.project` zawiera mapę, źródło terenu, trasę, spawny, strefy, taktykę, nagrody i
skrypt w współrzędnych świata. Zapis jest atomowy. Odczyt waliduje wszystko:
- uszkodzone wiersze kolizji i prędkości są naprawiane albo pomijane;
- trasa, spawny i strefy spoza mapy oraz nieznane potwory są pomijane;
- każdy problem trafia do listy ostrzeżeń w oknie i w dzienniku.

Tylko plik, który w ogóle nie jest projektem, daje błąd. Uszkodzony OTBM, items.otb albo
XML spawnów kończy się czytelnym komunikatem, a obecna mapa zostaje bez zmian.

Trening bez okna działa bez zmian na zapisanym projekcie:
`python -m rl_twin.trainer projekt.json --episodes 3000`. Wytrenowane modele przenoszą się po nazwach cech i akcji (`--resume`).

## Prawdziwe kolizje (tibiamaps) i walka na regułach

Kolor minimapy mówi tylko, jak wygląda podłoga. Piasek z kamieniem, czaszką albo stalagmitem
dalej jest rysowany jako piasek, a w grze nie da się na niego wejść. Na Darashia Dragon Lair
to ok. 9% piasku i 12% bruku. Przycisk **Prawdziwe kolizje (tibiamaps)**:

1. znajduje wczytany PNG minimapy na mapie świata (dane github.com/tibiamaps/tibia-map-data,
   dowolne powiększenie x1/x2/x3…);
2. bierze przechodniość i koszt pól z warstwy pathfindingu Tibii (`floor-ZZ-path.png`,
   ten sam format co `Minimap_WaypointCost` klienta);
3. dokłada piętra wyżej i niżej ze schodami, ustawia współrzędne świata (eksport skryptu)
   i przenosi trasę, spawny i strefy na prawdziwe piętro. Ostrzega o punktach trasy na
   zablokowanych polach.

Pliki pięter są pobierane raz do `data/cache/tibiamaps/`.

Trasa na mapie jest rysowana po polach, którymi druid naprawdę pójdzie (najkrótsza droga wokół
ścian). Czerwona przerywana linia oznacza brak przejścia między punktami.

**Steruje: reguły (bez modelu)**: skryptowa walka do testowania tras i taktyki bez trenowania:
leczenie i mikstury, zejście z fali/belki, Avalanche przy 3+ potworach, Exori w słabszy żywioł,
SD na zdrowe cele, odejście z kontaktu na cooldownie. Moduł `rl_twin/rules.py`.

## MapParser (PNG → siatka)

1. Obraz wczytywany jako RGBA (Pillow, awaryjnie Qt `QImage`).
2. Ściana = piksel ciemny (każdy kanał < „Próg czerni”, domyślnie 20), przezroczysty, albo w kolorze oznaczonym jako ściana. Domyślne ściany z palety minimapy Tibii: pustka, ściana jaskini/góra `(153,51,0)`, skały `(102,102,102)`, woda `(51,102,153)`, mur `(255,51,0)`, lawa, drzewa. Po wczytaniu mapy lista „Kolory mapy” pokazuje wszystkie kolory z udziałem % — zaznaczenie/odznaczenie od razu przelicza kolizje.
3. „Piksele na pole” = **auto**: minimapa (≤ 256 kolorów) → skala = największe s dzielące ≥ 90% długości jednokolorowych odcinków (np. zrzut minimapy powiększony 3× → 3 px = 1 pole) i przesunięcie siatki z fazy zmian koloru; zrzut klienta → 32 px. Ręczna wartość wyłącza auto.
4. Przy większych kaflach decyduje „Próg zajętości”.

Mapa PNG należy do piętra, na którym ją wczytano (tło ukryte na innych). Scenariusz zapisuje `map_floor`, `map_cell_px`, `map_offset`, `map_wall_colors`; bez kolizji mapa jest parsowana ponownie.

Edycja: narzędzie **Przesuń** (chwyć spawn/punkt trasy i upuść; ściana/zajęte pole = odmowa), **PPM na znaczniku** usuwa dokładnie ten znacznik, przycisk „Usuń potwory (piętro)”. Znaczniki mają stały rozmiar na ekranie, a postacie nie maleją poniżej ~20 px przy oddaleniu. Po końcu epizodu figury z symulacji znikają — zostają tylko edytowalne spawny.

Ruch druida: `env._movement_blocker()` odrzuca pole poza siatką, ścianę i pole z potworem — druid zostaje, zegar +100 ms, kara −0.5. Kierunek do punktu trasy: w linii prostej i po najkrótszej ścieżce omijającej ściany (`waypoint_path_*`).

## Dane potworów

Wartości w `monsters.json` pochodzą z ogólnej wiedzy o Tibii i są **przybliżone** — przed poważnym treningiem zweryfikuj je z TibiaWiki. Format: `elemental_resistances` = procent otrzymywanych obrażeń (100 neutralnie, 0 immunitet, 120 słabość).

## Format eksportu modelu

`druid_policy.npz`: `W0,b0` (362→128), `W1,b1` (128→128), `W2,b2` (128→21) + `metadata` (JSON: `feature_names`, `action_names`, stałe obserwacji, `env_config`). Inferencja: `argmax(relu(relu(x@W0+b0)@W1+b1)@W2+b2)`. Przy zainstalowanym PyTorch dodatkowo `druid_policy.pt` zgodny z `nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)`.

W `brainstem.py`: wypełnij `GameSnapshot`, przepuść przez `encode_observation()` z `rl_twin/observation.py` (nie przepisuj — kolejność cech musi się zgadzać) i podaj do `BrainstemPolicy.act_name()`. Modele z wcześniejszych faz (119/7, 190/13, 198/14) nie mają obecnego układu (362 cechy, 21 akcji): `load_policy()` / `--resume` przenosi wagi po nazwach cech i akcji (stare zachowanie bez zmian, nowe cechy = 0, nowa akcja = neutralna, epsilon wraca do połowy). Nagroda zmieniła się na ekonomiczną, więc i tak dotrenuj. Mapę zagrożenia w `brainstem.py` licz tą samą funkcją `rl_twin.aoe.danger_window` (pozycje i kierunki potworów + czary z `monsters.json`).

## Sprite'y (AssetManager)

Przy starcie GUI brakujące obrazki są pobierane w tle, z kilku źródeł po kolei:
1. pole `"sprite"` w `monsters.json`, jeśli to pełny URL,
2. oficjalna biblioteka tibia.com: `static.tibia.com/images/library/{nazwa}.gif` (druid: obrazy strojów `charactertrade/outfits/144_0.gif`, `148_0.gif`, `130_0.gif`),
3. API TibiaWiki (`api.php` → bezpośredni link do pliku),
4. TibiaWiki `Special:FilePath`.

Nieudane pobrania są pamiętane 24 h w `sprites/_missing.json` (z wersją listy źródeł — po zmianie źródeł stare wpisy są ignorowane). „Pobierz grafiki ponownie” wymusza próbę. Własne pliki: wrzuć `sprites/Nazwa_Potwora.gif|png`. `sprites/` jest w `.gitignore` (grafiki CipSoft). Bez GUI: `python3 -m rl_twin.assets`.
