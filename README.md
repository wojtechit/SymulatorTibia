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

## MapParser (PNG → siatka)

1. Obraz wczytywany jako RGBA (Pillow, awaryjnie Qt `QImage`).
2. Piksel jest ścianą, gdy **każdy** kanał RGB jest poniżej „Progu czerni” (domyślnie 20 — czerń `#000000` razem z szumem kompresji), gdy jest przezroczysty, albo gdy jest czystym niebieskim `#0000FF` (±60).
3. „Piksele na pole” = **auto**: obraz z ≤ 256 kolorami to minimapa Tibii → 1 px = 1 pole; więcej kolorów = zrzut klienta → 32 px. Minimapę zapisaną jako JPEG (szum = tysiące kolorów) ustaw ręcznie na 1.
4. Przy 1 px/pole każdy ciemny piksel jest ścianą; przy większych kaflach decyduje „Próg zajętości”.

Mapa PNG należy do piętra, na którym ją wczytano — na innych piętrach tło jest ukryte, a ich kolizje są osobne. Scenariusz zapisuje `map_floor`; przy wczytaniu scenariusza bez kolizji mapa jest parsowana ponownie. Przed epizodem/treningiem GUI ostrzega, jeśli piętro startowe nie ma ani jednej ściany.

Ruch druida: `env._movement_blocker()` odrzuca pole poza siatką, ścianę (`1`) i pole zajęte przez potwora — druid zostaje na miejscu, zegar idzie 100 ms, kara −0.5. Obserwacja ma kierunek do punktu trasy zarówno w linii prostej (`waypoint_dx/dy`), jak i **po najkrótszej ścieżce omijającej ściany** (`waypoint_path_dx/dy`, `waypoint_path_distance`; Dijkstra, liczony raz na punkt trasy).

## Dane potworów

Wartości w `monsters.json` pochodzą z ogólnej wiedzy o Tibii i są **przybliżone** — przed poważnym treningiem zweryfikuj je z TibiaWiki. Format: `elemental_resistances` = procent otrzymywanych obrażeń (100 neutralnie, 0 immunitet, 120 słabość).

## Format eksportu modelu

`druid_policy.npz`: `W0,b0` (290→128), `W1,b1` (128→128), `W2,b2` (128→14) + `metadata` (JSON: `feature_names`, `action_names`, stałe obserwacji, `env_config`). Inferencja: `argmax(relu(relu(x@W0+b0)@W1+b1)@W2+b2)`. Przy zainstalowanym PyTorch dodatkowo `druid_policy.pt` zgodny z `nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)`.

W `brainstem.py`: wypełnij `GameSnapshot`, przepuść przez `encode_observation()` z `rl_twin/observation.py` (nie przepisuj — kolejność cech musi się zgadzać) i podaj do `BrainstemPolicy.act_name()`. Modele z wcześniejszych faz (119/7, 190/13, 198/14) nie pasują do obecnej wersji (290 cech, 14 akcji) — trzeba je wytrenować od nowa. Mapę zagrożenia w `brainstem.py` licz tą samą funkcją `rl_twin.aoe.danger_window` (pozycje i kierunki potworów + czary z `monsters.json`).

## Sprite'y (AssetManager)

Przy starcie GUI `rl_twin/assets.py` sprawdza katalog `sprites/` i w tle (QThread) pobiera brakujące obrazki dla Druida i każdego potwora z `monsters.json`:
`https://tibia.fandom.com/wiki/Special:FilePath/{Nazwa_Z_Podkreślnikami}.gif` (potem `.png`). Druid: `Outfit_Druid_Male.gif`, awaryjnie `Outfit_Druid_Female.gif`.

- Nagłówki przeglądarki (Fandom potrafi odrzucać domyślny User-Agent `python-requests`), timeout 10 s, limit 2 MB, walidacja nagłówka GIF/PNG (strona HTML zamiast obrazka nie zostanie zapisana), zapis atomowy.
- Nieudane pobrania trafiają do `sprites/_missing.json` i są ponawiane po 24 h — start nie mieli sieci za każdym razem.
- Brak sprite'a = kółko jak wcześniej. Własna nazwa pliku: pole `"sprite"` w `monsters.json`. Inne źródło: zmienna `TIBIA_SPRITE_BASE_URL`.
- Bez GUI: `python3 -m rl_twin.assets` (`--force` pobiera ponownie).
- `sprites/` jest w `.gitignore` — grafiki to własność CipSoft, pobieraj je lokalnie, nie wrzucaj do repo.
