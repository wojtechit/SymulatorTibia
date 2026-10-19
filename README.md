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

# Cyfrowy bliźniak RL (`rl_twin/`)

Środowisko treningowe dla agenta RL (Double DQN) + GUI do edycji scenariusza, testów ręcznych i treningu w tle.

```bash
python3 rl_twin_app.py                                   # GUI
python3 -m rl_twin.trainer scenario.json --episodes 5000 --out druid_policy   # trening bez GUI (np. na serwerze)
python3 brainstem_example.py druid_policy.npz            # test wyeksportowanej polityki
```

## Struktura

| Plik | Rola | Zależności |
|---|---|---|
| `rl_twin/observation.py` | `GameSnapshot` → wektor obserwacji (119 cech) | numpy |
| `rl_twin/env.py` | `DruidHuntEnv`: `reset()`, `step(action)`, Druid, Monster, nagrody | numpy |
| `rl_twin/agent.py` | Double DQN (MLP, Adam, replay buffer), eksport/import | numpy (torch opcjonalnie) |
| `rl_twin/trainer.py` | pętla treningowa (wspólna dla QThread i CLI) | numpy |
| `rl_twin/gui.py` | PyQt5: mapa, tryby, QThread, wykres nagród | PyQt5 |
| `brainstem_example.py` | wzorcowy loader modelu do `brainstem.py` | numpy |

## Środowisko

- **Akcje (Discrete 7):** 0 góra, 1 dół, 2 lewo, 3 prawo, 4 atak (Exori Frigo, zasięg 4 pola), 5 leczenie (Exura Gran), 6 Great Mana Potion.
- **Utamo Vita** jest utrzymywane automatycznie (checkbox), jego stan jest w obserwacji.
- **Nagrody:** +100 zabicie, −5 za każde otrzymane trafienie, −1000 śmierć, −1 za krok, −0.5 za akcję niemożliwą (cooldown/brak many/brak celu/ściana), +2 za osiągnięcie punktu trasy. Wszystko w `RewardConfig`.
- **Koniec epizodu:** śmierć lub wszystkie potwory martwe (`terminated`), limit kroków (`truncated`).
- Potwory poruszają się z prędkością względną do Druida (`speed_monster / speed_druid` pola na tick), więc wolniejsze da się kitować.

## Format eksportowanego modelu

`druid_policy.npz` (zawsze) — zwykłe archiwum numpy, ładowane z `allow_pickle=False`:

| Klucz | Kształt | Opis |
|---|---|---|
| `W0`, `b0` | (119, 128), (128,) | warstwa 1, ReLU |
| `W1`, `b1` | (128, 128), (128,) | warstwa 2, ReLU |
| `W2`, `b2` | (128, 7), (7,) | wyjście: Q-wartości akcji (bez aktywacji) |
| `metadata` | string JSON | `layer_sizes`, `feature_names` (kolejność cech!), `action_names`, stałe obserwacji, statystyki treningu, `env_config` |

Inferencja: `q = relu(relu(x@W0+b0)@W1+b1)@W2+b2`, akcja = `argmax(q)`.

`druid_policy.pt` (gdy zainstalowany PyTorch) — `{"state_dict", "metadata"}`; `state_dict` pasuje do
`nn.Sequential(nn.Linear(119,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,7))`.

W `brainstem.py` wypełniasz `GameSnapshot` danymi z gry, wołasz `encode_observation()` z `rl_twin/observation.py`
(ten sam kod co w treningu — nie przepisuj go) i podajesz wynik do `BrainstemPolicy.act()`.
