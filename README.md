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
