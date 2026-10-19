"""
AssetManager: downloads missing sprites from TibiaWiki (Fandom) into sprites/.

    manager = AssetManager("sprites")
    manager.ensure([("Giant Spider", None), ("Druid", None)])

Files are fetched from  {base_url}{Name_With_Underscores}.gif  (Special:FilePath
redirects to the static image host), validated by their magic bytes and saved
as  sprites/{Name_With_Underscores}.gif|.png. Failed names are remembered in
sprites/_missing.json and retried after RETRY_MISSING_AFTER_S. No Qt here, so it
runs in a QThread or from the command line:

    python -m rl_twin.assets
"""

import json
import os
import time
from dataclasses import dataclass

try:
    import requests
except ImportError:  # the simulator still runs, only without downloads
    requests = None

DEFAULT_BASE_URL = os.environ.get("TIBIA_SPRITE_BASE_URL", "https://tibia.fandom.com/wiki/Special:FilePath/")
DEFAULT_HEADERS = {
    # Fandom rejects the default python-requests agent on some edges; present a regular browser
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36 SymulatorTibia/2.0"),
    "Accept": "image/avif,image/webp,image/apng,image/gif,image/png,image/*;q=0.8,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.8",
    "Referer": "https://tibia.fandom.com/",
}
IMAGE_EXTENSIONS = (".gif", ".png")
MISSING_INDEX = "_missing.json"
RETRY_MISSING_AFTER_S = 24 * 3600
MAX_DOWNLOAD_BYTES = 2_000_000

# Player characters have no creature page; these are TibiaWiki outfit files, tried in order
CHARACTER_SPRITES = {
    "Druid": ("Outfit_Druid_Male.gif", "Outfit_Druid_Female.gif"),
}


def wiki_file_name(name):
    """'Giant Spider' -> 'Giant_Spider' (TibiaWiki file naming)."""
    return "_".join(name.strip().split())


def detect_image_extension(data):
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return None


@dataclass
class DownloadResult:
    name: str
    status: str          # "cached" | "downloaded" | "failed" | "skipped"
    path: str = None
    message: str = ""


class AssetManager:
    def __init__(self, sprites_dir, base_url=DEFAULT_BASE_URL, timeout=10.0,
                 headers=None, log=print, session=None):
        self.sprites_dir = os.path.abspath(sprites_dir)
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.timeout = timeout
        self.headers = dict(DEFAULT_HEADERS, **(headers or {}))
        self.log = log
        self._session = session
        os.makedirs(self.sprites_dir, exist_ok=True)

    # ----- local lookup ---------------------------------------------------- #

    def local_path(self, name):
        base = os.path.join(self.sprites_dir, wiki_file_name(name))
        for extension in IMAGE_EXTENSIONS:
            if os.path.isfile(base + extension):
                return base + extension
        return None

    def candidates(self, name, sprite_hint=None):
        """Remote file names to try, most specific first."""
        if sprite_hint:
            return [sprite_hint] if isinstance(sprite_hint, str) else list(sprite_hint)
        if name in CHARACTER_SPRITES:
            return list(CHARACTER_SPRITES[name])
        stem = wiki_file_name(name)
        return [stem + ".gif", stem + ".png"]

    # ----- downloading ----------------------------------------------------- #

    def ensure(self, entries, force=False, should_stop=None):
        """
        Make sure every (name, sprite_hint) in `entries` has a local sprite.
        Never raises on network problems; returns a DownloadResult per entry.
        """
        missing = self._read_missing_index()
        results = []
        for name, hint in entries:
            if should_stop is not None and should_stop():
                break
            existing = self.local_path(name)
            if existing and not force:
                results.append(DownloadResult(name, "cached", existing))
                continue
            failed_at = missing.get(name)
            if failed_at and not force and time.time() - failed_at < RETRY_MISSING_AFTER_S:
                results.append(DownloadResult(name, "skipped", None, "niedawno nieudane - pomijam"))
                continue
            result = self.fetch(name, hint)
            if result.status == "downloaded":
                missing.pop(name, None)
            else:
                missing[name] = time.time()
            results.append(result)
        self._write_missing_index(missing)
        return results

    def fetch(self, name, sprite_hint=None):
        if requests is None:
            self.log(f"[WARN] Brak biblioteki 'requests' - pomijam pobieranie {name}.")
            return DownloadResult(name, "failed", None, "brak requests")
        errors = []
        for remote_name in self.candidates(name, sprite_hint):
            url = self.base_url + remote_name
            self.log(f"[INFO] Pobieranie obrazka dla {name} ({remote_name})...")
            try:
                data = self._download(url)
            except requests.Timeout:
                errors.append(f"{remote_name}: przekroczono czas ({self.timeout:g} s)")
            except requests.HTTPError as exc:
                errors.append(f"{remote_name}: HTTP {exc.response.status_code if exc.response is not None else '?'}")
            except requests.RequestException as exc:
                errors.append(f"{remote_name}: błąd sieci ({type(exc).__name__})")
            except ValueError as exc:
                errors.append(f"{remote_name}: {exc}")
            else:
                extension = detect_image_extension(data)
                if extension is None:
                    errors.append(f"{remote_name}: odpowiedź nie jest obrazkiem GIF/PNG")
                    continue
                try:
                    path = self._save(name, data, extension)
                except OSError as exc:
                    errors.append(f"{remote_name}: zapis nieudany ({exc})")
                    continue
                self.log(f"[INFO] Pobieranie obrazka dla {name}... OK! ({len(data)} B → {os.path.basename(path)})")
                return DownloadResult(name, "downloaded", path)
        message = "; ".join(errors)
        self.log(f"[WARN] Nie udało się pobrać obrazka dla {name}: {message}")
        return DownloadResult(name, "failed", None, message)

    def _download(self, url):
        response = self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=True)
        try:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(chunk_size=16384):
                data.extend(chunk)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("plik większy niż limit 2 MB")
            return bytes(data)
        finally:
            response.close()

    def _save(self, name, data, extension):
        path = os.path.join(self.sprites_dir, wiki_file_name(name) + extension)
        for stale in IMAGE_EXTENSIONS:   # drop an older copy in the other format
            other = os.path.join(self.sprites_dir, wiki_file_name(name) + stale)
            if stale != extension and os.path.isfile(other):
                os.remove(other)
        temp = path + ".part"
        with open(temp, "wb") as handle:
            handle.write(data)
        os.replace(temp, path)
        return path

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.headers)
        return self._session

    # ----- failure index --------------------------------------------------- #

    def _missing_index_path(self):
        return os.path.join(self.sprites_dir, MISSING_INDEX)

    def _read_missing_index(self):
        try:
            with open(self._missing_index_path(), encoding="utf-8") as handle:
                data = json.load(handle)
            return {str(k): float(v) for k, v in data.items()}
        except (OSError, ValueError, AttributeError):
            return {}

    def _write_missing_index(self, missing):
        try:
            with open(self._missing_index_path(), "w", encoding="utf-8") as handle:
                json.dump(missing, handle, indent=2)
        except OSError:
            pass


def sprite_entries(monster_db, characters=("Druid",)):
    """(name, sprite_hint) pairs for every character class and monster in the database."""
    entries = [(name, None) for name in characters]
    entries += [(name, monster_db.get(name).sprite) for name in monster_db.names()]
    return entries


def main():
    import argparse

    from .monsters import MonsterDatabase

    parser = argparse.ArgumentParser(description="Pobierz brakujące sprite'y z TibiaWiki")
    parser.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "sprites"))
    parser.add_argument("--monsters", help="plik monsters.json (domyślnie wbudowany)")
    parser.add_argument("--force", action="store_true", help="pobierz ponownie także istniejące")
    args = parser.parse_args()
    manager = AssetManager(args.dir)
    results = manager.ensure(sprite_entries(MonsterDatabase.load(args.monsters)), force=args.force)
    counts = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"[INFO] Gotowe: {counts}")


if __name__ == "__main__":
    main()
