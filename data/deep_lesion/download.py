import os
import requests
import zipfile
import argparse
from tqdm import tqdm

# Stała lista 10 linków do plików DeepLesion
FILES_URLS = [
    "https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip",
    "https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip",
    "https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip",
    "https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip",
    "https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip",
    "https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip",
    "https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip",
    "https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip",
    "https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip",
    "https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip"
]

def download_file_with_progress(url, output_path):
    """Pobiera plik z URL, pokazując pasek postępu (tqdm)."""
    print(f"Pobieranie: {os.path.basename(output_path)}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Sprawdź, czy żądanie HTTP się powiodło
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        print(f"Pobrano pomyślnie: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nBłąd podczas pobierania {url}: {e}")
        # Usuń częściowo pobrany plik, aby uniknąć błędów
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def unzip_file(zip_path, extract_to):
    """Rozpakowuje plik .zip do wskazanego katalogu."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Rozpakowywanie {os.path.basename(zip_path)} do {extract_to}...")
            zip_ref.extractall(extract_to)
            print(f"Rozpakowano pomyślnie.")
    except zipfile.BadZipFile:
        print(f"Błąd: Plik {zip_path} jest uszkodzony lub nie jest plikiem zip.")
    except Exception as e:
        print(f"Nieoczekiwany błąd podczas rozpakowywania {zip_path}: {e}")

def main():
    # 1. Definiowanie argumentów skryptu
    parser = argparse.ArgumentParser(description="Pobieranie i rozpakowywanie zestawu danych DeepLesion.")
    
    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        default="data",
        help="Katalog docelowy do pobrania i rozpakowania plików. Domyślnie: ./data"
    )
    
    parser.add_argument(
        "-n", "--num_files",
        type=int,
        default=1,
        help="Liczba plików .zip do pobrania (wartość 1-10). Domyślnie: 1"
    )
    
    args = parser.parse_args()
    
    # 2. Walidacja i przygotowanie katalogów
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    num_to_download = args.num_files
    if not (1 <= num_to_download <= 10):
        print(f"Ostrzeżenie: Liczba plików ({num_to_download}) jest poza zakresem [1, 10].")
        num_to_download = max(1, min(10, num_to_download)) # Ogranicz do zakresu
        print(f"Ustawiono liczbę plików do pobrania na: {num_to_download}")

    print(f"Rozpoczynam pobieranie. Katalog docelowy: {data_dir}")
    print(f"Liczba plików do pobrania: {num_to_download}")
    
    # 3. Główna pętla (jak w skrypcie .sh)
    for i in range(num_to_download):
        # Formatowanie nazwy pliku "Images_png_01.zip", "Images_png_02.zip", itd.
        # (i jest od 0, więc dodajemy 1)
        idx_str = f"{i + 1:02d}"
        filename = f"Images_png_{idx_str}.zip"
        zip_path = os.path.join(data_dir, filename)
        url = FILES_URLS[i]
        
        print(f"\n--- Przetwarzanie pliku {i+1}/{num_to_download} ---")

        # Sprawdzenie, czy plik .zip już istnieje
        if os.path.exists(zip_path):
            print(f"Plik {filename} już istnieje w '{data_dir}'. Pomijanie.")
        else:
            # Jeśli nie istnieje - pobierz
            if download_file_with_progress(url, zip_path):
                # Jeśli pobieranie się udało - rozpakuj
                unzip_file(zip_path, data_dir)
                
    print(f"\nZakończono. Pliki znajdują się w katalogu '{data_dir}'.")

if __name__ == "__main__":
    main()