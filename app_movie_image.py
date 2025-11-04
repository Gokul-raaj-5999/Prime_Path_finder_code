import requests
import re

def extract_title_and_year(movie_name):
    match = re.search(r"\((\d{4})\)", movie_name)
    if match:
        year = match.group(1)
        title = movie_name.replace(f"({year})", "").strip()
        return title.strip(), int(year)
    else:
        return movie_name.strip(), None
    

def get_poster_url_from_title(movie_title, year=None):
    """
    Fetch movie poster URL from OMDb API using movie title (and optionally year).
    Returns a poster URL or a fallback placeholder if not found.
    """
    API_KEY = "afce0350"  # Your OMDb API key
    base_url = "http://www.omdbapi.com/"
    title, year = extract_title_and_year(movie_title)
    params = {
        "apikey": API_KEY,
        "t": title,
        "y": year,
    }

    # Include the year only if provided
    if year:
        params["y"] = str(year)

    try:
        resp = requests.get(base_url, params=params, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()

        if data.get("Response") == "False":
            return None

        poster = data.get("Poster", "")
        if poster and poster != "N/A":
            return poster
        else:
            return None
    except Exception as e:
        print(f"Poster fetch failed for '{title}': {e}")
        return None
