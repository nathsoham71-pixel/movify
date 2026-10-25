# Movify: Netflix-Style Hybrid Movie Recommender

A college mini-project that recommends movies the way Netflix does: pick a movie and get
personalised "Top Picks", "More Like This" and genre rows.

| Layer | Technology |
|---|---|
| Backend | Django 5 (Python 3.11+), SQLite via the Django ORM |
| Machine learning | pandas, scikit-learn (TF-IDF + cosine similarity), scikit-surprise (SVD) |
| Security | Shilling-attack (bot) filtering before model training |
| Frontend | Vanilla HTML5, CSS3 and JavaScript (Fetch API + DOM). No frameworks |
| Dataset | [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/latest/): 9,742 movies, 100,836 ratings, 610 users |

---

## Contents

1. [How it works](#how-it-works)
2. [Project structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Step 1: Open the project and create a virtual environment in VS Code](#step-1-open-the-project-and-create-a-virtual-environment-in-vs-code)
5. [Step 2: Install the requirements](#step-2-install-the-requirements)
6. [Step 3: Run migrations and load the MovieLens data](#step-3-run-migrations-and-load-the-movielens-data)
7. [Step 4: Start the Django server and open the frontend](#step-4-start-the-django-server-and-open-the-frontend)
8. [Using the app](#using-the-app)
9. [API reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## How it works

```
ratings.csv / movies.csv
        |  python manage.py load_movielens   (pandas + bulk_create)
        v
   SQLite database (Movie, Rating)
        |
        |  bot_defense.py: Django ORM aggregations flag fake accounts
        |    volume    > 1000 ratings in total
        |    velocity  > 500 ratings in one hour
        |    flat      20+ ratings with almost no variation
        |    extreme   90%+ of ratings are 5 stars (push) or <= 1 star (nuke)
        |    cluster   3+ accounts give one movie the same extreme score in the same hour
        v
   Clean pandas DataFrame
        |
        v
  ml_engine.py
    Content engine:       TF-IDF on genres + cosine similarity  -> 50 candidate movies
    Collaborative engine: Surprise SVD trained on clean ratings -> predicted rating per candidate
    Hybrid score = 0.7 x predicted rating (normalised) + 0.3 x genre similarity
    Unknown or new user (cold start): popularity replaces the SVD prediction
        |
        v
  views.py  ->  /api/recommendations/?user_id=1&movie_id=10  (JSON)
        |
        v
  frontend/app.js: fetch() + document.createElement -> Netflix-style carousels
```

The model trains automatically on the **first API request** (about 2 seconds) and then stays in memory.

---

## Project structure

```
cws/
├── backend/
│   ├── manage.py
│   ├── db.sqlite3                      # created by "migrate"
│   ├── requirements.txt
│   ├── cws_backend/                    # Django project configuration
│   │   ├── settings.py                 # installed apps, CORS, SQLite
│   │   ├── urls.py                     # mounts the API at /api/
│   │   ├── wsgi.py
│   │   └── asgi.py
│   └── recommender/                    # Django app
│       ├── models.py                   # Movie, Rating
│       ├── admin.py                    # admin pages for both models
│       ├── views.py                    # JSON API endpoints
│       ├── urls.py                     # API routes
│       ├── ml_engine.py                # TF-IDF + SVD hybrid engine
│       ├── bot_defense.py              # shilling-attack filtering
│       ├── migrations/
│       ├── management/commands/
│       │   └── load_movielens.py       # CSV ingestion command
│       └── data/
│           ├── movies.csv
│           └── ratings.csv
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

---

## Prerequisites

Install these once:

1. **Python 3.11 or newer**: <https://www.python.org/downloads/>
   On Windows, tick **"Add python.exe to PATH"** in the installer.
   Check it in a new terminal:
   ```bash
   python --version
   ```
2. **Visual Studio Code**: <https://code.visualstudio.com/>
3. VS Code extensions (Extensions panel, `Ctrl+Shift+X`):
   - **Python** (by Microsoft)
   - **Live Server** (by Ritwick Dey). Optional, but the easiest way to open the frontend.
4. **MovieLens data.** `movies.csv` and `ratings.csv` must be in `backend/recommender/data/`.
   If they are missing, download `ml-latest-small.zip` from
   <https://grouplens.org/datasets/movielens/latest/>, unzip it, and copy those two files there.

---

## Step 1: Open the project and create a virtual environment in VS Code

A virtual environment (`venv`) keeps this project's packages separate from the rest of your computer.

1. Open VS Code, then **File > Open Folder...** and choose the `cws` folder.
2. Open the terminal with **Terminal > New Terminal** (or <kbd>Ctrl</kbd>+<kbd>`</kbd>).
3. Go into the backend folder and create the venv:

   **Windows (PowerShell)**
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   ```

4. When it is active, your prompt starts with **`(venv)`**.
5. Tell VS Code to use it: press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>, choose
   **Python: Select Interpreter**, and pick the one under `backend\venv`
   (`.\backend\venv\Scripts\python.exe`). New terminals will then activate it automatically.

> **Windows error "running scripts is disabled on this system"?** Run this once, then try activating again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

> Every later command assumes you are in the `backend` folder with `(venv)` active.

---

## Step 2: Install the requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs Django, django-cors-headers, numpy, pandas, scikit-learn and scikit-surprise.
It takes 1 to 3 minutes the first time.

Check it worked:
```bash
python -c "import django, pandas, sklearn, surprise; print('All packages OK')"
```

---

## Step 3: Run migrations and load the MovieLens data

1. **Create the database tables.** This creates `backend/db.sqlite3`:
   ```bash
   python manage.py migrate
   ```

2. **Load the CSV files** into the database. This takes about 5 seconds:
   ```bash
   python manage.py load_movielens
   ```
   Expected output:
   ```
   Inserted 9,742 movies (0 already present).
   Processed 100,836 ratings (0 invalid/duplicate rows skipped).
   Done in 5.2s. Database now has 9,742 movies and 100,836 ratings.
   ```

   Useful options:
   ```bash
   python manage.py load_movielens --clear                 # wipe and reload everything
   python manage.py load_movielens --data-dir C:\path\to\ml-latest-small
   python manage.py load_movielens --batch-size 10000
   ```
   Running the command twice is safe: existing rows are skipped, not duplicated.

3. *(Optional)* Create an admin login so you can browse the data at <http://127.0.0.1:8000/admin/>:
   ```bash
   python manage.py createsuperuser
   ```

---

## Step 4: Start the Django server and open the frontend

You need **two things running**: the Django API (backend) and the web page (frontend).

### 4a. Start the backend

In the VS Code terminal (inside `backend`, with `(venv)` active):

```bash
python manage.py runserver
```

Leave this terminal open. Check it at <http://127.0.0.1:8000/api/health/>. You should see:

```json
{"status": "ok", "movies": 9742, "ratings": 100836}
```

### 4b. Open the frontend (pick one option)

**Option A: VS Code Live Server (recommended)**
1. In the Explorer, open `frontend/index.html`.
2. Right-click in the editor and choose **"Open with Live Server"**.
3. Your browser opens at `http://127.0.0.1:5500/frontend/index.html`.

**Option B: Python's built-in web server**
Open a **second** terminal (the `+` icon in the terminal panel) and run:
```bash
cd frontend
python -m http.server 5500
```
Then visit <http://127.0.0.1:5500>.

**Option C: Open the file directly**
Double-click `frontend/index.html`. This works too, because CORS allows every origin while `DEBUG` is on.

The first page load can take about 2 seconds while the model trains. After that, responses
take about 10 ms.

To stop a server, click its terminal and press <kbd>Ctrl</kbd>+<kbd>C</kbd>.

---

## Using the app

| Action | What happens |
|---|---|
| **Click any movie card** | It becomes the base movie; the hero banner and "Top Picks" row update |
| **Search box** | Type 2+ letters (e.g. `matrix`), then click a result |
| **User box and Switch** | Get recommendations for another MovieLens user (IDs 1 to 610) |
| **User `99999`** | A user with no history, so you see the **cold-start** fallback |
| **Hover a card** | It enlarges and shows match %, year, predicted rating and genres |
| **URL** | `index.html?user_id=5&movie_id=2571` opens directly with that user and movie |

Some heavy raters (for example users 68, 414 and 599) are filtered out by bot defense because they
have more than 1,000 ratings, so they also get cold-start results. To relax the rule, edit
`max_total_ratings` in `DefenseConfig` in `backend/recommender/bot_defense.py`.

---

## API reference

All endpoints accept **GET** only and return JSON. Base URL: `http://127.0.0.1:8000/api`

| Endpoint | Description |
|---|---|
| `/recommendations/?user_id=1&movie_id=10&n=5` | Hybrid recommendations (`n` defaults to 5, max 50) |
| `/movies/popular/?n=20&genre=Comedy` | Top-rated movies (Bayesian average), optionally for one genre |
| `/movies/search/?q=toy&n=10` | Title search |
| `/movies/<movie_id>/` | One movie's details |
| `/movies/<movie_id>/similar/?n=10` | Movies with similar genres |
| `/genres/` | All genre names |
| `/health/` | Server and database check |

Example: `GET /api/recommendations/?user_id=1&movie_id=10`

```json
{
  "user_id": 1,
  "seed_movie": {"movie_id": 10, "title": "GoldenEye", "year": 1995, "genres": ["Action", "Adventure", "Thriller"]},
  "strategy": "hybrid",
  "count": 5,
  "results": [
    {
      "movie_id": 1610, "title": "Hunt for Red October, The", "year": 1990,
      "genres": ["Action", "Adventure", "Thriller"],
      "score": 0.93, "similarity": 1.0, "predicted_rating": 4.6, "strategy": "hybrid"
    }
  ]
}
```

- `strategy` is `"hybrid"` for known users and `"cold_start"` for new or filtered users.
- Errors return a message with the right status code, e.g. `400 {"error": "Missing required query parameter 'user_id'."}` or `404 {"error": "Movie not found."}`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Reinstall Python with **"Add to PATH"** ticked, or use `py` instead of `python` on Windows |
| "running scripts is disabled" when activating | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError: No module named 'django'` | The venv is not active. Run `.\venv\Scripts\Activate.ps1` again (look for `(venv)`) |
| `scikit-surprise` fails to install with a C++ error | Use Python 3.11 to 3.13 (pre-built wheels exist), or install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and retry |
| `File not found: ...movies.csv` | Put `movies.csv` and `ratings.csv` in `backend/recommender/data/` |
| `no such table: recommender_movie` | Run `python manage.py migrate` |
| Page shows **"Backend offline"** | Start Django (`python manage.py runserver`) and click **Retry** |
| Rows say "No movies in the database" | Run `python manage.py load_movielens` |
| Port 8000 is already in use | `python manage.py runserver 8001`, then change `API_BASE` at the top of `frontend/app.js` to `http://127.0.0.1:8001/api` |
| Want to start completely fresh | Stop the server, delete `backend/db.sqlite3`, then run `migrate` and `load_movielens` again |

### Production notes

For anything beyond local development, set these environment variables before starting Django:

```bash
DJANGO_DEBUG=False
DJANGO_SECRET_KEY=<a long random string>
DJANGO_ALLOWED_HOSTS=yourdomain.com
CORS_ALLOWED_ORIGINS=https://your-frontend-domain.com
```

With `DEBUG` off, only the origins listed in `CORS_ALLOWED_ORIGINS` can call the API.

---

**Dataset credit:** F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History
and Context.* ACM Transactions on Interactive Intelligent Systems 5(4). GroupLens Research, University of Minnesota.
