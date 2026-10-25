/* ==========================================================================
   CineMatch frontend – vanilla JS, DOM built with createElement / appendChild.
   Flow:  fetch(/api/recommendations/) -> JSON -> hero + one carousel per row
   ========================================================================== */
(() => {
  "use strict";

  const API_URL = document.body.dataset.apiUrl;

  const heroEl = document.getElementById("hero");
  const heroTitleEl = document.getElementById("hero-title");
  const heroGenresEl = document.getElementById("hero-genres");
  const heroOverviewEl = document.getElementById("hero-overview");
  const heroLinkEl = document.getElementById("hero-link");
  const statusEl = document.getElementById("status");
  const userForm = document.getElementById("user-form");
  const userInput = document.getElementById("user-id-input");

  let requestCounter = 0;   // lets us ignore out-of-date responses if the user switches quickly

  /* ---------- tiny DOM helper ---------- */
  function createEl(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.classList.add(...className.split(" "));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function clearChildren(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function setStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.classList.toggle("status--error", isError);
  }

  /* ---------- API ---------- */
  async function fetchRecommendations(userId) {
    const params = new URLSearchParams();
    if (userId) params.set("user_id", userId);

    const response = await fetch(`${API_URL}?${params.toString()}`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    return response.json();
  }

  /* ---------- MOVIE CARD ---------- */
  function showFallback(card, title) {
    card.classList.add("movie-card--fallback");
    card.insertBefore(createEl("span", "movie-card__fallback-title", title), card.firstChild);
  }

  function createMovieCard(movie) {
    const card = createEl("article", "movie-card");
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", movie.title);

    if (movie.poster_url) {
      const img = createEl("img", "movie-card__img");
      img.src = movie.poster_url;
      img.alt = movie.title;
      img.loading = "lazy";
      img.addEventListener("error", () => {         // broken image -> title card instead
        img.remove();
        showFallback(card, movie.title);
      });
      card.appendChild(img);
    } else {
      showFallback(card, movie.title);
    }

    const info = createEl("div", "movie-card__info");
    info.appendChild(createEl("h3", "movie-card__title", movie.title));
    info.appendChild(createEl("p", "movie-card__genres", movie.genres.slice(0, 3).join(" • ")));
    card.appendChild(info);

    // clicking a card promotes it to the hero banner
    const promote = () => {
      renderHero(movie);
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
    card.addEventListener("click", promote);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        promote();
      }
    });

    return card;
  }

  /* ---------- ROW / CAROUSEL ---------- */
  function createArrow(direction, strip) {
    const btn = createEl("button", `row__arrow row__arrow--${direction}`, direction === "left" ? "‹" : "›");
    btn.type = "button";
    btn.setAttribute("aria-label", direction === "left" ? "Scroll left" : "Scroll right");
    btn.addEventListener("click", () => {
      const step = strip.clientWidth * 0.85 * (direction === "left" ? -1 : 1);
      strip.scrollBy({ left: step, behavior: "smooth" });
    });
    return btn;
  }

  function renderRow(rowData) {
    // find the empty container declared in index.html
    const section = document.querySelector(`.row[data-row-id="${rowData.id}"]`);
    if (!section) return;

    const titleEl = section.querySelector(".row__title");
    const viewport = section.querySelector(".row__viewport");
    const strip = section.querySelector(".row__posters");

    titleEl.textContent = rowData.title;
    clearChildren(strip);

    rowData.movies.forEach((movie) => strip.appendChild(createMovieCard(movie)));

    if (!viewport.querySelector(".row__arrow")) {
      viewport.appendChild(createArrow("left", strip));
      viewport.appendChild(createArrow("right", strip));
    }

    section.hidden = rowData.movies.length === 0;
    strip.scrollLeft = 0;
  }

  /* ---------- HERO ---------- */
  function renderHero(movie) {
    if (!movie) return;

    heroTitleEl.textContent = movie.title;
    heroGenresEl.textContent = movie.genres.join(" • ");
    heroOverviewEl.textContent = movie.overview || "";

    if (movie.backdrop_url) {
      heroEl.style.backgroundImage = `url("${movie.backdrop_url}")`;
      heroEl.classList.remove("hero--no-image");
    } else {
      heroEl.style.backgroundImage = "";
      heroEl.classList.add("hero--no-image");
    }

    if (movie.imdb_id) {
      heroLinkEl.href = `https://www.imdb.com/title/tt${movie.imdb_id}/`;
      heroLinkEl.hidden = false;
    } else {
      heroLinkEl.hidden = true;
    }
  }

  /* ---------- PAGE LOAD ---------- */
  async function loadPage(userId) {
    const myRequest = ++requestCounter;
    setStatus("Loading recommendations…");

    try {
      const data = await fetchRecommendations(userId);
      if (myRequest !== requestCounter) return;     // a newer request already took over

      renderHero(data.hero);
      document.querySelectorAll(".row").forEach((section) => { section.hidden = true; });
      data.rows.forEach(renderRow);
      setStatus("");
    } catch (error) {
      if (myRequest !== requestCounter) return;
      setStatus(`Could not load recommendations (${error.message}). Is the Django server running?`, true);
    }
  }

  userForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const userId = userInput.value.trim();

    const url = new URL(window.location.href);      // keep ?user_id= in the address bar
    if (userId) url.searchParams.set("user_id", userId);
    else url.searchParams.delete("user_id");
    window.history.replaceState({}, "", url);

    loadPage(userId);
  });

  const initialUser = new URLSearchParams(window.location.search).get("user_id") || "";
  userInput.value = initialUser;
  loadPage(initialUser);
})();
