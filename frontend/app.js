/* ==========================================================================
   Movify frontend - vanilla JavaScript (Fetch API + DOM manipulation)
   ========================================================================== */
"use strict";

const API_BASE = window.MOVIFY_API_BASE || "http://127.0.0.1:8000/api";

const DEFAULT_USER_ID = 1;
const DEFAULT_MOVIE_ID = 1;
const RECOMMENDATION_COUNT = 10;
const ROW_COUNT = 20;
const SKELETON_COUNT = 8;
const SEARCH_DEBOUNCE_MS = 250;
const GENRE_ROWS = ["Action", "Comedy", "Sci-Fi", "Drama", "Animation", "Thriller", "Horror", "Romance"];

const GENRE_HUES = {
    Action: 4, Adventure: 28, Animation: 195, Children: 160, Comedy: 45, Crime: 220,
    Documentary: 100, Drama: 265, Fantasy: 290, "Film-Noir": 210, Horror: 350, IMAX: 200,
    Musical: 320, Mystery: 240, Romance: 335, "Sci-Fi": 185, Thriller: 230, War: 80, Western: 30,
};

const state = {
    userId: DEFAULT_USER_ID,
    seedId: DEFAULT_MOVIE_ID,
    seedToken: 0,
    searchToken: 0,
    searchTimer: null,
    rows: {},
};

const dom = {};

/* ---------- API ---------- */

async function api(path, params = {}) {
    const url = new URL(API_BASE + path);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") {
            url.searchParams.set(key, value);
        }
    });

    let response;
    try {
        response = await fetch(url, { headers: { Accept: "application/json" } });
    } catch (networkError) {
        throw new Error(`Cannot reach the API at ${API_BASE}`);
    }

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || `Request failed (HTTP ${response.status})`);
    }
    return data;
}

/* ---------- DOM helpers ---------- */

function el(tag, options = {}, children = []) {
    const node = document.createElement(tag);
    if (options.className) node.className = options.className;
    if (options.text !== undefined) node.textContent = options.text;
    Object.entries(options.attrs || {}).forEach(([name, value]) => node.setAttribute(name, value));
    children.forEach((child) => node.appendChild(child));
    return node;
}

function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
}

function hashString(text) {
    let hash = 0;
    for (const char of text) hash = (hash * 31 + char.codePointAt(0)) >>> 0;
    return hash;
}

function initialOf(title) {
    const match = /[\p{L}\p{N}]/u.exec(title || "");
    return match ? match[0].toUpperCase() : "?";
}

/** Deterministic "poster" gradient derived from the movie's main genre and title. */
function artBackground(movie) {
    const hash = hashString(movie.title || "");
    const baseHue = GENRE_HUES[movie.genres && movie.genres[0]] ?? hash % 360;
    const hue = (baseHue + (hash % 40) - 20 + 360) % 360;
    const accent = (hue + 35) % 360;
    return `radial-gradient(circle at 22% 18%, hsla(${hue}, 80%, 48%, 0.85), transparent 58%),
            radial-gradient(circle at 85% 90%, hsla(${accent}, 70%, 35%, 0.7), transparent 55%),
            linear-gradient(135deg, hsl(${hue}, 55%, 20%), hsl(${accent}, 60%, 7%))`;
}

function formatMeta(movie) {
    const parts = [];
    if (movie.year) parts.push(String(movie.year));
    if (movie.genres && movie.genres.length) parts.push(movie.genres.join(" \u2022 "));
    return parts;
}

/* ---------- Status banner ---------- */

function showBanner(message, { retry = false } = {}) {
    clear(dom.banner);
    dom.banner.appendChild(el("span", { text: message }));
    if (retry) {
        const button = el("button", { className: "btn btn-small", text: "Retry", attrs: { type: "button" } });
        button.addEventListener("click", () => window.location.reload());
        dom.banner.appendChild(button);
    }
    dom.banner.hidden = false;
}

function hideBanner() {
    dom.banner.hidden = true;
}

/* ---------- Movie cards ---------- */

function createCard(movie, { badge = null, matchFrom = null } = {}) {
    const label = movie.year ? `${movie.title} (${movie.year})` : movie.title;
    const card = el("button", { className: "card", attrs: { type: "button", "aria-label": label, title: label } });

    const art = el("span", { className: "card-art", attrs: { "aria-hidden": "true" } });
    art.style.background = artBackground(movie);
    art.appendChild(el("span", { className: "card-initial", text: initialOf(movie.title) }));
    card.appendChild(art);

    if (badge) card.appendChild(el("span", { className: "card-badge", text: badge }));

    const info = el("span", { className: "card-info" });
    info.appendChild(el("span", { className: "card-title", text: movie.title }));

    const meta = el("span", { className: "card-meta" });
    if (matchFrom && typeof movie[matchFrom] === "number") {
        meta.appendChild(el("span", { className: "card-match", text: `${Math.round(movie[matchFrom] * 100)}% Match` }));
    }
    if (movie.year) meta.appendChild(el("span", { className: "card-year", text: String(movie.year) }));
    if (typeof movie.predicted_rating === "number" && movie.predicted_rating > 0) {
        meta.appendChild(el("span", { className: "card-rating", text: `\u2605 ${movie.predicted_rating.toFixed(1)}` }));
    }
    info.appendChild(meta);

    if (movie.genres && movie.genres.length) {
        info.appendChild(el("span", { className: "card-genres", text: movie.genres.slice(0, 3).join(" \u2022 ") }));
    }
    card.appendChild(info);

    card.addEventListener("click", () => selectSeed(movie.movie_id));
    return card;
}

function createSkeletonCard() {
    return el("div", { className: "card skeleton", attrs: { "aria-hidden": "true" } });
}

/* ---------- Rows (carousels) ---------- */

function createRow(key, title) {
    const titleId = `row-${key}-title`;
    const section = el("section", { className: "row", attrs: { id: `row-${key}`, "aria-labelledby": titleId } });

    const heading = el("h2", { className: "row-title", text: title, attrs: { id: titleId } });
    const subtitle = el("span", { className: "row-subtitle" });
    section.appendChild(el("div", { className: "row-header" }, [heading, subtitle]));

    const track = el("div", { className: "row-track", attrs: { role: "list" } });
    const prev = el("button", {
        className: "row-arrow row-arrow-prev is-hidden",
        text: "\u2039",
        attrs: { type: "button", "aria-label": `Scroll ${title} left` },
    });
    const next = el("button", {
        className: "row-arrow row-arrow-next",
        text: "\u203A",
        attrs: { type: "button", "aria-label": `Scroll ${title} right` },
    });

    const row = { section, heading, subtitle, track, prev, next };
    prev.addEventListener("click", () => track.scrollBy({ left: -track.clientWidth * 0.85 }));
    next.addEventListener("click", () => track.scrollBy({ left: track.clientWidth * 0.85 }));
    track.addEventListener("scroll", () => updateArrows(row), { passive: true });

    section.appendChild(el("div", { className: "row-viewport" }, [prev, track, next]));
    dom.rows.appendChild(section);
    state.rows[key] = row;
    return row;
}

function updateArrows(row) {
    const { track, prev, next } = row;
    const maxScroll = track.scrollWidth - track.clientWidth;
    prev.classList.toggle("is-hidden", track.scrollLeft <= 4);
    next.classList.toggle("is-hidden", track.scrollLeft >= maxScroll - 4);
}

function setRowLoading(row) {
    clear(row.track);
    for (let i = 0; i < SKELETON_COUNT; i += 1) row.track.appendChild(createSkeletonCard());
    updateArrows(row);
}

function setRowMessage(row, message, isError = false) {
    clear(row.track);
    row.track.appendChild(el("p", { className: isError ? "row-message error" : "row-message", text: message }));
    updateArrows(row);
}

function setRowMovies(row, movies, cardOptions = {}) {
    if (!movies.length) {
        setRowMessage(row, "Nothing to show here yet.");
        return;
    }
    clear(row.track);
    movies.forEach((movie, index) => {
        const options = { ...cardOptions };
        if (cardOptions.rankBadges && index < 3) options.badge = `TOP ${index + 1}`;
        const card = createCard(movie, options);
        card.setAttribute("role", "listitem");
        row.track.appendChild(card);
    });
    row.track.scrollLeft = 0;
    requestAnimationFrame(() => updateArrows(row));
}

/* ---------- Hero banner ---------- */

function renderHero(movie, { eyebrow, description, match } = {}) {
    dom.heroArt.style.background = artBackground(movie);
    dom.heroInitial.textContent = initialOf(movie.title);
    dom.heroEyebrow.textContent = eyebrow || "Featured";
    dom.heroTitle.textContent = movie.title;

    clear(dom.heroMeta);
    if (typeof match === "number") {
        dom.heroMeta.appendChild(el("span", { className: "match", text: `${Math.round(match * 100)}% Match` }));
    }
    formatMeta(movie).forEach((part) => dom.heroMeta.appendChild(el("span", { text: part })));

    dom.heroDescription.textContent = description || "";
    document.title = `${movie.title} | Movify`;
}

function heroDescription(strategy, count) {
    if (strategy === "hybrid") {
        return `Personalised for User ${state.userId}: ${count} titles that share this movie's genres, ` +
            "re-ranked by how highly our SVD model predicts you'll rate them.";
    }
    return `User ${state.userId} has no trusted rating history yet, so these ${count} picks are ranked ` +
        "by genre similarity and overall popularity (cold start).";
}

/* ---------- Data loading ---------- */

async function selectSeed(movieId, { scroll = true } = {}) {
    state.seedId = Number(movieId);
    const token = ++state.seedToken;
    saveState();

    const { recommendations, similar } = state.rows;
    setRowLoading(recommendations);
    setRowLoading(similar);
    if (scroll) window.scrollTo({ top: 0 });

    const [recResult, simResult] = await Promise.allSettled([
        api("/recommendations/", { user_id: state.userId, movie_id: state.seedId, n: RECOMMENDATION_COUNT }),
        api(`/movies/${state.seedId}/similar/`, { n: ROW_COUNT }),
    ]);
    if (token !== state.seedToken) return; // A newer selection superseded this one.

    if (recResult.status === "fulfilled") {
        const data = recResult.value;
        hideBanner();
        recommendations.heading.textContent = `Top Picks for User ${data.user_id}`;
        recommendations.subtitle.textContent = `Because you watched ${data.seed_movie.title}`;
        setRowMovies(recommendations, data.results, { matchFrom: "score", rankBadges: true });
        renderHero(data.seed_movie, {
            eyebrow: data.strategy === "hybrid" ? "Because you watched" : "New viewer pick",
            description: heroDescription(data.strategy, data.count),
            match: data.results.length ? data.results[0].score : undefined,
        });
    } else {
        setRowMessage(recommendations, recResult.reason.message, true);
        showBanner(recResult.reason.message, { retry: true });
    }

    if (simResult.status === "fulfilled") {
        const data = simResult.value;
        similar.heading.textContent = `More Like ${data.seed_movie.title}`;
        similar.subtitle.textContent = "Genre similarity (TF-IDF)";
        setRowMovies(similar, data.results, { matchFrom: "similarity" });
        if (recResult.status !== "fulfilled") renderHero(data.seed_movie, { eyebrow: "Featured" });
    } else {
        setRowMessage(similar, simResult.reason.message, true);
    }
}

async function loadPopularRow(row, genre) {
    setRowLoading(row);
    try {
        const data = await api("/movies/popular/", { n: ROW_COUNT, genre });
        setRowMovies(row, data.results, { rankBadges: !genre });
    } catch (error) {
        setRowMessage(row, error.message, true);
    }
}

function loadBrowseRows() {
    loadPopularRow(state.rows["top-rated"]);
    GENRE_ROWS.forEach((genre) => loadPopularRow(state.rows[`genre-${genre}`], genre));
}

/* ---------- Search ---------- */

function setSearchOpen(open) {
    dom.searchResults.hidden = !open;
    dom.searchInput.setAttribute("aria-expanded", String(open));
}

function renderSearchResults(movies) {
    clear(dom.searchResults);
    if (!movies.length) {
        dom.searchResults.appendChild(el("li", { className: "search-message", text: "No matching titles." }));
    }
    movies.forEach((movie) => {
        const button = el("button", { className: "search-result", attrs: { type: "button", role: "option" } }, [
            el("span", { className: "search-result-title", text: movie.title }),
            el("span", { className: "search-result-meta", text: formatMeta(movie).join(" \u00B7 ") }),
        ]);
        button.addEventListener("click", () => {
            dom.searchInput.value = "";
            setSearchOpen(false);
            selectSeed(movie.movie_id);
        });
        dom.searchResults.appendChild(el("li", {}, [button]));
    });
    setSearchOpen(true);
}

async function runSearch(query) {
    const token = ++state.searchToken;
    try {
        const data = await api("/movies/search/", { q: query, n: 8 });
        if (token === state.searchToken) renderSearchResults(data.results);
    } catch (error) {
        if (token !== state.searchToken) return;
        clear(dom.searchResults);
        dom.searchResults.appendChild(el("li", { className: "search-message", text: error.message }));
        setSearchOpen(true);
    }
}

function onSearchInput() {
    clearTimeout(state.searchTimer);
    const query = dom.searchInput.value.trim();
    if (query.length < 2) {
        state.searchToken += 1;
        setSearchOpen(false);
        return;
    }
    state.searchTimer = setTimeout(() => runSearch(query), SEARCH_DEBOUNCE_MS);
}

/* ---------- User switching & persistence ---------- */

function onUserSubmit(event) {
    event.preventDefault();
    const value = Number(dom.userInput.value);
    const valid = Number.isInteger(value) && value >= 1;
    dom.userInput.classList.toggle("invalid", !valid);
    if (!valid) return;
    state.userId = value;
    selectSeed(state.seedId, { scroll: false });
}

function saveState() {
    try {
        localStorage.setItem("movify.userId", String(state.userId));
        const url = new URL(window.location.href);
        url.searchParams.set("user_id", state.userId);
        url.searchParams.set("movie_id", state.seedId);
        window.history.replaceState(null, "", url);
    } catch (error) {
        // Storage/history can be unavailable (e.g. privacy mode); state still works in memory.
    }
}

function loadInitialState() {
    const params = new URLSearchParams(window.location.search);
    let stored = null;
    try {
        stored = localStorage.getItem("movify.userId");
    } catch (error) {
        stored = null;
    }
    const userId = Number(params.get("user_id") || stored || DEFAULT_USER_ID);
    const movieId = Number(params.get("movie_id") || DEFAULT_MOVIE_ID);
    state.userId = Number.isInteger(userId) && userId > 0 ? userId : DEFAULT_USER_ID;
    state.seedId = Number.isInteger(movieId) && movieId > 0 ? movieId : DEFAULT_MOVIE_ID;
    dom.userInput.value = state.userId;
}

/* ---------- Bootstrapping ---------- */

function cacheDom() {
    dom.navbar = document.getElementById("navbar");
    dom.banner = document.getElementById("status-banner");
    dom.rows = document.getElementById("rows");
    dom.heroArt = document.getElementById("hero-art");
    dom.heroInitial = document.getElementById("hero-initial");
    dom.heroEyebrow = document.getElementById("hero-eyebrow");
    dom.heroTitle = document.getElementById("hero-title");
    dom.heroMeta = document.getElementById("hero-meta");
    dom.heroDescription = document.getElementById("hero-description");
    dom.searchInput = document.getElementById("search-input");
    dom.searchResults = document.getElementById("search-results");
    dom.search = document.getElementById("search");
    dom.userForm = document.getElementById("user-form");
    dom.userInput = document.getElementById("user-input");
}

function buildRows() {
    createRow("recommendations", "Top Picks for You");
    createRow("similar", "More Like This");
    createRow("top-rated", "Top Rated on Movify");
    GENRE_ROWS.forEach((genre) => createRow(`genre-${genre}`, `${genre} Favourites`));
    window.addEventListener("resize", () => Object.values(state.rows).forEach(updateArrows));
}

function bindEvents() {
    window.addEventListener("scroll", () => {
        dom.navbar.classList.toggle("scrolled", window.scrollY > 10);
    }, { passive: true });

    dom.searchInput.addEventListener("input", onSearchInput);
    dom.searchInput.addEventListener("keydown", (event) => {
        if (event.key === "Escape") setSearchOpen(false);
    });
    document.addEventListener("click", (event) => {
        if (!dom.search.contains(event.target)) setSearchOpen(false);
    });

    dom.userForm.addEventListener("submit", onUserSubmit);

    document.getElementById("hero-picks").addEventListener("click", () => {
        state.rows.recommendations.section.scrollIntoView({ behavior: "smooth" });
    });
    document.getElementById("hero-similar").addEventListener("click", () => {
        state.rows.similar.section.scrollIntoView({ behavior: "smooth" });
    });
}

async function init() {
    cacheDom();
    loadInitialState();
    buildRows();
    bindEvents();

    try {
        await api("/health/");
    } catch (error) {
        dom.heroTitle.textContent = "Backend offline";
        dom.heroDescription.textContent =
            "Start the Django server (python manage.py runserver) and reload this page.";
        showBanner(`${error.message}. Is the Django server running?`, { retry: true });
        Object.values(state.rows).forEach((row) => setRowMessage(row, "Waiting for the backend\u2026", true));
        return;
    }

    selectSeed(state.seedId, { scroll: false });
    loadBrowseRows();
}

document.addEventListener("DOMContentLoaded", init);
