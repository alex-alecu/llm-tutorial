const courseRoot = document.documentElement;
const courseThemeButton = document.querySelector("#theme-button");
const storedCourseTheme = localStorage.getItem("macllm-theme");

if (storedCourseTheme) courseRoot.dataset.theme = storedCourseTheme;

courseThemeButton?.addEventListener("click", () => {
  const next = courseRoot.dataset.theme === "light" ? "dark" : "light";
  courseRoot.dataset.theme = next;
  localStorage.setItem("macllm-theme", next);
  courseThemeButton.setAttribute(
    "aria-label",
    `Switch to ${next === "light" ? "dark" : "light"} theme`,
  );
});

const chapterHeader = document.querySelector(".chapter-header");
if (chapterHeader && courseThemeButton) {
  const actions = document.createElement("div");
  actions.className = "header-actions";
  const githubLink = document.createElement("a");
  githubLink.className = "github-link";
  githubLink.href = "https://github.com/alex-alecu/llm-tutorial";
  githubLink.target = "_blank";
  githubLink.rel = "noreferrer";
  githubLink.setAttribute("aria-label", "View project on GitHub");
  githubLink.innerHTML = `
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path fill="currentColor" d="M8 0C3.58 0 0 3.64 0 8.13c0 3.59 2.29 6.64 5.47 7.71.4.08.55-.18.55-.39 0-.19-.01-.83-.01-1.51-2.01.38-2.53-.5-2.69-.96-.09-.23-.48-.96-.82-1.15-.28-.15-.68-.53-.01-.54.63-.01 1.08.59 1.23.83.72 1.23 1.87.88 2.33.67.07-.53.28-.88.51-1.08-1.78-.21-3.64-.91-3.64-4.02 0-.89.31-1.62.82-2.19-.08-.21-.36-1.04.08-2.16 0 0 .67-.22 2.2.84A7.5 7.5 0 0 1 8 3.91c.68 0 1.36.09 2 .27 1.53-1.06 2.2-.84 2.2-.84.44 1.12.16 1.95.08 2.16.51.57.82 1.3.82 2.19 0 3.12-1.87 3.81-3.65 4.02.29.25.54.74.54 1.5 0 1.08-.01 1.95-.01 2.22 0 .22.15.47.55.39A8.16 8.16 0 0 0 16 8.13C16 3.64 12.42 0 8 0Z"/>
    </svg>`;
  courseThemeButton.before(actions);
  actions.append(githubLink, courseThemeButton);
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    await navigator.clipboard.writeText(button.dataset.copy);
    const oldLabel = button.textContent;
    button.textContent = "Copied";
    button.classList.add("copied");
    window.setTimeout(() => {
      button.textContent = oldLabel;
      button.classList.remove("copied");
    }, 1400);
  });
});

function renderSource(code, source) {
  code.replaceChildren();
  source.replace(/\r\n/g, "\n").split("\n").forEach((line, index, lines) => {
    if (index === lines.length - 1 && line === "") return;
    const row = document.createElement("span");
    row.className = "source-line";
    row.dataset.line = String(index + 1);
    window.macllmHighlight.appendHighlighted(row, line || " ", "python");
    code.append(row);
  });
}

function parseLineRange(button) {
  const [start, end = start] = button.dataset.lines.split("-").map(Number);
  return { start, end };
}

function focusSourceRange(viewer, button, { scrollBehavior = null } = {}) {
  const { start, end } = parseLineRange(button);
  const lines = [...viewer.querySelectorAll(".source-line")];
  lines.forEach((line) => {
    const number = Number(line.dataset.line);
    line.classList.toggle("is-focused", number >= start && number <= end);
    line.classList.toggle("is-range-start", number === start);
    line.classList.toggle("is-range-end", number === end);
  });
  viewer.classList.add("has-focused-lines");
  const map = button.closest(".code-reading-map");
  map?.querySelectorAll(".code-note").forEach((note) => {
    const active = note.contains(button);
    note.classList.toggle("is-active", active);
    note.querySelector("button")?.setAttribute("aria-pressed", String(active));
  });
  const rangeLabel = start === end ? `Line ${start}` : `Lines ${start}–${end}`;
  viewer.dataset.focusedLines = rangeLabel;
  const noteTitle = button.closest(".code-note")?.querySelector("h3")?.textContent;
  const focusStatus = viewer.querySelector("[data-source-focus-status]");
  if (focusStatus) focusStatus.textContent = noteTitle ? `${rangeLabel} · ${noteTitle}` : rangeLabel;
  if (!scrollBehavior) return;
  const first = lines[start - 1];
  const last = lines[end - 1];
  const scroller = viewer.querySelector("pre");
  if (!first || !last || !scroller) return;
  requestAnimationFrame(() => {
    const scrollerTop = scroller.getBoundingClientRect().top;
    const rangeTop = first.getBoundingClientRect().top - scrollerTop + scroller.scrollTop;
    const rangeBottom = last.getBoundingClientRect().bottom - scrollerTop + scroller.scrollTop;
    const rangeHeight = rangeBottom - rangeTop;
    const target = rangeHeight > scroller.clientHeight - 48
      ? rangeTop - 20
      : (rangeTop + rangeBottom) / 2 - scroller.clientHeight / 2;
    scroller.scrollTo({ top: Math.max(0, target), behavior: scrollBehavior });
  });
}

function revealExplanation(map, note, behavior = "smooth") {
  if (!map || !note || map.scrollHeight <= map.clientHeight) return;
  const mapRect = map.getBoundingClientRect();
  const noteRect = note.getBoundingClientRect();
  const inset = 12;
  if (noteRect.top >= mapRect.top + inset && noteRect.bottom <= mapRect.bottom - inset) return;
  const noteTop = noteRect.top - mapRect.top + map.scrollTop;
  const target = noteTop - (map.clientHeight - noteRect.height) / 2;
  map.scrollTo({ top: Math.max(0, target), behavior });
}

function buildGuidedReaders() {
  const supportsHover = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  document.querySelectorAll(".source-viewer").forEach((viewer) => {
    const map = viewer.previousElementSibling;
    if (!map?.classList.contains("code-reading-map")) return;
    const buttons = [...map.querySelectorAll(`[data-focus-source="${viewer.id}"]`)];
    if (!buttons.length) return;
    const reader = document.createElement("div");
    reader.className = "guided-code";
    reader.setAttribute("aria-label", "Guided source explanation and exact code");
    map.before(reader);
    reader.append(map, viewer);
    viewer.open = true;
    map.setAttribute("aria-label", "Choose an explanation to highlight its exact code lines");
    const focusStatus = document.createElement("span");
    focusStatus.className = "source-focus-status";
    focusStatus.dataset.sourceFocusStatus = "";
    focusStatus.textContent = "Choose an explanation";
    viewer.querySelector(".source-meta")?.prepend(focusStatus);
    let hoverTimer = null;
    let activationVersion = 0;
    let lastSourceButton = null;

    const activate = async (button, scrollBehavior, { revealNote = false } = {}) => {
      const version = ++activationVersion;
      window.clearTimeout(hoverTimer);
      const loadedViewer = await loadSource(viewer);
      if (version !== activationVersion || loadedViewer.classList.contains("source-load-failed")) {
        return;
      }
      loadedViewer.open = true;
      focusSourceRange(loadedViewer, button, { scrollBehavior });
      if (revealNote) revealExplanation(map, button.closest(".code-note"));
    };

    const buttonForLine = (line) => {
      const number = Number(line?.dataset.line);
      if (!number) return null;
      return buttons.find((button) => {
        const { start, end } = parseLineRange(button);
        return number >= start && number <= end;
      }) || null;
    };

    const connectSourceLines = (loadedViewer) => {
      loadedViewer.querySelectorAll(".source-line").forEach((line) => {
        line.classList.toggle("has-explanation", Boolean(buttonForLine(line)));
      });
    };

    buttons.forEach((button) => {
      const note = button.closest(".code-note");
      button.type = "button";
      button.setAttribute("aria-controls", viewer.id);
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `${button.textContent}: ${note?.querySelector("h3")?.textContent}`);
      button.addEventListener("focus", () => activate(button, "smooth"));
      note?.addEventListener("click", () => activate(button, "smooth"));
      if (supportsHover) note?.addEventListener("pointerenter", () => {
        window.clearTimeout(hoverTimer);
        hoverTimer = window.setTimeout(() => activate(button, "smooth"), 90);
      });
      note?.addEventListener("pointerleave", () => window.clearTimeout(hoverTimer));
    });
    const sourceScroller = viewer.querySelector("pre");
    if (supportsHover) sourceScroller?.addEventListener("pointermove", (event) => {
      const line = event.target.closest(".source-line");
      const button = line && sourceScroller.contains(line) ? buttonForLine(line) : null;
      if (!button || button === lastSourceButton) return;
      lastSourceButton = button;
      window.clearTimeout(hoverTimer);
      hoverTimer = window.setTimeout(() => activate(button, null, { revealNote: true }), 70);
    });
    sourceScroller?.addEventListener("pointerleave", () => {
      lastSourceButton = null;
      window.clearTimeout(hoverTimer);
    });
    loadSource(viewer).then(connectSourceLines);
  });
}

const sourceLoads = new Map();

function loadSource(viewer) {
  if (sourceLoads.has(viewer.id)) return sourceLoads.get(viewer.id);
  const sourcePath = location.pathname.includes("/site/")
    ? viewer.dataset.pythonSource
    : viewer.dataset.pythonSource.replace(/^\.\.\/\.\.\//, "../");
  const code = viewer.querySelector("code");
  const status = viewer.querySelector("[data-source-status]");
  viewer.querySelector(".source-meta a").href = sourcePath;
  const load = fetch(sourcePath)
    .then((response) => {
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.text();
    })
    .then((source) => {
      renderSource(code, source);
      status.textContent = `${code.children.length} lines · exact repository source`;
      return viewer;
    })
    .catch(() => {
      status.innerHTML =
        'Source loading needs the local course server. Run <code>make serve</code> from the repository root, then reopen this page through <code>localhost:8000</code>.';
      viewer.classList.add("source-load-failed");
      return viewer;
    });
  sourceLoads.set(viewer.id, load);
  return load;
}

document.querySelectorAll("[data-python-source]").forEach((viewer) => {
  const status = viewer.querySelector("[data-source-status]");
  status.textContent = "Source loads when opened";
  viewer.addEventListener("toggle", () => {
    if (viewer.open) loadSource(viewer);
  });
  if (viewer.open) {
    const schedule = window.requestIdleCallback ?? ((task) => window.setTimeout(task, 250));
    schedule(() => loadSource(viewer));
  }
});

buildGuidedReaders();

document.querySelectorAll("[data-focus-source]").forEach((button) => {
  if (button.closest(".guided-code")) return;
  button.addEventListener("click", async () => {
    const target = document.querySelector(`#${button.dataset.focusSource}`);
    const viewer = target ? await loadSource(target) : null;
    if (!viewer || viewer.classList.contains("source-load-failed")) return;
    viewer.open = true;
    focusSourceRange(viewer, button, { scrollBehavior: "smooth" });
  });
});

const progress = document.querySelector("#reading-progress");
if (progress) {
  const updateProgress = () => {
    const available = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${available > 0 ? (window.scrollY / available) * 100 : 0}%`;
  };
  window.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();
}

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      revealObserver.unobserve(entry.target);
    });
  },
  { rootMargin: "100px 0px" },
);

document.querySelectorAll(".lesson-section, .checkpoint-card, .chapter-pager").forEach((element) => {
  element.classList.add("reveal-ready");
  revealObserver.observe(element);
});

if (location.hash) {
  const scrollToChapterAnchor = () => {
    document.getElementById(location.hash.slice(1))?.scrollIntoView();
  };
  requestAnimationFrame(scrollToChapterAnchor);
  // Opening guided readers changes page height; settle the anchor again after their source loads.
  Promise.allSettled(sourceLoads.values()).then(() => requestAnimationFrame(scrollToChapterAnchor));
}
