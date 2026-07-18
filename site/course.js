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
    row.textContent = line || " ";
    code.append(row);
  });
}

const sourceLoads = new Map();

document.querySelectorAll("[data-python-source]").forEach((viewer) => {
  const sourcePath = viewer.dataset.pythonSource;
  const code = viewer.querySelector("code");
  const status = viewer.querySelector("[data-source-status]");
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
        'Source loading needs the local course server. Run <code>python3 -m http.server 8000</code> from the repository root, then reopen this page through <code>localhost:8000</code>.';
      viewer.classList.add("source-load-failed");
      return viewer;
    });
  sourceLoads.set(viewer.id, load);
});

document.querySelectorAll("[data-focus-source]").forEach((button) => {
  button.addEventListener("click", async () => {
    const viewer = await sourceLoads.get(button.dataset.focusSource);
    if (!viewer || viewer.classList.contains("source-load-failed")) return;
    const [start, end = start] = button.dataset.lines.split("-").map(Number);
    viewer.querySelectorAll(".source-line").forEach((line) => {
      const number = Number(line.dataset.line);
      line.classList.toggle("is-focused", number >= start && number <= end);
    });
    const first = viewer.querySelector(`.source-line[data-line="${start}"]`);
    viewer.open = true;
    first?.scrollIntoView({ behavior: "smooth", block: "center" });
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
