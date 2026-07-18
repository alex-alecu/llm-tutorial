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
  githubLink.innerHTML = 'GitHub <span aria-hidden="true">↗</span>';
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
