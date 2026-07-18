const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

// Theme
const root = document.documentElement;
const themeButton = $("#theme-button");
const savedTheme = localStorage.getItem("macllm-theme");
if (savedTheme) root.dataset.theme = savedTheme;

themeButton.addEventListener("click", () => {
  const next = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = next;
  localStorage.setItem("macllm-theme", next);
  themeButton.setAttribute("aria-label", `Switch to ${next === "light" ? "dark" : "light"} theme`);
});

// Preset explorer
const presets = {
  quick: {
    label: "Quick · prove the pipeline",
    params: "8.0",
    layers: 6,
    tokens: "6M",
    context: "256",
    time: "10–30 min",
    timeWidth: "18%",
    purpose: "The complete loop in minutes: prepare, train, generate, inspect, and quantize.",
  },
  standard: {
    label: "Standard · recommended",
    params: "56.9",
    layers: 12,
    tokens: "82M",
    context: "512",
    time: "3–8 h",
    timeWidth: "52%",
    purpose: "The practical center: coherent tiny stories and a real afternoon training run.",
  },
  overnight: {
    label: "Overnight · push quality",
    params: "113.3",
    layers: 16,
    tokens: "205M",
    context: "512",
    time: "12–30 h",
    timeWidth: "88%",
    purpose: "More capacity and more updates when waiting longer is part of the experiment.",
  },
};

function renderPreset(name) {
  const preset = presets[name];
  $("#preset-name").textContent = preset.label;
  $("#preset-params").textContent = preset.params;
  $("#preset-purpose").textContent = preset.purpose;
  $("#preset-tokens").textContent = preset.tokens;
  $("#preset-context").textContent = preset.context;
  $("#preset-time").textContent = preset.time;
  $("#stack-layers").textContent = preset.layers;
  $("#time-fill").style.width = preset.timeWidth;

  const stack = $("#layer-stack");
  stack.replaceChildren();
  for (let index = 0; index < preset.layers; index += 1) {
    const layer = document.createElement("i");
    layer.style.setProperty("--layer-width", `${190 + index * 6}px`);
    stack.append(layer);
  }
}

$$('[data-preset]').forEach((button) => {
  button.addEventListener("click", () => {
    $$('[data-preset]').forEach((item) => item.classList.toggle("is-selected", item === button));
    renderPreset(button.dataset.preset);
  });
});
renderPreset("standard");

// Token shift lab
const tokenInput = $("#token-input");

function previewTokens(text) {
  return text.match(/\S+|\s+/g)?.filter((piece) => piece.trim()) ?? [];
}

function renderTokenShift() {
  const pieces = previewTokens(tokenInput.value).slice(0, 8);
  const grid = $("#token-grid");
  grid.replaceChildren();

  if (pieces.length < 2) {
    const message = document.createElement("p");
    message.textContent = "Add at least two words to make a prediction.";
    grid.append(message);
    $("#lesson-count").textContent = "0";
    return;
  }

  for (let index = 0; index < pieces.length - 1; index += 1) {
    const input = document.createElement("span");
    input.className = "token-cell";
    input.textContent = pieces[index];
    const arrow = document.createElement("span");
    arrow.className = "shift-arrow";
    arrow.textContent = "→";
    const target = document.createElement("span");
    target.className = "token-cell token-target";
    target.textContent = pieces[index + 1];
    grid.append(input, arrow, target);
  }
  $("#lesson-count").textContent = pieces.length - 1;
}

tokenInput.addEventListener("input", renderTokenShift);
renderTokenShift();

// Causal attention lab
const attentionTokens = ["Once", "upon", "a", "little", "time"];
const queryPosition = $("#query-position");

function renderAttention() {
  const selected = Number(queryPosition.value);
  const matrix = $("#attention-matrix");
  matrix.replaceChildren();

  const corner = document.createElement("span");
  corner.className = "matrix-cell matrix-label";
  corner.textContent = "query ↓ key →";
  matrix.append(corner);
  attentionTokens.forEach((token) => {
    const label = document.createElement("span");
    label.className = "matrix-cell matrix-label";
    label.textContent = token;
    matrix.append(label);
  });

  attentionTokens.forEach((query, row) => {
    const label = document.createElement("span");
    label.className = `matrix-cell matrix-label ${row > selected ? "future-row" : ""}`;
    label.textContent = query;
    matrix.append(label);

    attentionTokens.forEach((_, column) => {
      const cell = document.createElement("span");
      const visible = column <= row;
      cell.className = [
        "matrix-cell",
        visible ? "can-see" : "",
        row === selected ? "current-row" : "",
        row > selected ? "future-row" : "",
      ].join(" ");
      cell.textContent = visible ? "●" : "·";
      cell.setAttribute("aria-label", `${query} ${visible ? "can" : "cannot"} attend to ${attentionTokens[column]}`);
      matrix.append(cell);
    });
  });

  $("#query-token").textContent = attentionTokens[selected];
  const visible = attentionTokens.slice(0, selected + 1).join(", ");
  $("#attention-detail").textContent = `“${attentionTokens[selected]}” may read: ${visible}. Future tokens remain hidden.`;
}

queryPosition.addEventListener("input", renderAttention);
renderAttention();

// Transformer block explainer
const architectureText = {
  attention: "Attention lets tokens exchange context. Each position gathers useful information from earlier positions.",
  ffn: "SwiGLU computes independently at every position, transforming the context attention just collected.",
  residual: "The residual stream adds each update to the old representation, preserving a direct path through depth.",
};

$$('[data-architecture]').forEach((button) => {
  button.addEventListener("click", () => {
    $$('[data-architecture]').forEach((item) => item.classList.toggle("is-selected", item === button));
    $$(".block-node").forEach((node) => node.classList.remove("is-active"));
    const stage = button.dataset.architecture;
    if (stage === "attention") $(".node-attention").classList.add("is-active");
    if (stage === "ffn") $(".node-ffn").classList.add("is-active");
    if (stage === "residual") $$(".merge-node").forEach((node) => node.classList.add("is-active"));
    $("#architecture-detail").textContent = architectureText[stage];
  });
});

// Loss chart
const lossScenarios = {
  healthy: {
    train: [4.2, 3.75, 3.25, 2.87, 2.58, 2.34, 2.18, 2.06, 1.98],
    valid: [4.25, 3.84, 3.38, 3.02, 2.76, 2.56, 2.43, 2.37, 2.33],
    diagnosis: "Both lines fall and remain close: useful patterns transfer to unseen stories.",
    description: "Healthy training where training and validation losses both decrease.",
  },
  overfit: {
    train: [4.2, 3.65, 3.05, 2.52, 2.05, 1.67, 1.35, 1.09, 0.88],
    valid: [4.25, 3.82, 3.42, 3.17, 3.04, 3.08, 3.22, 3.43, 3.68],
    diagnosis: "Training improves while validation reverses: the model is memorizing. Stop or add better data.",
    description: "Overfitting where training loss falls while validation loss rises late in training.",
  },
  underfit: {
    train: [4.2, 4.05, 3.95, 3.88, 3.82, 3.78, 3.75, 3.72, 3.7],
    valid: [4.25, 4.12, 4.03, 3.97, 3.92, 3.89, 3.87, 3.85, 3.84],
    diagnosis: "Both lines flatten high: the model needs more updates, better data, or more capacity.",
    description: "Underfitting where both training and validation losses remain high.",
  },
};

function lineGeometry(values) {
  const left = 52;
  const top = 38;
  const width = 600;
  const height = 250;
  const min = 0.7;
  const max = 4.5;
  const points = values.map((value, index) => ({
    x: left + (index * width) / (values.length - 1),
    y: top + ((max - value) * height) / (max - min),
  }));
  return {
    path: points.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" "),
    end: points.at(-1),
  };
}

function renderLoss(name) {
  const scenario = lossScenarios[name];
  const train = lineGeometry(scenario.train);
  const valid = lineGeometry(scenario.valid);
  $("#train-line").setAttribute("d", train.path);
  $("#valid-line").setAttribute("d", valid.path);
  [["#train-end", train.end], ["#valid-end", valid.end]].forEach(([selector, point]) => {
    $(selector).setAttribute("cx", point.x);
    $(selector).setAttribute("cy", point.y);
  });
  [["#train-label", train.end], ["#valid-label", valid.end]].forEach(([selector, point]) => {
    $(selector).setAttribute("x", point.x + 10);
    $(selector).setAttribute("y", point.y + 4);
  });
  $("#loss-diagnosis").textContent = scenario.diagnosis;
  $("#loss-chart-desc").textContent = scenario.description;
}

const chartGrid = $(".chart-grid");
for (let index = 0; index < 5; index += 1) {
  const horizontal = document.createElementNS("http://www.w3.org/2000/svg", "line");
  horizontal.setAttribute("x1", "52");
  horizontal.setAttribute("x2", "652");
  horizontal.setAttribute("y1", String(38 + index * 62.5));
  horizontal.setAttribute("y2", String(38 + index * 62.5));
  chartGrid.append(horizontal);
}

$$('[data-loss]').forEach((button) => {
  button.addEventListener("click", () => {
    $$('[data-loss]').forEach((item) => item.classList.toggle("is-selected", item === button));
    renderLoss(button.dataset.loss);
  });
});
renderLoss("healthy");

// Fine-tuning weight fields
for (let index = 0; index < 120; index += 1) {
  const full = document.createElement("i");
  full.style.setProperty("--shade", `${20 + (index % 7) * 6}%`);
  $("#full-weight-field").append(full);

  const lora = document.createElement("i");
  if ([7, 31, 53, 79, 103, 116].includes(index)) lora.className = "adapter-weight";
  $("#lora-weight-field").append(lora);
}

// Quantization lab
const quantNotes = {
  8: "A conservative reduction: larger files, but values stay closer to the original weights.",
  6: "A middle ground when 4-bit changes an answer you care about.",
  4: "The practical default: strong compression with usually modest quality change.",
  3: "Aggressive compression. Test carefully; fewer levels can visibly change behavior.",
};

function renderQuantization(bits) {
  const levels = 2 ** bits;
  const size = Math.round((56.9e6 * bits / 8 / 1024 ** 2) * 1.18);
  $("#quant-label").textContent = `${bits}-bit MLX`;
  $("#quant-size").textContent = size;
  $("#level-count").textContent = levels.toLocaleString();
  $("#compression-ratio").textContent = `${(217 / size).toFixed(1)}× smaller`;
  $("#quant-vessel").style.height = `${Math.max(11, (size / 217) * 100)}%`;
  $("#quant-caution").textContent = quantNotes[bits];

  const visual = $("#levels-visual");
  visual.replaceChildren();
  const bars = Math.min(levels, 64);
  for (let index = 0; index < bars; index += 1) {
    const bar = document.createElement("i");
    bar.style.setProperty("--level-height", String(index / Math.max(1, bars - 1)));
    visual.append(bar);
  }
}

$$('[data-bits]').forEach((button) => {
  button.addEventListener("click", () => {
    $$('[data-bits]').forEach((item) => item.classList.toggle("is-selected", item === button));
    renderQuantization(Number(button.dataset.bits));
  });
});
renderQuantization(4);

// Reveal dense sections only when they approach the viewport.
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

$$('.reveal, .lab, .act-heading, .run-steps > li').forEach((element) => {
  element.classList.add("reveal-ready");
  revealObserver.observe(element);
});

if (location.hash) {
  requestAnimationFrame(() => document.getElementById(location.hash.slice(1))?.scrollIntoView());
}

// Copyable commands
$$('[data-copy]').forEach((button) => {
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
