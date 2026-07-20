const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function makeStepper(root, stages, render) {
  let index = 0;
  let timer = null;
  const previous = root.querySelector("[data-step-previous]");
  const next = root.querySelector("[data-step-next]");
  const play = root.querySelector("[data-step-play]");

  const show = (nextIndex) => {
    index = (nextIndex + stages.length) % stages.length;
    render(stages[index], index);
    previous.disabled = index === 0;
    next.textContent = index === stages.length - 1 ? "Start again" : "Next step";
  };

  const stop = () => {
    window.clearInterval(timer);
    timer = null;
    play.textContent = "Play";
    play.setAttribute("aria-pressed", "false");
  };

  previous.addEventListener("click", () => {
    stop();
    show(index - 1);
  });
  next.addEventListener("click", () => {
    stop();
    show(index + 1);
  });
  play.addEventListener("click", () => {
    if (timer) {
      stop();
      return;
    }
    play.textContent = "Pause";
    play.setAttribute("aria-pressed", "true");
    timer = window.setInterval(() => show(index + 1), 1450);
  });
  show(0);
  if (reducedMotion) play.textContent = "Step automatically";
}

function setStageText(root, stage, index, total) {
  root.querySelector("[data-step-count]").textContent = `${index + 1} / ${total}`;
  root.querySelector("[data-step-title]").textContent = stage.title;
  root.querySelector("[data-step-detail]").textContent = stage.detail;
}

document.querySelectorAll("[data-neuron-demo]").forEach((root) => {
  const stages = [
    {
      key: "inputs",
      title: "Inputs arrive as numbers",
      detail: "This tiny example receives x₁ = 0.8 and x₂ = −0.4. A real token vector contains hundreds of numbers.",
    },
    {
      key: "weighted",
      title: "Weights scale every connection",
      detail: "Each neuron multiplies inputs by learned weights, adds them, then applies an activation. The result becomes the next layer’s input.",
    },
    {
      key: "prediction",
      title: "The forward pass makes a prediction",
      detail: "Two hidden layers reduce to one output: 0.62. The target is 1.00, so the loss records a difference of 0.38.",
    },
    {
      key: "backward",
      title: "Backpropagation sends blame backward",
      detail: "The chain rule follows the exact operations in reverse and computes one gradient—an up or down hint—for every weight.",
    },
    {
      key: "update",
      title: "The optimizer nudges the weights",
      detail: "AdamW turns gradients into small updates. The next forward pass uses the new weights and should make a slightly better prediction.",
    },
  ];
  makeStepper(root, stages, (stage, index) => {
    root.dataset.phase = stage.key;
    setStageText(root, stage, index, stages.length);
    root.querySelectorAll("[data-weight]").forEach((label) => {
      label.textContent = stage.key === "update" ? label.dataset.updated : label.dataset.weight;
    });
  });
});

document.querySelectorAll("[data-training-demo]").forEach((root) => {
  const stages = [
    { key: "batch", title: "1 · Make prediction pairs", detail: "Input IDs [41, 92, 8] are paired with targets [92, 8, 17]: each target is one token ahead." },
    { key: "forward", title: "2 · Run the transformer forward", detail: "Embedding vectors pass through attention and feed-forward neurons in every block. Learned weights stay fixed during this pass." },
    { key: "loss", title: "3 · Compare logits with targets", detail: "Cross-entropy turns all wrong next-token scores into one loss: 2.31 in this miniature step." },
    { key: "backward", title: "4 · Backpropagate gradients", detail: "Automatic differentiation walks from loss to logits, blocks, embeddings, and every weight in reverse operation order." },
    { key: "clip", title: "5 · Clip one global gradient norm", detail: "If the combined gradient is too large, every gradient is scaled together. Direction is preserved; the step becomes safer." },
    { key: "update", title: "6 · AdamW updates the weights", detail: "The optimizer changes parameters, not activations or token IDs. The next batch begins a new forward pass with those updated numbers." },
  ];
  makeStepper(root, stages, (stage, index) => {
    root.dataset.phase = stage.key;
    setStageText(root, stage, index, stages.length);
  });
});

document.querySelectorAll("[data-inference-demo]").forEach((root) => {
  const generated = [
    { token: "little", id: 117, probability: "46%" },
    { token: "robot", id: 64, probability: "38%" },
    { token: "learned", id: 205, probability: "31%" },
  ];
  let cycle = 0;
  const stages = [
    { key: "encode", title: "1 · Encode the prompt", detail: "The tokenizer maps visible text pieces to integer IDs. IDs are indexes, not meanings." },
    { key: "embed", title: "2 · Look up vectors", detail: "Each ID selects one learned embedding vector. Position rotation later lets attention distinguish token order." },
    { key: "transform", title: "3 · Refine vectors and keep a KV cache", detail: "Attention mixes visible context; feed-forward neurons transform each position. Every layer stores keys and values for reuse." },
    { key: "scores", title: "4 · Decode the final vector into logits", detail: "The last position is projected to one score per vocabulary token. Softmax turns those scores into probabilities." },
    { key: "choose", title: "5 · Choose one token", detail: "Greedy decoding picks the largest score; sampling can pick another likely token. Only one ID continues the loop." },
    { key: "append", title: "6 · Append it and loop", detail: "The chosen output becomes the next input. Only that new token runs through the model while cached keys and values represent the earlier prefix." },
  ];

  makeStepper(root, stages, (stage, index) => {
    root.dataset.phase = stage.key;
    setStageText(root, stage, index, stages.length);
    const next = generated[cycle % generated.length];
    root.querySelector("[data-next-token]").textContent = next.token;
    root.querySelector("[data-next-id]").textContent = next.id;
    root.querySelector("[data-next-probability]").textContent = next.probability;
    if (stage.key === "append") {
      const tokenRow = root.querySelector("[data-token-row]");
      if (![...tokenRow.children].some((item) => item.dataset.cycle === String(cycle))) {
        const token = document.createElement("span");
        token.dataset.cycle = String(cycle);
        token.className = "generated-token";
        token.innerHTML = `${next.token}<small>${next.id}</small>`;
        tokenRow.append(token);
        root.querySelector("[data-cache-count]").textContent = String(tokenRow.children.length);
        cycle += 1;
      }
    }
  });
});
