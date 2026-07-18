(() => {
  const patterns = {
    python:
      /(#.*$)|('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")|\b(and|as|assert|async|await|break|case|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|match|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)\b|\b(\d+(?:\.\d+)?)\b|(@[A-Za-z_]\w*)|\b([A-Za-z_]\w*)(?=\s*\()/g,
    bash:
      /(#.*$)|('(?:[^']*)'|"(?:\\.|[^"\\])*")|(\$[A-Za-z_]\w*|\$\{[^}]+\})|(--?[\w-]+)|\b(activate|cat|cd|macllm|open|pip|python|python3|source)\b|\b(\d+(?:\.\d+)?)\b/g,
  };

  const classes = {
    python: ["comment", "string", "keyword", "number", "decorator", "function"],
    bash: ["comment", "string", "variable", "option", "command", "number"],
  };

  function appendHighlighted(target, source, language) {
    const pattern = patterns[language];
    const names = classes[language];
    let cursor = 0;

    for (const match of source.matchAll(pattern)) {
      target.append(document.createTextNode(source.slice(cursor, match.index)));
      const group = match.slice(1).findIndex((value) => value !== undefined);
      const token = document.createElement("span");
      token.className = `syntax-${names[group]}`;
      token.textContent = match[0];
      target.append(token);
      cursor = match.index + match[0].length;
    }
    target.append(document.createTextNode(source.slice(cursor)));
  }

  function textWithBreaks(node) {
    return [...node.childNodes]
      .map((child) => (child.nodeName === "BR" ? "\n" : child.textContent))
      .join("");
  }

  function highlightBlock(code, language = "bash") {
    const source = textWithBreaks(code);
    code.replaceChildren();
    source.split("\n").forEach((line, index, lines) => {
      appendHighlighted(code, line, language);
      if (index < lines.length - 1) code.append(document.createElement("br"));
    });
    code.dataset.highlighted = "true";
  }

  window.macllmHighlight = { appendHighlighted, highlightBlock };

  document.querySelectorAll(".command-block code, .next-command code").forEach((code) => {
    highlightBlock(code, "bash");
  });
})();
