(function () {
  const launcher = document.getElementById("chatbotLauncher");
  const panel = document.getElementById("chatbotPanel");
  const minimizeBtn =
    document.getElementById("chatbotMinimize") ||
    document.getElementById("chatbotClose") ||
    panel?.querySelector(".chatbot-control");
  const form = document.getElementById("chatbotForm");
  const input = document.getElementById("chatbotInput");
  const thread = document.getElementById("chatbotThread");
  const resizeHandle = document.getElementById("chatbotResizeHandle");

  if (!launcher || !panel || !form || !input || !thread) {
    return;
  }

  function setOpenState(isOpen) {
    panel.classList.toggle("is-open", isOpen);
    launcher.setAttribute("aria-expanded", String(isOpen));
    panel.setAttribute("aria-hidden", String(!isOpen));
    if (isOpen) {
      window.setTimeout(() => input.focus(), 50);
    }
  }

  function enableResize() {
    if (!resizeHandle) {
      return;
    }

    let isResizing = false;
    let startX = 0;
    let startY = 0;
    let startWidth = 0;
    let startHeight = 0;

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    resizeHandle.addEventListener("pointerdown", (event) => {
      if (window.innerWidth <= 900) {
        return;
      }
      isResizing = true;
      startX = event.clientX;
      startY = event.clientY;
      startWidth = panel.offsetWidth;
      startHeight = panel.offsetHeight;

      resizeHandle.setPointerCapture(event.pointerId);
      panel.classList.add("is-resizing");
      event.preventDefault();
    });

    window.addEventListener("pointermove", (event) => {
      if (!isResizing) {
        return;
      }

      const nextWidth = startWidth + (event.clientX - startX);
      const nextHeight = startHeight + (event.clientY - startY);

      const minWidth = 360;
      const minHeight = 420;
      const maxWidth = window.innerWidth - 28;
      const maxHeight = window.innerHeight - 120;

      panel.style.width = `${clamp(nextWidth, minWidth, maxWidth)}px`;
      panel.style.height = `${clamp(nextHeight, minHeight, maxHeight)}px`;
    });

    window.addEventListener("pointerup", () => {
      if (!isResizing) {
        return;
      }
      isResizing = false;
      panel.classList.remove("is-resizing");
    });
  }

  function appendMessage(text, type) {
    const item = document.createElement("div");
    item.className = `chatbot-msg ${type}`;
    item.textContent = text;
    thread.appendChild(item);
    thread.scrollTop = thread.scrollHeight;
  }

  function appendSuggestions(suggestions) {
    if (!Array.isArray(suggestions) || suggestions.length === 0) {
      return;
    }

    const wrap = document.createElement("div");
    wrap.className = "chatbot-suggestions";

    suggestions.slice(0, 3).forEach((text) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chatbot-chip";
      chip.textContent = text;
      chip.addEventListener("click", () => {
        input.value = text;
        form.requestSubmit();
      });
      wrap.appendChild(chip);
    });

    thread.appendChild(wrap);
    thread.scrollTop = thread.scrollHeight;
  }

  async function sendMessage(message) {
    try {
      const response = await fetch("/chatbot/message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Chat request failed.");
      }

      appendMessage(data.reply || "I am not sure about that yet.", "bot");
      appendSuggestions(data.suggestions || []);
    } catch (error) {
      appendMessage(error.message || "Unable to reach chatbot right now.", "bot");
    }
  }

  launcher.addEventListener("click", () => {
    const nextState = !panel.classList.contains("is-open");
    setOpenState(nextState);
  });

  minimizeBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    setOpenState(false);
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) {
      return;
    }

    appendMessage(message, "user");
    input.value = "";
    sendMessage(message);
  });

  enableResize();

  // Open in expanded mode when the page loads.
  setOpenState(true);
})();
