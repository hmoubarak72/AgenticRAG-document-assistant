/* AgenticRAG-document-assistant frontend — vanilla JS. Talks to the FastAPI backend served on the same origin. */
(() => {
  const API = window.AGENTICRAG_API_BASE || "";
  const $ = (id) => document.getElementById(id);
  const els = {
    thread: $("thread"), hero: $("hero"), scroll: $("scroll"), input: $("input"), composer: $("composer"),
    send: $("sendBtn"), fileInput: $("fileInput"), fileInputInline: $("fileInputInline"), dropzone: $("dropzone"),
    fileList: $("fileList"), docCount: $("docCount"), fileCountLabel: $("fileCountLabel"), chips: $("formatChips"),
    uploadStatus: $("uploadStatus"), statusDot: $("statusDot"), modeLabel: $("modeLabel"), recent: $("recentList"),
    newChat: $("newChat"), scopePill: $("scopePill"), dzLimit: $("dzLimit"),
  };

  const NODE_LABELS = {
    planner: "Understanding intent", retrieve: "Searching Qdrant", grade: "Checking evidence",
    rewrite: "Refining query", generate: "Generating grounded answer", direct_answer: "Answering directly",
    clarify: "Asking for clarification", respond: "Formatting response",
  };
  const STRATEGY_BADGE = {
    DIRECT_ANSWER: ["direct", "Direct answer"], RAG_RETRIEVAL: ["rag", "Document retrieval"],
    CLARIFICATION: ["clar", "Clarification"],
  };

  let documents = [];
  let selected = new Set();
  let busy = false;
  let conversations = load("agenticrag.conversations", []);
  let current = newConversation();

  // ------------------------------------------------------------ storage (best effort)
  function load(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  }
  function save() {
    try { localStorage.setItem("agenticrag.conversations", JSON.stringify(conversations.slice(0, 30))); } catch { /* ignore */ }
  }

  // ------------------------------------------------------------ utils
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function renderMarkdown(md) {
    let html;
    if (window.marked) {
      html = window.marked.parse(md, { breaks: true, gfm: true });
      if (window.DOMPurify) html = window.DOMPurify.sanitize(html);
    } else {
      html = escapeHtml(md).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
    }
    // [1] -> citation chip
    return html.replace(/\[(\d{1,2})\](?![^<]*<\/code>)/g, '<span class="cite" data-cite="$1">$1</span>');
  }
  const fmtBytes = (b) => (!b ? "" : b < 1024 * 1024 ? `${Math.max(1, Math.round(b / 1024))} KB` : `${(b / 1048576).toFixed(1)} MB`);
  const scrollDown = () => { els.scroll.scrollTop = els.scroll.scrollHeight; };

  // ------------------------------------------------------------ health & library
  async function checkHealth() {
    try {
      const h = await (await fetch(`${API}/health`)).json();
      els.statusDot.className = "status-dot ok";
      els.statusDot.title = `Connected · LLM: ${h.llm}`;
      els.modeLabel.textContent = h.llm === "offline" ? "Offline mode · add GROQ_API_KEY" : `Private session · ${h.llm}`;
      els.dzLimit.textContent = `Up to ${h.max_upload_mb} MB per file`;
      els.chips.innerHTML = h.supported_formats.filter((f) => ![".htm", ".markdown"].includes(f))
        .map((f) => `<span class="chip">${f.slice(1)}</span>`).join("");
      els.fileInput.accept = els.fileInputInline.accept = h.supported_formats.join(",");
    } catch {
      els.statusDot.className = "status-dot err";
      els.modeLabel.textContent = "Backend unreachable";
    }
  }

  async function refreshDocuments() {
    try {
      documents = (await (await fetch(`${API}/documents`)).json()).documents || [];
    } catch { documents = []; }
    selected = new Set([...selected].filter((id) => documents.some((d) => d.doc_id === id)));
    renderLibrary();
  }

  function renderLibrary() {
    els.docCount.textContent = documents.length;
    els.fileCountLabel.textContent = `${documents.length} file${documents.length === 1 ? "" : "s"}`;
    els.fileList.innerHTML = documents.map((d) => {
      const ext = (d.file_type || d.document.split(".").pop() || "").toLowerCase();
      return `<li class="file ${selected.has(d.doc_id) ? "selected" : ""}" data-id="${d.doc_id}">
        <div class="file-type t-${ext}">${escapeHtml(ext.slice(0, 4))}</div>
        <div class="file-meta"><div class="file-name" title="${escapeHtml(d.document)}">${escapeHtml(d.document)}</div>
          <div class="file-sub">${selected.has(d.doc_id) ? "In scope · " : ""}Processed and ready · ${d.chunks} chunks ${fmtBytes(d.size_bytes) ? "· " + fmtBytes(d.size_bytes) : ""}</div></div>
        <button class="file-x" title="Remove" data-del="${d.doc_id}">×</button></li>`;
    }).join("");
    els.scopePill.hidden = selected.size === 0;
    els.scopePill.textContent = `Scoped to ${selected.size} file${selected.size === 1 ? "" : "s"}`;
  }

  els.fileList.addEventListener("click", async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) {
      e.stopPropagation();
      await fetch(`${API}/documents/${del.dataset.del}`, { method: "DELETE" });
      selected.delete(del.dataset.del);
      return refreshDocuments();
    }
    const li = e.target.closest(".file");
    if (!li) return;
    selected.has(li.dataset.id) ? selected.delete(li.dataset.id) : selected.add(li.dataset.id);
    renderLibrary();
  });

  // ------------------------------------------------------------ uploads
  async function uploadFiles(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      els.uploadStatus.hidden = false;
      els.uploadStatus.innerHTML = `<span class="spinner"></span> Processing ${escapeHtml(f.name)} (${i + 1}/${files.length})…`;
      const form = new FormData();
      form.append("files", f);
      try {
        const res = await fetch(`${API}/upload`, { method: "POST", body: form });
        const body = await res.json();
        const r = (body.results || body.detail || [])[0] || {};
        if (r.status === "error") throw new Error(r.error);
      } catch (err) {
        els.uploadStatus.innerHTML = `⚠ ${escapeHtml(f.name)}: ${escapeHtml(err.message || err)}`;
        await refreshDocuments();
        return;
      }
      await refreshDocuments();
    }
    els.uploadStatus.innerHTML = `✓ ${files.length} file${files.length > 1 ? "s" : ""} processed and ready`;
    setTimeout(() => { els.uploadStatus.hidden = true; }, 3500);
  }

  [els.fileInput, els.fileInputInline].forEach((inp) => inp.addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; }));
  ["dragenter", "dragover"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }));
  els.dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => { e.preventDefault(); if (!els.dropzone.contains(e.target)) uploadFiles(e.dataTransfer.files); });

  // ------------------------------------------------------------ conversations
  function newConversation() {
    return { id: Date.now().toString(36), title: "", messages: [] };
  }
  function renderRecent() {
    if (!conversations.length) { els.recent.innerHTML = '<li class="muted small">No conversations yet</li>'; return; }
    els.recent.innerHTML = conversations.map((c) => `<li data-cid="${c.id}" class="${c.id === current.id ? "active" : ""}">${escapeHtml(c.title || "Untitled")}</li>`).join("");
  }
  els.recent.addEventListener("click", (e) => {
    const li = e.target.closest("[data-cid]");
    if (!li || busy) return;
    const conv = conversations.find((c) => c.id === li.dataset.cid);
    if (conv) openConversation(conv);
  });
  function openConversation(conv) {
    current = conv;
    els.thread.innerHTML = "";
    els.hero.hidden = conv.messages.length > 0;
    conv.messages.forEach((m) => (m.role === "user" ? addUserBubble(m.content) : renderBotMessage(addBotShell(), m.result)));
    renderRecent();
    scrollDown();
  }
  els.newChat.addEventListener("click", () => { if (!busy) openConversation(newConversation()); els.input.focus(); });

  // ------------------------------------------------------------ chat rendering
  function addUserBubble(text) {
    const div = document.createElement("div");
    div.className = "msg-user";
    div.textContent = text;
    els.thread.appendChild(div);
  }
  function addBotShell() {
    const wrap = document.createElement("div");
    wrap.className = "msg-bot";
    wrap.innerHTML = `<div class="bot-head"><div class="bot-avatar">A</div><span class="badges"></span><span class="latency"></span></div>
      <details class="trace" open><summary><span class="spinner"></span><span class="trace-title">Agent is thinking…</span></summary><ol></ol></details>
      <div class="answer" hidden></div><div class="sources" hidden></div>`;
    els.thread.appendChild(wrap);
    return wrap;
  }
  function addTraceStep(shell, step) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="node-tag">${escapeHtml(step.node)}</span><span class="detail">${escapeHtml(step.detail)}</span>`;
    shell.querySelector(".trace ol").appendChild(li);
    shell.querySelector(".trace-title").textContent = `${NODE_LABELS[step.node] || step.node}…`;
    scrollDown();
  }
  function renderBotMessage(shell, r) {
    const trace = shell.querySelector(".trace");
    const ol = trace.querySelector("ol");
    if (!ol.children.length) (r.trace || []).forEach((s) => addTraceStep(shell, s));
    trace.querySelector(".spinner")?.remove();
    trace.querySelector(".trace-title").textContent = `Agent trace · ${(r.trace || []).length} steps`;
    trace.open = false;

    const [cls, label] = STRATEGY_BADGE[r.strategy] || ["", r.strategy || "?"];
    let badges = `<span class="badge ${cls}">${label}</span>`;
    if (r.confidence && r.confidence !== "n/a") badges += `<span class="badge ${r.confidence}">Confidence: ${r.confidence}</span>`;
    shell.querySelector(".badges").innerHTML = badges;
    shell.querySelector(".latency").textContent = r.latency_ms ? `${(r.latency_ms / 1000).toFixed(1)}s` : "";

    const ans = shell.querySelector(".answer");
    ans.hidden = false;
    ans.innerHTML = renderMarkdown(r.answer || "");

    const src = shell.querySelector(".sources");
    if (r.sources && r.sources.length) {
      src.hidden = false;
      src.innerHTML = `<div class="sources-title">Sources</div>` + r.sources.map((s) => `
        <div class="source" data-idx="${s.index}"><div class="source-head"><span class="source-idx">[${s.index}]</span>
          <span class="source-name">${escapeHtml(s.document)}${s.page ? ` · p.${s.page}` : ""}</span>
          <span class="source-score">score ${Number(s.score).toFixed(2)}</span></div>
          <div class="source-snippet">${escapeHtml(s.snippet)}…</div></div>`).join("");
    }
  }
  els.thread.addEventListener("click", (e) => {
    const src = e.target.closest(".source");
    if (src) return src.classList.toggle("open");
    const cite = e.target.closest(".cite");
    if (cite) {
      const target = cite.closest(".msg-bot").querySelector(`.source[data-idx="${cite.dataset.cite}"]`);
      if (target) {
        target.classList.add("open", "flash");
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
        setTimeout(() => target.classList.remove("flash"), 1200);
      }
    }
  });

  // ------------------------------------------------------------ sending
  async function ask(question) {
    if (busy || !question.trim()) return;
    busy = true;
    els.send.disabled = true;
    els.hero.hidden = true;
    const history = current.messages.map((m) => ({ role: m.role, content: m.role === "user" ? m.content : m.result.answer }));
    current.messages.push({ role: "user", content: question });
    if (!current.title) {
      current.title = question.slice(0, 60);
      conversations = [current, ...conversations.filter((c) => c.id !== current.id)];
      renderRecent();
    }
    addUserBubble(question);
    const shell = addBotShell();
    scrollDown();

    let result = null;
    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history, doc_ids: selected.size ? [...selected] : null }),
      });
      if (!res.ok || !res.body) throw new Error(`Server returned ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const ev = (block.match(/^event: (.*)$/m) || [])[1];
          const data = (block.match(/^data: (.*)$/m) || [])[1];
          if (!data) continue;
          const payload = JSON.parse(data);
          if (ev === "step") addTraceStep(shell, payload);
          else if (ev === "result") result = payload;
          else if (ev === "error") throw new Error(payload.error);
        }
      }
      if (!result) throw new Error("No result received");
      renderBotMessage(shell, result);
      current.messages.push({ role: "assistant", result });
      save();
    } catch (err) {
      shell.querySelector(".trace .spinner")?.remove();
      const ans = shell.querySelector(".answer");
      ans.hidden = false;
      ans.classList.add("error");
      ans.textContent = `Something went wrong: ${err.message || err}. Check the backend logs and your GROQ_API_KEY.`;
      current.messages.pop();
    } finally {
      busy = false;
      els.send.disabled = false;
      scrollDown();
    }
  }

  els.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = els.input.value;
    els.input.value = "";
    autosize();
    ask(q);
  });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); els.composer.requestSubmit(); }
  });
  function autosize() { els.input.style.height = "auto"; els.input.style.height = `${Math.min(els.input.scrollHeight, 200)}px`; }
  els.input.addEventListener("input", autosize);
  document.querySelectorAll(".sugg").forEach((b) => b.addEventListener("click", () => ask(b.dataset.prompt)));

  // ------------------------------------------------------------ boot
  renderRecent();
  checkHealth();
  refreshDocuments();
  setInterval(checkHealth, 30000);
  els.input.focus();
})();
