/* DocChat: загрузка документов, чат со стримингом ответа (SSE), ссылки на источники.
   Весь текст из сети и документов вставляется через textContent — innerHTML не используется. */
(() => {
  'use strict';

  const I18N = {
    ru: {
      'drop.title': 'Перетащите документ сюда',
      'drop.hint': 'или нажмите, чтобы выбрать · PDF, TXT, MD',
      'docs.title': 'Документы',
      'docs.empty': 'Пока ничего не загружено.',
      'composer.placeholder': 'Задайте вопрос по документу…',
      'composer.send': 'Спросить',
      'empty.title': 'Спросите свои документы',
      'empty.text': 'Загрузите PDF слева, затем задавайте вопросы. Ответы приходят со ссылками на страницы.',
      'empty.ready': 'Документ готов. Спросите о чём угодно из него.',
      'upload.progress': 'Загружаю и читаю: ',
      'upload.done': 'Готово: ',
      'upload.failed': 'Не загрузилось: ',
      'upload.network': 'нет связи с сервером',
      'doc.pages': 'стр.',
      'doc.delete': 'Удалить документ',
      'sources.title': 'Источники',
      'source.page': 'стр.',
      'mode.claude': 'Claude · ',
      'mode.retrieval': 'поиск без LLM',
      'note.claude': 'Фрагменты документов отправляются в Anthropic API. Документы удаляются через {h} ч.',
      'note.retrieval': 'Режим без LLM: показываю найденные фрагменты. Документы удаляются через {h} ч.',
      'ask.none': 'Отметьте хотя бы один документ.',
      'ask.error': 'Не удалось получить ответ.',
      'suggest.1': 'Кратко перескажи документ',
      'suggest.2': 'Какие сроки и суммы упомянуты?',
      'suggest.3': 'Каковы основные требования?',
    },
    en: {
      'drop.title': 'Drop a document here',
      'drop.hint': 'or click to choose · PDF, TXT, MD',
      'docs.title': 'Documents',
      'docs.empty': 'Nothing uploaded yet.',
      'composer.placeholder': 'Ask a question about the document…',
      'composer.send': 'Ask',
      'empty.title': 'Ask your documents',
      'empty.text': 'Upload a PDF on the left, then ask questions. Answers come with page references.',
      'empty.ready': 'Document is ready. Ask anything about it.',
      'upload.progress': 'Uploading and reading: ',
      'upload.done': 'Done: ',
      'upload.failed': 'Upload failed: ',
      'upload.network': 'cannot reach the server',
      'doc.pages': 'pp.',
      'doc.delete': 'Delete document',
      'sources.title': 'Sources',
      'source.page': 'p.',
      'mode.claude': 'Claude · ',
      'mode.retrieval': 'retrieval-only',
      'note.claude': 'Document passages are sent to the Anthropic API. Documents are deleted after {h} h.',
      'note.retrieval': 'Retrieval-only mode: showing matched passages. Documents are deleted after {h} h.',
      'ask.none': 'Select at least one document.',
      'ask.error': 'Could not get an answer.',
      'suggest.1': 'Summarize the document briefly',
      'suggest.2': 'Which deadlines and amounts are mentioned?',
      'suggest.3': 'What are the main requirements?',
    },
  };

  const STORAGE_KEY = 'docchat.lang';
  const HISTORY_LIMIT = 6;

  const els = {
    drop: document.getElementById('drop'),
    file: document.getElementById('file'),
    status: document.getElementById('upload-status'),
    docs: document.getElementById('docs'),
    docsEmpty: document.getElementById('docs-empty'),
    note: document.getElementById('note'),
    messages: document.getElementById('messages'),
    suggestions: document.getElementById('suggestions'),
    form: document.getElementById('form'),
    question: document.getElementById('question'),
    send: document.getElementById('send'),
    badge: document.getElementById('mode-badge'),
    lang: document.getElementById('lang'),
  };

  const state = {
    lang: 'ru',
    config: null,
    docs: [],
    selected: new Set(),
    history: [],
    busy: false,
    lastStatus: null,
  };

  const t = (key) => I18N[state.lang][key] ?? I18N.ru[key] ?? key;

  function h(tag, props = {}, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? '' : String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  /* ---------- язык ---------- */

  function detectLang() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && I18N[stored]) return stored;
    } catch (_) { /* приватный режим */ }
    return (navigator.language || '').toLowerCase().startsWith('ru') ? 'ru' : 'en';
  }

  function applyLang() {
    document.documentElement.lang = state.lang;
    document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
      el.setAttribute('placeholder', t(el.dataset.i18nPlaceholder));
    });
    els.lang.textContent = state.lang === 'ru' ? 'EN' : 'RU';
    renderMode();
    renderDocs();
    renderSuggestions();
    renderEmpty();
  }

  els.lang.addEventListener('click', () => {
    state.lang = state.lang === 'ru' ? 'en' : 'ru';
    try { localStorage.setItem(STORAGE_KEY, state.lang); } catch (_) { /* не критично */ }
    applyLang();
  });

  /* ---------- API ---------- */

  async function api(path, options) {
    const response = await fetch(path, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.message || `HTTP ${response.status}`);
    return body;
  }

  /* ---------- режим и заметка ---------- */

  function renderMode() {
    if (!state.config) return;
    const claude = state.config.mode === 'claude';
    els.badge.hidden = false;
    els.badge.className = `badge ${claude ? 'claude' : 'retrieval'}`;
    els.badge.textContent = claude ? `${t('mode.claude')}${state.config.model}` : t('mode.retrieval');
    els.note.textContent = t(claude ? 'note.claude' : 'note.retrieval').replace('{h}', state.config.ttl_hours);
  }

  /* ---------- документы ---------- */

  function setStatus(text, isError = false) {
    els.status.textContent = text;
    els.status.className = `status${isError ? ' error' : ''}`;
  }

  function renderDocs() {
    els.docs.replaceChildren(...state.docs.map((doc) => h(
      'li', { class: 'doc' },
      h('input', {
        type: 'checkbox', checked: state.selected.has(doc.id), 'aria-label': doc.name,
        onchange: (event) => {
          if (event.target.checked) state.selected.add(doc.id); else state.selected.delete(doc.id);
        },
      }),
      h('div', { class: 'doc-body' },
        h('div', { class: 'doc-name', title: doc.name }, doc.name),
        h('div', { class: 'doc-meta' }, `${doc.pages} ${t('doc.pages')} · ${doc.chunks}`)),
      h('button', {
        class: 'icon-btn', type: 'button', title: t('doc.delete'), 'aria-label': t('doc.delete'),
        onclick: () => removeDoc(doc.id),
      }, '×'))));
    els.docs.querySelectorAll('input[type="checkbox"]').forEach((box, index) => {
      box.checked = state.selected.has(state.docs[index].id);
    });
    els.docsEmpty.hidden = state.docs.length > 0;
  }

  async function refreshDocs() {
    const { documents } = await api('/api/docs');
    const known = new Set(state.docs.map((d) => d.id));
    state.docs = documents;
    state.selected = new Set(documents.filter((d) => state.selected.has(d.id) || !known.has(d.id)).map((d) => d.id));
    renderDocs();
    renderSuggestions();
    renderEmpty();
  }

  async function uploadFiles(files) {
    for (const file of files) {
      setStatus(`${t('upload.progress')}${file.name}`);
      const form = new FormData();
      form.append('file', file);
      try {
        const doc = await api('/api/docs', { method: 'POST', body: form });
        setStatus(`${t('upload.done')}${doc.name}`);
      } catch (error) {
        const reason = error instanceof TypeError ? t('upload.network') : error.message;
        setStatus(`${t('upload.failed')}${file.name} — ${reason}`, true);
        return refreshDocs();
      }
    }
    await refreshDocs();
  }

  async function removeDoc(id) {
    try { await api(`/api/docs/${id}`, { method: 'DELETE' }); } catch (_) { /* уже удалён */ }
    state.selected.delete(id);
    await refreshDocs();
  }

  els.drop.addEventListener('click', () => els.file.click());
  els.drop.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); els.file.click(); }
  });
  els.file.addEventListener('change', () => {
    if (els.file.files.length) uploadFiles([...els.file.files]);
    els.file.value = '';
  });
  ['dragenter', 'dragover'].forEach((name) => els.drop.addEventListener(name, (event) => {
    event.preventDefault();
    els.drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((name) => els.drop.addEventListener(name, (event) => {
    event.preventDefault();
    els.drop.classList.remove('over');
  }));
  els.drop.addEventListener('drop', (event) => {
    if (event.dataTransfer?.files?.length) uploadFiles([...event.dataTransfer.files]);
  });

  /* ---------- вывод ответа ---------- */

  const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|_\([^)]+\)_|\[\d{1,3}\])/g;

  function renderInline(target, text, sources, onCite) {
    for (const part of text.split(INLINE)) {
      if (!part) continue;
      const cite = /^\[(\d{1,3})\]$/.exec(part);
      if (cite && sources.some((s) => s.n === Number(cite[1]))) {
        target.append(h('button', {
          class: 'cite', type: 'button', onclick: () => onCite(Number(cite[1])),
        }, cite[1]));
      } else if (/^\*\*.+\*\*$/.test(part)) {
        target.append(h('strong', {}, part.slice(2, -2)));
      } else if (/^`.+`$/.test(part)) {
        target.append(h('code', {}, part.slice(1, -1)));
      } else if (/^_\(.+\)_$/.test(part)) {
        target.append(h('em', {}, part.slice(1, -1)));
      } else {
        target.append(part);
      }
    }
  }

  function renderRich(container, text, sources, onCite) {
    const blocks = [];
    for (const block of text.split(/\n{2,}/)) {
      const lines = block.split('\n').filter((line) => line.trim());
      if (!lines.length) continue;
      const bullets = lines.every((line) => /^\s*(?:[-*•]|\d+[.)])\s+/.test(line));
      if (bullets) {
        const list = h('ul');
        for (const line of lines) {
          const item = h('li');
          renderInline(item, line.replace(/^\s*(?:[-*•]|\d+[.)])\s+/, ''), sources, onCite);
          list.append(item);
        }
        blocks.push(list);
      } else {
        const paragraph = h('p');
        lines.forEach((line, index) => {
          if (index) paragraph.append(h('br'));
          renderInline(paragraph, line, sources, onCite);
        });
        blocks.push(paragraph);
      }
    }
    container.replaceChildren(...blocks);
  }

  function renderSources(sources) {
    if (!sources.length) return null;
    const cards = sources.map((s) => h(
      'div', { class: 'source', id: `src-${s.uid}` },
      h('div', { class: 'source-head' }, `[${s.n}] ${s.doc_name} · ${t('source.page')} ${s.page}`),
      h('div', { class: 'source-text' }, s.text)));
    return { node: h('details', { class: 'sources' }, h('summary', {}, `${t('sources.title')} (${sources.length})`),
      h('div', { class: 'source-list' }, cards)), cards };
  }

  /* ---------- сообщения ---------- */

  function renderEmpty() {
    els.messages.querySelector('.empty')?.remove();
    if (els.messages.querySelector('.msg')) return;
    const ready = state.docs.length > 0;
    els.messages.append(h('div', { class: 'empty' },
      h('h1', {}, t('empty.title')),
      h('p', {}, t(ready ? 'empty.ready' : 'empty.text'))));
  }

  function renderSuggestions() {
    const show = state.docs.length > 0 && state.history.length === 0;
    els.suggestions.replaceChildren(...(show ? ['suggest.1', 'suggest.2', 'suggest.3'].map((key) => h(
      'button', { class: 'suggestion', type: 'button', onclick: () => ask(t(key)) }, t(key))) : []));
  }

  function scrollDown() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  function addMessage(role, className = '') {
    els.messages.querySelector('.empty')?.remove();
    const bubble = h('div', { class: 'bubble' });
    const wrapper = h('div', { class: `msg ${role} ${className}`.trim() }, bubble);
    els.messages.append(wrapper);
    scrollDown();
    return { wrapper, bubble };
  }

  function parseEvent(block) {
    let name = 'message';
    let data = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).trim();
    }
    return data ? { name, data: JSON.parse(data) } : null;
  }

  async function ask(question) {
    question = question.trim();
    if (!question || state.busy) return;
    const ids = [...state.selected];
    if (!ids.length) { setStatus(t('ask.none'), true); return; }

    state.busy = true;
    els.send.disabled = true;
    els.question.value = '';
    autoGrow();

    addMessage('user').bubble.textContent = question;
    const answer = addMessage('assistant');
    answer.bubble.classList.add('typing');
    state.history.push({ role: 'user', content: question });
    renderSuggestions();

    const body = {
      question,
      doc_ids: ids,
      history: state.history.slice(0, -1).slice(-HISTORY_LIMIT),
    };
    let sources = [];
    let text = '';
    let sourcesBox = null;
    let failed = false;

    const onCite = (n) => {
      const card = sourcesBox?.cards[n - 1];
      if (!card) return;
      sourcesBox.node.open = true;
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.classList.add('flash');
      setTimeout(() => card.classList.remove('flash'), 1600);
    };
    const paint = () => renderRich(answer.bubble, text, sources, onCite);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      if (!response.ok) {
        const info = await response.json().catch(() => ({}));
        throw new Error(info.message || `HTTP ${response.status}`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop();
        for (const block of blocks) {
          const event = parseEvent(block);
          if (!event) continue;
          if (event.name === 'sources') {
            sources = event.data.sources.map((s, i) => ({ ...s, uid: `${Date.now()}-${i}` }));
            sourcesBox = renderSources(sources);
          } else if (event.name === 'delta') {
            text += event.data.text;
            paint();
            scrollDown();
          } else if (event.name === 'error') {
            failed = true;
            answer.wrapper.classList.add('error');
            text += (text ? '\n\n' : '') + event.data.message;
            paint();
          }
        }
      }
    } catch (error) {
      failed = true;
      answer.wrapper.classList.add('error');
      text = error instanceof TypeError ? t('upload.network') : (error.message || t('ask.error'));
      paint();
    } finally {
      answer.bubble.classList.remove('typing');
      paint();
      if (sourcesBox) answer.wrapper.append(sourcesBox.node);
      if (failed) state.history.pop();
      else state.history.push({ role: 'assistant', content: text });
      state.busy = false;
      els.send.disabled = false;
      scrollDown();
      els.question.focus();
    }
  }

  /* ---------- ввод ---------- */

  function autoGrow() {
    els.question.style.height = 'auto';
    els.question.style.height = `${Math.min(els.question.scrollHeight, 160)}px`;
  }

  els.question.addEventListener('input', autoGrow);
  els.question.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.form.addEventListener('submit', (event) => {
    event.preventDefault();
    ask(els.question.value);
  });

  /* ---------- старт ---------- */

  async function init() {
    state.lang = detectLang();
    try {
      state.config = await api('/api/config');
      await refreshDocs();
    } catch (_) {
      setStatus(t('upload.network'), true);
    }
    applyLang();
  }

  init();
})();
