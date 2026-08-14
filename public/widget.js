(() => {
  const script = document.currentScript,
    apiKey = script?.dataset.apiKey;
  if (!apiKey) return;
  const base = new URL(script.src).origin;
  let sessionId,
    sessionKey,
    conversationId,
    leadSubmitted = false;
  const esc = (s) =>
    String(s || '').replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  async function sessionStorageKey() {
    const material = new TextEncoder().encode(`orbit-widget-session:${apiKey}`),
      digest = await crypto.subtle.digest('SHA-256', material);
    return `tai_session_${Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, '0'),
    )
      .join('')
      .slice(0, 32)}`;
  }
  async function start() {
    let config;
    try {
      const r = await fetch(base + '/v1/config', {
        headers: { 'x-api-key': apiKey },
      });
      if (!r.ok) return;
      config = await r.json();
      sessionKey = await sessionStorageKey();
      try {
        const storedSession = localStorage.getItem(sessionKey);
        if (/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
          storedSession || '',
        ))
          sessionId = storedSession;
        else if (storedSession) localStorage.removeItem(sessionKey);
      } catch {}
    } catch {
      return;
    }
    const w = config.widget || {},
      theme = config.theme || {},
      color = w.primaryColor || theme.primaryColor || '#6d5dfc',
      dark = w.dark ?? theme.dark ?? false;
    const root = document.createElement('div');
    root.id = 'tenant-ai-widget';
    const icon = { chat: '●', sparkles: '✦', help: '?' }[w.icon] || '●';
    root.innerHTML = `<button class="tai-launch" aria-label="Open ${esc(w.assistantName || config.assistantName || 'assistant')}">${icon}</button><section hidden role="dialog" aria-label="Chat assistant"><header>${w.logoUrl ? `<img src="${esc(w.logoUrl)}" alt="">` : ''}${w.avatarUrl ? `<img src="${esc(w.avatarUrl)}" alt="">` : ''}<div><b>${esc(w.assistantName || config.assistantName || 'Assistant')}</b><small>Online</small></div><button class="tai-close" aria-label="Close assistant">×</button></header><div class="tai-log" aria-live="polite"><p>${esc(w.welcomeMessage || config.welcomeMessage || 'How can I help?')}</p>${(w.suggestedQuestions || []).map((q) => `<button class="tai-q">${esc(q)}</button>`).join('')}</div><form class="tai-chat"><input aria-label="Message" placeholder="Type your message…" maxlength="4000" required><button aria-label="Send">➤</button></form></section>`;
    const side = (w.position || theme.position) === 'bottom-left' ? 'left' : 'right';
    Object.assign(root.style, {
      position: 'fixed',
      [side]: '20px',
      bottom: '20px',
      zIndex: '2147483647',
      fontFamily: 'Inter,system-ui,sans-serif',
      '--tai-color': color,
      '--tai-bg': dark ? '#11141b' : '#fff',
      '--tai-fg': dark ? '#f3f4f6' : '#172033',
      '--tai-radius': `${w.radius ?? theme.radius ?? 18}px`,
    });
    const style = document.createElement('style');
    style.textContent = `#tenant-ai-widget *{box-sizing:border-box}#tenant-ai-widget .tai-launch{width:56px;height:56px;border:0;border-radius:50%;background:var(--tai-color);color:white;font-size:20px;box-shadow:0 12px 30px #0004;cursor:pointer;float:right}#tenant-ai-widget section{position:absolute;bottom:70px;${side}:0;width:min(${w.width || 360}px,calc(100vw - 32px));height:min(${w.height || 520}px,calc(100vh - 110px));border-radius:var(--tai-radius);overflow:hidden;background:var(--tai-bg);color:var(--tai-fg);box-shadow:0 24px 80px #0005;flex-direction:column;border:1px solid #8883}#tenant-ai-widget section:not([hidden]){display:flex}#tenant-ai-widget header{display:flex;align-items:center;gap:10px;padding:15px;background:var(--tai-color);color:#fff}#tenant-ai-widget header img{width:32px;height:32px;border-radius:50%;object-fit:cover}#tenant-ai-widget header div{flex:1}#tenant-ai-widget small{display:block;opacity:.7}#tenant-ai-widget .tai-close{border:0;background:none;color:white;font-size:24px;cursor:pointer}#tenant-ai-widget .tai-log{flex:1;overflow:auto;padding:16px;white-space:pre-wrap;font-size:14px}#tenant-ai-widget .tai-log p{background:#8882;padding:11px;border-radius:12px}#tenant-ai-widget .tai-q{border:1px solid var(--tai-color);color:var(--tai-color);background:transparent;border-radius:20px;padding:7px;margin:3px;cursor:pointer}#tenant-ai-widget .tai-chat{display:flex;padding:10px;border-top:1px solid #8883}#tenant-ai-widget .tai-chat input{flex:1;border:0;background:transparent;color:inherit;padding:10px;outline:none}#tenant-ai-widget form button{border:0;background:var(--tai-color);color:#fff;border-radius:10px;padding:9px 12px}#tenant-ai-widget .tai-lead{display:grid;gap:7px;background:#8882;padding:11px;border-radius:12px;white-space:normal}#tenant-ai-widget .tai-lead input,#tenant-ai-widget .tai-lead textarea{width:100%;border:1px solid #8885;border-radius:7px;background:var(--tai-bg);color:var(--tai-fg);padding:8px;font:inherit}#tenant-ai-widget .tai-status{font-size:12px}`;
    document.head.append(style);
    const section = root.querySelector('section'),
      launch = root.querySelector('.tai-launch'),
      close = root.querySelector('.tai-close'),
      input = root.querySelector('.tai-chat input'),
      log = root.querySelector('.tai-log');
    const addLine = (speaker, text) => {
      const p = document.createElement('p');
      p.textContent = `${speaker}: ${text}`;
      log.append(p);
    };
    const showLeadForm = (fields) => {
      if (leadSubmitted || log.querySelector('.tai-lead')) return;
      const allowed = ['name', 'email', 'phone', 'requirement'],
        selected = [...new Set(['name', ...(Array.isArray(fields) ? fields : [])])].filter((x) =>
          allowed.includes(x),
        ),
        form = document.createElement('form');
      form.className = 'tai-lead';
      form.setAttribute('aria-label', 'Contact details');
      const labels = {
        name: 'Name',
        email: 'Email',
        phone: 'Phone',
        requirement: 'How can we help?',
      };
      selected.forEach((field) => {
        const control = document.createElement(field === 'requirement' ? 'textarea' : 'input');
        control.name = field;
        control.setAttribute('aria-label', labels[field]);
        control.placeholder = labels[field];
        control.maxLength = field === 'requirement' ? 2000 : field === 'phone' ? 40 : 120;
        if (field === 'name') control.required = true;
        if (field === 'email') control.type = 'email';
        form.append(control);
      });
      const button = document.createElement('button');
      button.type = 'submit';
      button.textContent = 'Send details';
      const status = document.createElement('div');
      status.className = 'tai-status';
      status.setAttribute('role', 'status');
      form.append(button, status);
      form.onsubmit = async (event) => {
        event.preventDefault();
        button.disabled = true;
        status.textContent = 'Sending…';
        const data = Object.fromEntries(new FormData(form).entries());
        try {
          const response = await fetch(base + '/v1/leads', {
            method: 'POST',
            headers: { 'content-type': 'application/json', 'x-api-key': apiKey },
            body: JSON.stringify({ ...data, ...(conversationId ? { conversationId } : {}) }),
          });
          if (!response.ok) throw Error();
          leadSubmitted = true;
          form
            .querySelectorAll('input,textarea,button')
            .forEach((control) => (control.disabled = true));
          status.textContent = 'Thanks — your details were sent.';
        } catch {
          button.disabled = false;
          status.textContent = 'Unable to send. Please check your details and try again.';
        }
      };
      log.append(form);
    };
    launch.onclick = () => (section.hidden = !section.hidden);
    close.onclick = () => (section.hidden = true);
    root.querySelectorAll('.tai-q').forEach(
      (b) =>
        (b.onclick = () => {
          input.value = b.textContent;
          input.focus();
        }),
    );
    root.querySelector('.tai-chat').onsubmit = async (e) => {
      e.preventDefault();
      const text = input.value;
      input.value = '';
      addLine('You', text);
      try {
        const r = await fetch(base + '/v1/chat', {
            method: 'POST',
            headers: { 'content-type': 'application/json', 'x-api-key': apiKey },
            body: JSON.stringify({ message: text, ...(sessionId ? { sessionId } : {}) }),
          }),
          j = await r.json();
        if (!r.ok) throw Error();
        sessionId = j.sessionId;
        conversationId = j.conversationId;
        try {
          localStorage.setItem(sessionKey, sessionId);
        } catch {}
        addLine('Assistant', j.answer);
        if (j.leadCollection?.enabled) showLeadForm(j.leadCollection.fields);
      } catch {
        addLine('Assistant', 'Unable to connect.');
      }
      log.scrollTop = log.scrollHeight;
    };
    document.body.append(root);
  }
  start();
})();
