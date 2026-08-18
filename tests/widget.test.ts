import { describe, expect, it } from 'vitest';
import { readFile } from 'node:fs/promises';

const widget = await readFile(new URL('../public/widget.js', import.meta.url), 'utf8');

describe('customer widget streaming UX', () => {
  it('uses the existing SSE protocol and renders token chunks progressively', () => {
    expect(widget).toContain("/v1/chat/stream");
    expect(widget).toContain("eventName==='token'");
    expect(widget).toContain("eventName==='done'");
    expect(widget).toContain("eventName==='error'");
    expect(widget).toContain("answer+=data.token");
    expect(widget).toContain("getReader()");
  });

  it('has delayed processing state, retryable errors, and duplicate-send protection', () => {
    expect(widget).toContain("setTimeout(()=>{processing=showProcessing();},250)");
    expect(widget).toContain("if (busy) return");
    expect(widget).toContain('tai-retry');
    expect(widget).toContain('Sorry, I couldn’t complete that response.');
  });

  it('keeps tenant identity dynamic and applies scoped glass/mobile styling', () => {
    expect(widget).toContain('w.assistantName || config.assistantName');
    expect(widget).toContain('backdrop-filter:blur');
    expect(widget).toContain('@media(max-width:480px)');
    expect(widget).not.toMatch(/PureGym|ChatGPT|OpenAI|Gemini|Claude/);
  });
});
