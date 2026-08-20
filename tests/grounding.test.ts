import { describe, expect, it } from 'vitest';
import { validateGrounding } from '../src/services/grounding.js';
const chunk = (text: string) => [{ id: '1', text, source: 'tenant.md', score: 0.9 }];
describe('bounded grounding validation', () => {
  it('accepts an evidence-supported qualified price', () => {
    expect(validateGrounding('What is the price?', 'It starts at £17.99 per month, plus a joining fee.', chunk('Pricing: from £17.99 per month plus joining fee; prices vary by gym.')).ok).toBe(true);
  });
  it('rejects unsupported numeric claims and explicit unknown invention', () => {
    const result = validateGrounding('How much is the day pass?', 'A day pass costs £25.', chunk('Day-pass pricing: UNKNOWN. The current price depends on the selected gym.'));
    expect(result.ok).toBe(false);
    expect(result.reasons).toContain('unsupported numeric claim');
  });
  it('rejects an absolute answer that drops a qualification', () => {
    expect(validateGrounding('Are all gyms open 24/7?', 'All gyms are open 24/7.', chunk('Most gyms are open 24/7, but some have different hours.')).ok).toBe(false);
  });
});
