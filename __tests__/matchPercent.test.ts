import { matchPercent, SCORE_CEILING, SCORE_FLOOR } from '../src/native/search';

describe('matchPercent', () => {
  it('maps the floor score to the minimum displayed percent', () => {
    expect(matchPercent(SCORE_FLOOR)).toBe(1);
  });

  it('maps the ceiling score to the maximum displayed percent', () => {
    expect(matchPercent(SCORE_CEILING)).toBe(99);
  });

  it('maps the midpoint score to roughly 50%', () => {
    const mid = (SCORE_FLOOR + SCORE_CEILING) / 2;
    expect(matchPercent(mid)).toBe(50);
  });

  it('clamps scores below the floor to 1%, never 0 or negative', () => {
    expect(matchPercent(0)).toBe(1);
    expect(matchPercent(-1)).toBe(1);
  });

  it('clamps scores above the ceiling to 99%, never 100', () => {
    expect(matchPercent(1)).toBe(99);
  });
});
