const mockTokenize = jest.fn((text: string) => [...text].map(c => c.charCodeAt(0)));

jest.mock('../src/native/SiftEmbedder', () => ({
  embedText: jest.fn(),
  readAsset: jest.fn().mockResolvedValue('mock merges'),
  searchImages: jest.fn().mockResolvedValue([]),
}));

jest.mock('../src/native/tokenizer', () => ({
  CLIPTokenizer: jest.fn().mockImplementation(() => ({ tokenize: mockTokenize })),
}));

import { embedText, searchImages } from '../src/native/SiftEmbedder';
import { searchByText } from '../src/native/search';

describe('searchByText', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('embeds the query through both standard CLIP prompt templates', async () => {
    (embedText as jest.Mock).mockResolvedValue([1, 0]);

    await searchByText('dog', 5);

    expect(mockTokenize).toHaveBeenCalledWith('a photo of dog');
    expect(mockTokenize).toHaveBeenCalledWith('a photo containing dog');
    expect(embedText).toHaveBeenCalledTimes(2);
  });

  it('averages the two template embeddings and re-normalizes to unit length', async () => {
    (embedText as jest.Mock)
      .mockResolvedValueOnce([1, 0])
      .mockResolvedValueOnce([0, 1]);

    await searchByText('cat', 5);

    const [queryEmbedding] = (searchImages as jest.Mock).mock.calls[0];
    // average of [1,0] and [0,1] is [0.5, 0.5]; normalized is [~0.707, ~0.707].
    expect(queryEmbedding[0]).toBeCloseTo(Math.SQRT1_2, 4);
    expect(queryEmbedding[1]).toBeCloseTo(Math.SQRT1_2, 4);
    // Unit length after normalization.
    const norm = Math.sqrt(queryEmbedding[0] ** 2 + queryEmbedding[1] ** 2);
    expect(norm).toBeCloseTo(1, 4);
  });

  it('passes topK through to the native search unchanged', async () => {
    (embedText as jest.Mock).mockResolvedValue([1, 0]);

    await searchByText('sunset', 17);

    const [, topK] = (searchImages as jest.Mock).mock.calls[0];
    expect(topK).toBe(17);
  });
});
