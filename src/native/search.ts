import { embedText, readAsset, searchImages, type SearchHit } from './SiftEmbedder';
import { CLIPTokenizer } from './tokenizer';

let tokenizerPromise: Promise<CLIPTokenizer> | null = null;

/** Build the tokenizer once from the bundled BPE merges asset. */
function getTokenizer(): Promise<CLIPTokenizer> {
  if (!tokenizerPromise) {
    tokenizerPromise = readAsset('bpe_merges.txt').then(
      merges => new CLIPTokenizer(merges),
    );
  }
  return tokenizerPromise;
}

// Standard CLIP prompt templates — wrapping the raw query measurably improves
// retrieval (see tools/model-conversion/verify_search.py). Capped at two so
// query latency (~50-150ms/template) stays under the 300ms search budget.
const PROMPT_TEMPLATES = ['a photo of {}', 'a photo containing {}'];

function l2Normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  return norm === 0 ? v : v.map(x => x / norm);
}

/**
 * Full text→image search: tokenize the query (JS), run the text encoder
 * (native) once per prompt template, average + re-normalize the resulting
 * embeddings, then brute-force cosine over the stored image embeddings
 * (native).
 */
export async function searchByText(
  query: string,
  topK: number,
): Promise<SearchHit[]> {
  const tokenizer = await getTokenizer();
  const embeddings = await Promise.all(
    PROMPT_TEMPLATES.map(template => {
      const tokens = tokenizer.tokenize(template.replace('{}', query));
      return embedText(tokens);
    }),
  );

  const dims = embeddings[0].length;
  const averaged = new Array(dims).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < dims; i++) averaged[i] += emb[i] / embeddings.length;
  }
  const queryEmbedding = l2Normalize(averaged);

  return searchImages(queryEmbedding, topK);
}

// CLIP cosine scores sit ~0.15-0.32 for good matches; map to a friendlier
// "match %" so the number reads meaningfully to a user. Prompt-templated
// queries shift this distribution vs. raw-query scores, so these need
// re-tuning against real on-device queries.
export const SCORE_FLOOR = 0.1; // score mapped to 0%
export const SCORE_CEILING = 0.3; // score mapped to 100%

export function matchPercent(score: number): number {
  return Math.max(
    1,
    Math.min(
      99,
      Math.round(((score - SCORE_FLOOR) / (SCORE_CEILING - SCORE_FLOOR)) * 100),
    ),
  );
}
