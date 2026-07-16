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

/**
 * Full text→image search: tokenize the query (JS), run the text encoder
 * (native), then brute-force cosine over the stored image embeddings (native).
 */
export async function searchByText(
  query: string,
  topK: number,
): Promise<SearchHit[]> {
  const tokenizer = await getTokenizer();
  const tokens = tokenizer.tokenize(query);
  const queryEmbedding = await embedText(tokens);
  return searchImages(queryEmbedding, topK);
}
