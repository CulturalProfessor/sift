import { NativeEventEmitter, NativeModules } from 'react-native';

export interface IndexResult {
  added: number;
  removed: number;
  totalIndexed: number;
}

export interface IndexProgress {
  done: number;
  total: number;
}

export interface SearchHit {
  assetId: string;
  uri: string;
  score: number;
  isVideo: boolean;
  timestampMs: number;
}

export interface VideoIndexResult {
  videosIndexed: number;
  videosRemoved: number;
  totalVideos: number;
  totalKeyframes: number;
}

interface SiftEmbedderNative {
  embedImage(uri: string): Promise<number[]>;
  embedText(tokenIds: number[]): Promise<number[]>;
  searchImages(queryEmbedding: number[], topK: number): Promise<SearchHit[]>;
  indexGallery(maxCount: number): Promise<IndexResult>;
  indexVideos(maxCount: number): Promise<VideoIndexResult>;
  indexedCount(): Promise<number>;
  indexedIds(): Promise<string[]>;
  readAsset(name: string): Promise<string>;
}

const { SiftEmbedder } = NativeModules as { SiftEmbedder: SiftEmbedderNative };

const emitter = new NativeEventEmitter(NativeModules.SiftEmbedder);

export function embedImage(uri: string): Promise<number[]> {
  return SiftEmbedder.embedImage(uri);
}

export function embedText(tokenIds: number[]): Promise<number[]> {
  return SiftEmbedder.embedText(tokenIds);
}

export function searchImages(
  queryEmbedding: number[],
  topK: number,
): Promise<SearchHit[]> {
  return SiftEmbedder.searchImages(queryEmbedding, topK);
}

export function indexGallery(maxCount = 0): Promise<IndexResult> {
  return SiftEmbedder.indexGallery(maxCount);
}

export function indexVideos(maxCount = 0): Promise<VideoIndexResult> {
  return SiftEmbedder.indexVideos(maxCount);
}

export function indexedCount(): Promise<number> {
  return SiftEmbedder.indexedCount();
}

export function indexedIds(): Promise<string[]> {
  return SiftEmbedder.indexedIds();
}

export function readAsset(name: string): Promise<string> {
  return SiftEmbedder.readAsset(name);
}

export function onIndexProgress(cb: (p: IndexProgress) => void): () => void {
  const sub = emitter.addListener('SiftIndexProgress', cb);
  return () => sub.remove();
}

export function onVideoProgress(cb: (p: IndexProgress) => void): () => void {
  const sub = emitter.addListener('SiftVideoProgress', cb);
  return () => sub.remove();
}
