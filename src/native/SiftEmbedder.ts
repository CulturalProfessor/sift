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

export interface LibraryStats {
  photosIndexed: number;
  photosTotal: number;
  videosIndexed: number;
  videosTotal: number;
  keyframes: number;
}

interface SiftEmbedderNative {
  embedImage(uri: string): Promise<number[]>;
  embedText(tokenIds: number[]): Promise<number[]>;
  searchImages(
    queryEmbedding: number[],
    topK: number,
    token: number,
  ): Promise<SearchHit[]>;
  indexGallery(maxCount: number): Promise<IndexResult>;
  indexVideos(maxCount: number): Promise<VideoIndexResult>;
  indexedCount(): Promise<number>;
  indexedIds(): Promise<string[]>;
  libraryStats(): Promise<LibraryStats>;
  getSettings(): Promise<IndexSettings>;
  setIndexThrottle(ms: number): Promise<void>;
  setIndexScope(
    sinceMs: number,
    maxFiles: number,
    indexVideos: boolean,
  ): Promise<void>;
  setMatchMin(percent: number): Promise<void>;
  setTopK(k: number): Promise<void>;
  clearIndex(): Promise<void>;
  readAsset(name: string): Promise<string>;
  openVideoAt(uri: string, timestampMs: number): Promise<void>;
  openInGallery(uri: string, isVideo: boolean): Promise<void>;
  deleteAsset(uri: string, isVideo: boolean): Promise<boolean>;
}

export interface IndexSettings {
  throttleMs: number;
  deviceDefaultMs: number;
  indexSinceMs: number; // 0 = all time
  indexMaxFiles: number; // 0 = no limit
  indexVideos: boolean;
  matchMinPercent: number;
  topK: number;
}

const { SiftEmbedder } = NativeModules as { SiftEmbedder: SiftEmbedderNative };

const emitter = new NativeEventEmitter(NativeModules.SiftEmbedder);

export function embedImage(uri: string): Promise<number[]> {
  return SiftEmbedder.embedImage(uri);
}

export function embedText(tokenIds: number[]): Promise<number[]> {
  return SiftEmbedder.embedText(tokenIds);
}

// Bumped on every call so a stale in-flight native search (still churning
// through the executor from an earlier keystroke) can detect it's superseded
// and bail out early instead of running to completion for a discarded result.
let searchToken = 0;

export function searchImages(
  queryEmbedding: number[],
  topK: number,
): Promise<SearchHit[]> {
  return SiftEmbedder.searchImages(queryEmbedding, topK, ++searchToken);
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

export function libraryStats(): Promise<LibraryStats> {
  return SiftEmbedder.libraryStats();
}

export function getSettings(): Promise<IndexSettings> {
  return SiftEmbedder.getSettings();
}

export function setIndexThrottle(ms: number): Promise<void> {
  return SiftEmbedder.setIndexThrottle(ms);
}

export function setIndexScope(
  sinceMs: number,
  maxFiles: number,
  indexVideos: boolean,
): Promise<void> {
  return SiftEmbedder.setIndexScope(sinceMs, maxFiles, indexVideos);
}

export function setMatchMin(percent: number): Promise<void> {
  return SiftEmbedder.setMatchMin(percent);
}

export function setTopK(k: number): Promise<void> {
  return SiftEmbedder.setTopK(k);
}

export function clearIndex(): Promise<void> {
  return SiftEmbedder.clearIndex();
}

export function indexedIds(): Promise<string[]> {
  return SiftEmbedder.indexedIds();
}

export function readAsset(name: string): Promise<string> {
  return SiftEmbedder.readAsset(name);
}

export function openVideoAt(uri: string, timestampMs: number): Promise<void> {
  return SiftEmbedder.openVideoAt(uri, timestampMs);
}

export function openInGallery(uri: string, isVideo: boolean): Promise<void> {
  return SiftEmbedder.openInGallery(uri, isVideo);
}

/** Resolves true if deleted, false if the user declined the system confirmation. */
export function deleteAsset(uri: string, isVideo: boolean): Promise<boolean> {
  return SiftEmbedder.deleteAsset(uri, isVideo);
}

export function onIndexProgress(cb: (p: IndexProgress) => void): () => void {
  const sub = emitter.addListener('SiftIndexProgress', cb);
  return () => sub.remove();
}

export function onVideoProgress(cb: (p: IndexProgress) => void): () => void {
  const sub = emitter.addListener('SiftVideoProgress', cb);
  return () => sub.remove();
}
