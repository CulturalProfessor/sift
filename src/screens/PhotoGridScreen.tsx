import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  Animated,
  Dimensions,
  Easing,
  FlatList,
  Image,
  Modal,
  PermissionsAndroid,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import {
  addRecentQuery,
  deleteAsset,
  getRecentQueries,
  getSettings,
  indexGallery,
  indexVideos,
  libraryStats,
  onIndexProgress,
  onVideoProgress,
  openInGallery,
  openVideoAt,
  searchByAsset,
  type LibraryStats,
  type SearchHit,
} from '../native/SiftEmbedder';
import { searchByText } from '../native/search';
import { Shimmer } from '../components/Shimmer';
import { SettingsModal } from '../components/SettingsModal';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

const NUM_COLUMNS = 3;
const DEFAULT_MATCH_MIN = 60; // only show results at or above this match %, until settings load
const DEFAULT_TOP_K = 5; // max results fetched per search, until settings load
const GAP = 2;
const TILE = (Dimensions.get('window').width - GAP * (NUM_COLUMNS + 1)) / NUM_COLUMNS;

const EXAMPLE_QUERIES = [
  'sunset',
  'a dog or cat',
  'food',
  'people smiling',
  'documents',
  'at the beach',
  'gym',
];

type PermissionState = 'checking' | 'granted' | 'denied';

async function requestMediaPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') return false;
  if (Platform.Version >= 33) {
    const res = await PermissionsAndroid.requestMultiple([
      PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES,
      PermissionsAndroid.PERMISSIONS.READ_MEDIA_VIDEO,
      // Optional: only gates the indexing-progress notification, not core
      // functionality, so a decline here doesn't affect the return value.
      PermissionsAndroid.PERMISSIONS.POST_NOTIFICATIONS,
    ]);
    return (
      res[PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES] ===
      PermissionsAndroid.RESULTS.GRANTED
    );
  }
  const result = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE,
  );
  return result === PermissionsAndroid.RESULTS.GRANTED;
}

function formatTimestamp(ms: number): string {
  const totalSec = Math.round(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// CLIP cosine scores sit ~0.15-0.32 for good matches; map to a friendlier
// "match %" so the number reads meaningfully to a user. Prompt-templated
// queries (see native/search.ts) shift this distribution vs. raw-query
// scores, so these need re-tuning against real on-device queries.
const SCORE_FLOOR = 0.1; // score mapped to 0%
const SCORE_CEILING = 0.3; // score mapped to 100%
function matchPercent(score: number): number {
  return Math.max(
    1,
    Math.min(
      99,
      Math.round(((score - SCORE_FLOOR) / (SCORE_CEILING - SCORE_FLOOR)) * 100),
    ),
  );
}

/** Thin determinate progress bar for "N / total" indexing counts. */
function ProgressBar({ progress }: { progress: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, progress)) * 100);
  return (
    <View style={styles.progressTrack}>
      <View style={[styles.progressFill, { width: `${pct}%` }]} />
    </View>
  );
}

/** Gently pulsing dot for the "indexing" status. */
function PulseDot() {
  const v = useRef(new Animated.Value(0.4)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(v, { toValue: 1, duration: 700, useNativeDriver: true }),
        Animated.timing(v, { toValue: 0.4, duration: 700, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [v]);
  return <Animated.View style={[styles.pulseDot, { opacity: v }]} />;
}

// Enough rows to fill the screen while a search runs.
const SKELETON_COUNT =
  Math.ceil((Dimensions.get('window').height - 120) / (TILE + GAP)) * NUM_COLUMNS;

/** Full-screen shimmer skeleton shown while a search runs. */
function SkeletonGrid() {
  return (
    <View style={styles.skeletonGrid}>
      {Array.from({ length: SKELETON_COUNT }).map((_, i) => (
        <Shimmer key={i} width={TILE} height={TILE} radius={4} style={styles.skelTile} />
      ))}
    </View>
  );
}

function EmptyState({
  stats,
  indexing,
  failedCount,
  onPickExample,
}: {
  stats: LibraryStats | null;
  indexing: boolean;
  failedCount: number;
  onPickExample: (q: string) => void;
}) {
  const enter = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(enter, {
      toValue: 1,
      duration: 450,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [enter]);

  const translateY = enter.interpolate({ inputRange: [0, 1], outputRange: [16, 0] });

  // A highlight that travels chip to chip, hinting the examples are tappable.
  // Each chip owns an Animated value; moving the highlight cross-fades the old
  // chip down and the new one up, so the transition is smooth (not a jump).
  const anims = useRef(EXAMPLE_QUERIES.map(() => new Animated.Value(0))).current;
  const [highlight, setHighlight] = useState(0);
  useEffect(() => {
    const iv = setInterval(
      () => setHighlight(p => (p + 1) % EXAMPLE_QUERIES.length),
      1500,
    );
    return () => clearInterval(iv);
  }, []);
  useEffect(() => {
    Animated.parallel(
      anims.map((a, i) =>
        Animated.timing(a, {
          toValue: i === highlight ? 1 : 0,
          duration: 500,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: false,
        }),
      ),
    ).start();
  }, [highlight, anims]);

  return (
    <Animated.View style={[styles.empty, { opacity: enter, transform: [{ translateY }] }]}>
      <Text style={styles.heroMark}>Sift</Text>
      <Text style={styles.heroTagline}>
        Find any photo or video on your device, just by describing it.
      </Text>
      <Text style={styles.tryHint}>Try an example</Text>
      <View style={styles.chips}>
        {EXAMPLE_QUERIES.map((q, i) => {
          const a = anims[i];
          return (
            <AnimatedPressable
              key={q}
              onPress={() => onPickExample(q)}
              style={[
                styles.chip,
                {
                  borderColor: a.interpolate({
                    inputRange: [0, 1],
                    outputRange: ['#2a2a2a', '#3b82f6'],
                  }),
                  backgroundColor: a.interpolate({
                    inputRange: [0, 1],
                    outputRange: ['#171717', '#182541'],
                  }),
                  transform: [
                    {
                      scale: a.interpolate({
                        inputRange: [0, 1],
                        outputRange: [1, 1.05],
                      }),
                    },
                  ],
                },
              ]}
            >
              <Text style={styles.chipText}>{q}</Text>
            </AnimatedPressable>
          );
        })}
      </View>

      {stats && (
        <View style={styles.indexStatus}>
          <Text style={styles.indexStatusText}>
            {stats.photosIndexed.toLocaleString()} / {stats.photosTotal.toLocaleString()} photos
          </Text>
          {indexing && stats.photosTotal > 0 && (
            <ProgressBar progress={stats.photosIndexed / stats.photosTotal} />
          )}
          <Text style={[styles.indexStatusText, styles.videoStatusText]}>
            {stats.videosIndexed.toLocaleString()} / {stats.videosTotal.toLocaleString()} videos
          </Text>
          {indexing && stats.videosTotal > 0 && (
            <ProgressBar progress={stats.videosIndexed / stats.videosTotal} />
          )}
          {indexing && (
            <>
              <View style={styles.indexingRow}>
                <PulseDot />
                <Text style={styles.indexingText}>Indexing…</Text>
              </View>
              <Text style={styles.indexHelp}>
                Making your photos and videos searchable. New ones are added
                automatically. You can already search whatever is ready.
              </Text>
            </>
          )}
          {failedCount > 0 && (
            <Text style={styles.indexFailedText}>
              {failedCount} item{failedCount === 1 ? '' : 's'} couldn't be indexed
            </Text>
          )}
        </View>
      )}
    </Animated.View>
  );
}


export function PhotoGridScreen() {
  const [permissionState, setPermissionState] =
    useState<PermissionState>('checking');
  const [stats, setStats] = useState<LibraryStats | null>(null);
  const [indexingPhotos, setIndexingPhotos] = useState(false);
  const [indexingVideos, setIndexingVideos] = useState(false);
  const indexing = indexingPhotos || indexingVideos;
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [viewerItem, setViewerItem] = useState<SearchHit | null>(null);
  const [matchMin, setMatchMin] = useState(DEFAULT_MATCH_MIN);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  // Non-null while showing "find similar" results instead of a text search.
  const [similarTo, setSimilarTo] = useState<SearchHit | null>(null);
  const [searchError, setSearchError] = useState(false);
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const [searchFocused, setSearchFocused] = useState(false);
  const [failedCount, setFailedCount] = useState(0);

  useEffect(() => {
    requestMediaPermission().then(granted =>
      setPermissionState(granted ? 'granted' : 'denied'),
    );
  }, []);

  useEffect(() => {
    getSettings()
      .then(s => {
        setMatchMin(s.matchMinPercent);
        setTopK(s.topK);
      })
      .catch(e => console.log('getSettings failed', e));
  }, []);

  const refreshRecentQueries = useCallback(() => {
    getRecentQueries()
      .then(setRecentQueries)
      .catch(e => console.log('getRecentQueries failed', e));
  }, []);

  useEffect(() => {
    refreshRecentQueries();
  }, [refreshRecentQueries]);

  // Called by Settings when match % or result count changes, so search
  // behaves with the new values immediately rather than after a re-open.
  const onSearchSettingsChanged = useCallback((min: number, k: number) => {
    setMatchMin(min);
    setTopK(k);
  }, []);

  const refreshStats = useCallback(() => {
    libraryStats()
      .then(setStats)
      .catch(e => console.log('libraryStats failed', e));
  }, []);

  const removeFromResults = useCallback((item: SearchHit) => {
    setResults(prev =>
      prev
        ? prev.filter(
            r => !(r.assetId === item.assetId && r.isVideo === item.isVideo),
          )
        : prev,
    );
  }, []);

  const confirmDelete = useCallback(
    (item: SearchHit) => {
      Alert.alert(
        'Delete this item?',
        'This permanently removes it from your device and can’t be undone.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Delete',
            style: 'destructive',
            onPress: () => {
              deleteAsset(item.uri, item.isVideo)
                .then(deleted => {
                  if (deleted) {
                    removeFromResults(item);
                    setViewerItem(cur =>
                      cur && cur.assetId === item.assetId && cur.isVideo === item.isVideo
                        ? null
                        : cur,
                    );
                    refreshStats();
                  }
                })
                .catch(() => Alert.alert('Could not delete', 'Please try again.'));
            },
          },
        ],
      );
    },
    [removeFromResults, refreshStats],
  );

  // Counts in-flight calls rather than blocking re-entry: a scope change while a
  // pass is already running must still kick off a fresh (correctly-scoped) call
  // rather than get silently dropped — the native side cancels the stale one
  // quickly once superseded, so overlap here is brief, not wasted duplicate work.
  const activePhotoIndexCalls = useRef(0);
  const startIndex = useCallback(async () => {
    activePhotoIndexCalls.current++;
    setIndexingPhotos(true);
    try {
      const result = await indexGallery(0);
      if (result.failed > 0) setFailedCount(n => n + result.failed);
      refreshStats();
    } finally {
      activePhotoIndexCalls.current--;
      if (activePhotoIndexCalls.current === 0) setIndexingPhotos(false);
    }
  }, [refreshStats]);

  const activeVideoIndexCalls = useRef(0);
  const startVideoIndex = useCallback(async () => {
    activeVideoIndexCalls.current++;
    setIndexingVideos(true);
    try {
      const result = await indexVideos(0);
      if (result.videosFailed > 0) setFailedCount(n => n + result.videosFailed);
      refreshStats();
    } finally {
      activeVideoIndexCalls.current--;
      if (activeVideoIndexCalls.current === 0) setIndexingVideos(false);
    }
  }, [refreshStats]);

  useEffect(() => {
    if (permissionState !== 'granted') return;
    refreshStats();
    // Refresh the indexed/total counts as indexing progresses (throttled by the
    // native progress cadence, which fires every ~10 items).
    const offPhotos = onIndexProgress(() => refreshStats());
    const offVideos = onVideoProgress(() => refreshStats());
    return () => {
      offPhotos();
      offVideos();
    };
  }, [permissionState, refreshStats]);

  // Photos and videos index concurrently on separate native threads (encoder
  // calls are still serialized under the hood, but decode/IO overlaps).
  // Respects the scope set in Settings.
  const runFullIndex = useCallback(() => {
    startVideoIndex();
    startIndex();
  }, [startVideoIndex, startIndex]);

  // Foreground auto-index on open.
  const autoIndexed = useRef(false);
  useEffect(() => {
    if (permissionState === 'granted' && !autoIndexed.current) {
      autoIndexed.current = true;
      runFullIndex();
    }
  }, [permissionState, runFullIndex]);

  // When the user changes what to index, re-run so it prunes/adds to match.
  const onScopeChanged = useCallback(() => {
    setFailedCount(0);
    refreshStats();
    runFullIndex();
  }, [refreshStats, runFullIndex]);

  const searchSeq = useRef(0);
  const runSearch = useCallback(async (raw: string) => {
    const q = raw.trim();
    if (q === '') {
      setResults(null);
      setSearching(false);
      return;
    }
    const seq = ++searchSeq.current;
    setSearching(true);
    setSearchError(false);
    try {
      const hits = await searchByText(q, topK);
      if (seq === searchSeq.current) {
        setResults(hits);
        addRecentQuery(q)
          .then(refreshRecentQueries)
          .catch(e => console.log('addRecentQuery failed', e));
      }
    } catch (e) {
      console.log('search error', e);
      if (seq === searchSeq.current) {
        setResults(null);
        setSearchError(true);
      }
    } finally {
      if (seq === searchSeq.current) setSearching(false);
    }
  }, [topK, refreshRecentQueries]);

  useEffect(() => {
    if (query.trim() === '') {
      // A "find similar" run also clears the query (to swap out of text-search
      // mode) — don't invalidate that in-flight request here, only a genuine
      // clear-to-empty.
      if (similarTo === null) {
        // Invalidate any in-flight search so a late result can't repopulate
        // the cleared grid ("30 results for '' ").
        searchSeq.current++;
        setResults(null);
        setSearching(false);
        setSearchError(false);
      }
      return;
    }
    setSimilarTo(null);
    const t = setTimeout(() => runSearch(query), 350);
    return () => clearTimeout(t);
  }, [query, runSearch, similarTo]);

  const pickExample = useCallback(
    (q: string) => {
      setQuery(q);
      runSearch(q);
    },
    [runSearch],
  );

  const clearSearch = useCallback(() => {
    setQuery('');
    setResults(null);
    setSimilarTo(null);
    setSearchError(false);
  }, []);

  // "Find similar": no text query involved, so it bypasses runSearch/searchByText
  // and hits the instant asset-embedding search directly.
  const searchSimilar = useCallback(
    async (item: SearchHit) => {
      const seq = ++searchSeq.current;
      setQuery('');
      setSimilarTo(item);
      setSearching(true);
      setSearchError(false);
      try {
        const hits = await searchByAsset(item.assetId, item.isVideo, topK);
        if (seq === searchSeq.current) setResults(hits);
      } catch (e) {
        console.log('find similar error', e);
        if (seq === searchSeq.current) {
          setResults(null);
          setSearchError(true);
        }
      } finally {
        if (seq === searchSeq.current) setSearching(false);
      }
    },
    [topK],
  );

  const retrySearch = useCallback(() => {
    if (similarTo) searchSimilar(similarTo);
    else runSearch(query);
  }, [similarTo, searchSimilar, runSearch, query]);

  // Only treat results as "showable" once a text query or a "find similar"
  // is active — guards against a stale in-flight search landing after the
  // query was cleared.
  const hasResults = (query.trim() !== '' || similarTo !== null) && results !== null;
  const showRecentQueries =
    searchFocused &&
    !searching &&
    !searchError &&
    query.trim() === '' &&
    similarTo === null &&
    recentQueries.length > 0;
  // Only surface confident matches.
  const shownResults = (results ?? []).filter(
    r => matchPercent(r.score) >= matchMin,
  );

  if (permissionState === 'checking') {
    return (
      <View style={styles.center}>
        <Text style={styles.heroMark}>Sift</Text>
      </View>
    );
  }
  if (permissionState === 'denied') {
    return (
      <View style={styles.center}>
        <Text style={styles.heroMark}>Sift</Text>
        <Text style={[styles.message, styles.deniedText]}>
          Sift needs access to your photos & videos to search them.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.searchRow}>
        <TextInput
          style={styles.searchInput}
          placeholder="Search your photos & videos…"
          placeholderTextColor="#6b7280"
          value={query}
          onChangeText={setQuery}
          onSubmitEditing={() => runSearch(query)}
          onFocus={() => setSearchFocused(true)}
          onBlur={() => setSearchFocused(false)}
          returnKeyType="search"
          autoCapitalize="none"
        />
        {query !== '' || similarTo !== null ? (
          <Pressable
            style={styles.iconButton}
            onPress={clearSearch}
            accessibilityLabel="Clear search"
            accessibilityRole="button"
          >
            <Text style={styles.iconButtonText}>✕</Text>
          </Pressable>
        ) : (
          <Pressable
            style={styles.iconButton}
            onPress={() => setSettingsVisible(true)}
            accessibilityLabel="Settings"
            accessibilityRole="button"
          >
            <Text style={styles.iconButtonText}>⚙</Text>
          </Pressable>
        )}
      </View>

      {showRecentQueries && (
        <View style={styles.chips}>
          {recentQueries.map(q => (
            <Pressable
              key={q}
              onPress={() => pickExample(q)}
              style={styles.chip}
            >
              <Text style={styles.chipText}>{q}</Text>
            </Pressable>
          ))}
        </View>
      )}

      {hasResults && !searching && shownResults.length > 0 && (
        <Text style={styles.resultCount}>
          {similarTo
            ? `${shownResults.length} similar items`
            : `${shownResults.length} results for “${query.trim()}”`}
        </Text>
      )}

      {searching ? (
        <SkeletonGrid />
      ) : searchError ? (
        <Pressable style={styles.errorState} onPress={retrySearch}>
          <Text style={styles.errorText}>Search failed — tap to retry</Text>
        </Pressable>
      ) : !hasResults ? (
        <EmptyState
          stats={stats}
          indexing={indexing}
          failedCount={failedCount}
          onPickExample={pickExample}
        />
      ) : shownResults.length === 0 ? (
        <View style={styles.noMatch}>
          <Text style={styles.noMatchText}>
            {similarTo
              ? 'No strong similar matches.'
              : `No strong matches for “${query.trim()}”.`}
          </Text>
          <Text style={styles.noMatchHint}>
            Try describing it differently, or index more of your library.
          </Text>
        </View>
      ) : (
        <FlatList
          data={shownResults}
          numColumns={NUM_COLUMNS}
          keyExtractor={r => `${r.assetId}${r.isVideo ? `-${r.timestampMs}` : ''}`}
          columnWrapperStyle={styles.row}
          contentContainerStyle={styles.gridContent}
          renderItem={({ item }) => (
            <Pressable
              onPress={() =>
                item.isVideo
                  ? openVideoAt(item.uri, item.timestampMs)
                  : setViewerItem(item)
              }
              onLongPress={() =>
                Alert.alert('', undefined, [
                  {
                    text: 'Find Similar',
                    onPress: () => searchSimilar(item),
                  },
                  {
                    text: 'Open in Gallery',
                    onPress: () => openInGallery(item.uri, item.isVideo),
                  },
                  {
                    text: 'Delete',
                    style: 'destructive',
                    onPress: () => confirmDelete(item),
                  },
                  { text: 'Cancel', style: 'cancel' },
                ])
              }
              delayLongPress={300}
            >
              <Image source={{ uri: item.uri }} style={styles.thumbnail} />
              <View style={styles.matchBadge}>
                <Text style={styles.matchText}>{matchPercent(item.score)}%</Text>
              </View>
              {item.isVideo && (
                <View style={styles.videoBadge}>
                  <Text style={styles.videoBadgeText}>
                    ▶ {formatTimestamp(item.timestampMs)}
                  </Text>
                </View>
              )}
            </Pressable>
          )}
        />
      )}

      <SettingsModal
        visible={settingsVisible}
        onClose={() => setSettingsVisible(false)}
        onScopeChanged={onScopeChanged}
        onReset={onScopeChanged}
        onSearchSettingsChanged={onSearchSettingsChanged}
      />

      <Modal
        visible={viewerItem !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setViewerItem(null)}
      >
        <Pressable style={styles.viewer} onPress={() => setViewerItem(null)}>
          {viewerItem && (
            <Image
              source={{ uri: viewerItem.uri }}
              style={styles.viewerImage}
              resizeMode="contain"
            />
          )}
          {viewerItem && (
            <View style={styles.viewerActions}>
              <Pressable
                style={styles.openGalleryBtn}
                onPress={() => {
                  const item = viewerItem;
                  setViewerItem(null);
                  searchSimilar(item);
                }}
              >
                <Text style={styles.openGalleryText}>Find Similar</Text>
              </Pressable>
              <Pressable
                style={styles.openGalleryBtn}
                onPress={() => openInGallery(viewerItem.uri, false)}
              >
                <Text style={styles.openGalleryText}>Open in Gallery</Text>
              </Pressable>
              <Pressable
                style={styles.deleteBtn}
                onPress={() => confirmDelete(viewerItem)}
              >
                <Text style={styles.openGalleryText}>Delete</Text>
              </Pressable>
            </View>
          )}
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a' },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#0a0a0a',
  },
  message: { color: '#fff', textAlign: 'center', fontSize: 16 },
  deniedText: { color: '#9ca3af', fontSize: 14, marginTop: 14 },

  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingTop: 12,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    color: '#fff',
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 11,
    fontSize: 15,
    borderWidth: 1,
    borderColor: '#262626',
  },
  iconButton: {
    backgroundColor: '#1a1a1a',
    width: 42,
    height: 42,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#262626',
  },
  iconButtonText: { color: '#9ca3af', fontSize: 16 },

  resultCount: {
    color: '#9ca3af',
    fontSize: 13,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },

  row: { gap: GAP, paddingHorizontal: GAP },
  gridContent: { gap: GAP, paddingTop: 8 },
  thumbnail: { width: TILE, height: TILE, borderRadius: 4, backgroundColor: '#1a1a1a' },
  videoBadge: {
    position: 'absolute',
    bottom: 6,
    left: 6,
    backgroundColor: 'rgba(0,0,0,0.72)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  videoBadgeText: { color: '#fff', fontSize: 11, fontWeight: '600' },

  skeletonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: GAP,
    paddingHorizontal: GAP,
    paddingTop: 8,
  },
  skelTile: { borderRadius: 4 },

  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
    paddingBottom: 60,
  },
  heroMark: {
    color: '#fff',
    fontSize: 46,
    fontWeight: '800',
    letterSpacing: -1,
  },
  heroTagline: {
    color: '#9ca3af',
    fontSize: 15,
    textAlign: 'center',
    marginTop: 10,
    lineHeight: 22,
  },
  matchBadge: {
    position: 'absolute',
    top: 6,
    left: 6,
    backgroundColor: 'rgba(0,0,0,0.72)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  matchText: { color: '#e5e7eb', fontSize: 10.5, fontWeight: '700' },
  tryHint: {
    color: '#6b7280',
    fontSize: 12,
    marginTop: 30,
    marginBottom: 12,
  },
  chips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
  },
  chip: {
    backgroundColor: '#171717',
    borderWidth: 1,
    borderColor: '#2a2a2a',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
  },
  chipText: { color: '#d1d5db', fontSize: 13 },

  indexStatus: {
    alignItems: 'center',
    marginTop: 40,
  },
  indexStatusText: {
    color: '#6b7280',
    fontSize: 12,
    textAlign: 'center',
    lineHeight: 18,
  },
  videoStatusText: { marginTop: 6 },
  progressTrack: {
    width: 200,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#262626',
    marginTop: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
    backgroundColor: '#3b82f6',
  },
  indexingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 8,
  },
  indexingText: { color: '#6b7280', fontSize: 12 },
  indexHelp: {
    color: '#6b7280',
    fontSize: 11.5,
    textAlign: 'center',
    lineHeight: 17,
    marginTop: 8,
    maxWidth: 300,
  },
  indexFailedText: {
    color: '#f59e0b',
    fontSize: 11.5,
    textAlign: 'center',
    marginTop: 10,
  },
  pulseDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#22c55e',
  },
  viewer: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.95)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  viewerImage: { width: '100%', height: '100%' },
  viewerActions: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 10,
  },
  openGalleryBtn: {
    backgroundColor: 'rgba(37,99,235,0.9)',
    paddingHorizontal: 20,
    paddingVertical: 11,
    borderRadius: 24,
  },
  deleteBtn: {
    backgroundColor: 'rgba(220,38,38,0.9)',
    paddingHorizontal: 20,
    paddingVertical: 11,
    borderRadius: 24,
  },
  openGalleryText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  noMatch: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    paddingBottom: 80,
  },
  noMatchText: { color: '#e5e7eb', fontSize: 16, textAlign: 'center' },
  noMatchHint: {
    color: '#6b7280',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 8,
  },
  errorState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    paddingBottom: 80,
  },
  errorText: { color: '#f87171', fontSize: 15, textAlign: 'center' },
});
