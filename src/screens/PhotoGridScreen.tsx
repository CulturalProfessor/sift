import { useCallback, useEffect, useRef, useState } from 'react';
import {
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
  indexGallery,
  indexVideos,
  libraryStats,
  onIndexProgress,
  openVideoAt,
  type LibraryStats,
  type SearchHit,
} from '../native/SiftEmbedder';
import { searchByText } from '../native/search';
import { Shimmer } from '../components/Shimmer';
import { SettingsModal } from '../components/SettingsModal';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

const NUM_COLUMNS = 3;
const GAP = 2;
const TILE = (Dimensions.get('window').width - GAP * (NUM_COLUMNS + 1)) / NUM_COLUMNS;

const EXAMPLE_QUERIES = [
  'sunset',
  'a dog or cat',
  'food',
  'people smiling',
  'documents',
  'at the beach',
];

type PermissionState = 'checking' | 'granted' | 'denied';

async function requestMediaPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') return false;
  if (Platform.Version >= 33) {
    const res = await PermissionsAndroid.requestMultiple([
      PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES,
      PermissionsAndroid.PERMISSIONS.READ_MEDIA_VIDEO,
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
// "match %" so the number reads meaningfully to a user.
function matchPercent(score: number): number {
  return Math.max(1, Math.min(99, Math.round(((score - 0.1) / 0.2) * 100)));
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
  onPickExample,
}: {
  stats: LibraryStats | null;
  indexing: boolean;
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
          <Text style={styles.indexStatusText}>
            {stats.videosIndexed.toLocaleString()} / {stats.videosTotal.toLocaleString()} videos
          </Text>
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
        </View>
      )}
    </Animated.View>
  );
}

export function PhotoGridScreen() {
  const [permissionState, setPermissionState] =
    useState<PermissionState>('checking');
  const [stats, setStats] = useState<LibraryStats | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [viewerUri, setViewerUri] = useState<string | null>(null);

  useEffect(() => {
    requestMediaPermission().then(granted =>
      setPermissionState(granted ? 'granted' : 'denied'),
    );
  }, []);

  const refreshStats = useCallback(() => {
    libraryStats().then(setStats).catch(() => {});
  }, []);

  const indexingRef = useRef(false);
  const startIndex = useCallback(async () => {
    if (indexingRef.current) return;
    indexingRef.current = true;
    setIndexing(true);
    try {
      await indexGallery(0);
      refreshStats();
    } finally {
      indexingRef.current = false;
      setIndexing(false);
    }
  }, [refreshStats]);

  useEffect(() => {
    if (permissionState !== 'granted') return;
    refreshStats();
    // Refresh the indexed/total counts as indexing progresses (throttled by the
    // native progress cadence, which fires every ~10 items).
    const off = onIndexProgress(() => refreshStats());
    return off;
  }, [permissionState, refreshStats]);

  // Videos first (fewer), then photos. Respects the scope set in Settings.
  const runFullIndex = useCallback(() => {
    setIndexing(true);
    indexVideos(0)
      .catch(() => {})
      .finally(() => {
        refreshStats();
        startIndex();
      });
  }, [startIndex, refreshStats]);

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
    try {
      const hits = await searchByText(q, 30);
      if (seq === searchSeq.current) setResults(hits);
    } catch (e) {
      console.log('search error', e);
    } finally {
      if (seq === searchSeq.current) setSearching(false);
    }
  }, []);

  useEffect(() => {
    if (query.trim() === '') {
      // Invalidate any in-flight search so a late result can't repopulate the
      // cleared grid ("30 results for '' ").
      searchSeq.current++;
      setResults(null);
      setSearching(false);
      return;
    }
    const t = setTimeout(() => runSearch(query), 350);
    return () => clearTimeout(t);
  }, [query, runSearch]);

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
  }, []);

  // Only treat results as "showable" when the query is non-empty — guards
  // against a stale in-flight search landing after the field was cleared.
  const hasResults = query.trim() !== '' && results !== null;

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
          returnKeyType="search"
          autoCapitalize="none"
        />
        {query !== '' ? (
          <Pressable style={styles.iconButton} onPress={clearSearch}>
            <Text style={styles.iconButtonText}>✕</Text>
          </Pressable>
        ) : (
          <Pressable
            style={styles.iconButton}
            onPress={() => setSettingsVisible(true)}
          >
            <Text style={styles.iconButtonText}>⚙</Text>
          </Pressable>
        )}
      </View>

      {hasResults && !searching && (
        <Text style={styles.resultCount}>
          {results!.length} results for “{query.trim()}”
        </Text>
      )}

      {searching ? (
        <SkeletonGrid />
      ) : hasResults ? (
        <FlatList
          data={results ?? []}
          numColumns={NUM_COLUMNS}
          keyExtractor={r => `${r.assetId}${r.isVideo ? `-${r.timestampMs}` : ''}`}
          columnWrapperStyle={styles.row}
          contentContainerStyle={styles.gridContent}
          renderItem={({ item }) => (
            <Pressable
              onPress={() =>
                item.isVideo
                  ? openVideoAt(item.uri, item.timestampMs)
                  : setViewerUri(item.uri)
              }
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
      ) : (
        <EmptyState
          stats={stats}
          indexing={indexing}
          onPickExample={pickExample}
        />
      )}

      <SettingsModal
        visible={settingsVisible}
        onClose={() => setSettingsVisible(false)}
        onScopeChanged={onScopeChanged}
        onReset={onScopeChanged}
      />

      <Modal
        visible={viewerUri !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setViewerUri(null)}
      >
        <Pressable style={styles.viewer} onPress={() => setViewerUri(null)}>
          {viewerUri && (
            <Image
              source={{ uri: viewerUri }}
              style={styles.viewerImage}
              resizeMode="contain"
            />
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
});
