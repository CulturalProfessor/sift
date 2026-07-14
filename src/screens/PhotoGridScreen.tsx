import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FlatList,
  Image,
  PermissionsAndroid,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import {
  CameraRoll,
  type PhotoIdentifier,
} from '@react-native-camera-roll/camera-roll';
import {
  indexGallery,
  indexVideos,
  indexedCount,
  indexedIds,
  onIndexProgress,
  type SearchHit,
} from '../native/SiftEmbedder';
import { searchByText } from '../native/search';

const NUM_COLUMNS = 3;

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

/** MediaStore asset id = last path segment of the content uri. */
function assetIdFromUri(uri: string): string {
  return uri.split('/').pop() ?? uri;
}

export function PhotoGridScreen() {
  const [permissionState, setPermissionState] =
    useState<PermissionState>('checking');
  const [photos, setPhotos] = useState<PhotoIdentifier[]>([]);
  const [loading, setLoading] = useState(false);

  const [indexed, setIndexed] = useState(0);
  const [indexedSet, setIndexedSet] = useState<Set<string>>(new Set());
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(
    null,
  );

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    requestMediaPermission().then(granted =>
      setPermissionState(granted ? 'granted' : 'denied'),
    );
  }, []);

  const refreshIndexed = useCallback(() => {
    indexedCount().then(setIndexed).catch(() => {});
    indexedIds()
      .then(ids => setIndexedSet(new Set(ids)))
      .catch(() => {});
  }, []);

  const loadPhotos = useCallback(async () => {
    setLoading(true);
    try {
      const page = await CameraRoll.getPhotos({ first: 90, assetType: 'Photos' });
      setPhotos(page.edges);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (permissionState === 'granted') {
      loadPhotos();
      refreshIndexed();
    }
    return onIndexProgress(setProgress);
  }, [permissionState, loadPhotos, refreshIndexed]);

  const indexingRef = useRef(false);
  const startIndex = useCallback(async () => {
    if (indexingRef.current) return;
    indexingRef.current = true;
    setIndexing(true);
    setProgress(null);
    try {
      const res = await indexGallery(0); // 0 = whole gallery (incremental)
      setIndexed(res.totalIndexed);
      refreshIndexed();
    } finally {
      indexingRef.current = false;
      setIndexing(false);
      setProgress(null);
    }
  }, [refreshIndexed]);

  // Foreground auto-index on app open: incremental, so it only does new work.
  // Videos first (far fewer, so moment-search comes online quickly), then photos.
  // Search stays usable on whatever is already indexed while it runs.
  const autoIndexed = useRef(false);
  useEffect(() => {
    if (permissionState === 'granted' && !autoIndexed.current) {
      autoIndexed.current = true;
      // Capped while validating video keyframing; lift to 0 (all) later.
      indexVideos(15)
        .catch(() => {})
        .finally(() => startIndex());
    }
  }, [permissionState, startIndex]);

  // Guards against out-of-order results: a slower earlier query must not
  // overwrite a newer one's results.
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

  // Debounce search-as-you-type: run ~350ms after the user stops typing.
  useEffect(() => {
    if (query.trim() === '') {
      setResults(null);
      return;
    }
    const t = setTimeout(() => runSearch(query), 350);
    return () => clearTimeout(t);
  }, [query, runSearch]);

  const clearSearch = useCallback(() => {
    setQuery('');
    setResults(null);
  }, []);

  const gridData = useMemo(
    () =>
      results
        ? results.map(r => ({
            id: `${r.assetId}${r.isVideo ? `-${r.timestampMs}` : ''}`,
            uri: r.uri,
            indexed: true,
            isVideo: r.isVideo,
            timestampMs: r.timestampMs,
          }))
        : photos.map(p => {
            const uri = p.node.image.uri;
            return {
              id: p.node.id,
              uri,
              indexed: indexedSet.has(assetIdFromUri(uri)),
              isVideo: false,
              timestampMs: 0,
            };
          }),
    [results, photos, indexedSet],
  );

  if (permissionState === 'checking') {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>Checking permissions…</Text>
      </View>
    );
  }

  if (permissionState === 'denied') {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>
          Sift needs photo library access to index your photos.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.searchRow}>
        <TextInput
          style={styles.searchInput}
          placeholder="Search your photos…"
          placeholderTextColor="#6b7280"
          value={query}
          onChangeText={setQuery}
          onSubmitEditing={() => runSearch(query)}
          returnKeyType="search"
          autoCapitalize="none"
        />
        {results !== null ? (
          <Pressable style={styles.iconButton} onPress={clearSearch}>
            <Text style={styles.iconButtonText}>✕</Text>
          </Pressable>
        ) : (
          <Pressable
            style={[styles.iconButton, indexing && styles.iconButtonDisabled]}
            onPress={startIndex}
            disabled={indexing}
          >
            <Text style={styles.iconButtonText}>
              {indexing ? '…' : 'Index'}
            </Text>
          </Pressable>
        )}
      </View>

      <Text style={styles.status}>
        {searching
          ? 'Searching…'
          : results !== null
            ? `${results.length} results for “${query.trim()}”`
            : progress
              ? `indexing ${progress.done}/${progress.total}`
              : `${loading ? '…' : photos.length} photos · ${indexed} indexed`}
      </Text>

      <FlatList
        data={gridData}
        numColumns={NUM_COLUMNS}
        keyExtractor={item => item.id}
        renderItem={({ item }) => (
          <View style={styles.thumbWrap}>
            <Image source={{ uri: item.uri }} style={styles.thumbnail} />
            {item.indexed && !results && <View style={styles.indexedDot} />}
            {item.isVideo && (
              <View style={styles.videoBadge}>
                <Text style={styles.videoBadgeText}>
                  ▶ {formatTimestamp(item.timestampMs)}
                </Text>
              </View>
            )}
          </View>
        )}
      />
    </View>
  );
}

const THUMB_SIZE = 120;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#000',
  },
  message: { color: '#fff', textAlign: 'center', fontSize: 16 },
  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingTop: 12,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    backgroundColor: '#1f2937',
    color: '#fff',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 15,
  },
  iconButton: {
    backgroundColor: '#2563eb',
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 8,
  },
  iconButtonDisabled: { backgroundColor: '#374151' },
  iconButtonText: { color: '#fff', fontSize: 14 },
  status: {
    color: '#9ca3af',
    fontSize: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  thumbWrap: { position: 'relative' },
  thumbnail: { width: THUMB_SIZE, height: THUMB_SIZE, margin: 1 },
  indexedDot: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#22c55e',
    borderWidth: 1,
    borderColor: '#000',
  },
  videoBadge: {
    position: 'absolute',
    bottom: 6,
    left: 6,
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  videoBadgeText: { color: '#fff', fontSize: 11, fontWeight: '600' },
});
