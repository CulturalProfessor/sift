import { useEffect, useState } from 'react';
import {
  Alert,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  View,
} from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import Slider from '@react-native-community/slider';
import { colors } from '../theme';
import {
  clearIndex,
  getSettings,
  setIndexScope,
  setIndexThrottle,
  setMatchMin,
  setTopK,
} from '../native/SiftEmbedder';

const DAY = 86400000;

const RANGE_PRESETS = [
  { label: 'All time', days: 0 },
  { label: 'Past year', days: 365 },
  { label: 'Past 6 months', days: 182 },
  { label: 'Past 3 months', days: 91 },
  { label: 'Past month', days: 30 },
];

const MAXFILE_PRESETS = [
  { label: 'No limit', n: 0 },
  { label: '1,000 most recent', n: 1000 },
  { label: '5,000 most recent', n: 5000 },
  { label: '10,000 most recent', n: 10000 },
  { label: '20,000 most recent', n: 20000 },
];

const SPEED_PRESETS = [
  { label: 'Eco', ms: 250, desc: 'Coolest, slowest indexing' },
  { label: 'Balanced', ms: 120, desc: 'Moderate heat and speed' },
  { label: 'Fast', ms: 40, desc: 'Fastest, runs warmer' },
];

function nearestSpeed(ms: number): number {
  return SPEED_PRESETS.reduce(
    (best, p) => (Math.abs(p.ms - ms) < Math.abs(best - ms) ? p.ms : best),
    SPEED_PRESETS[0].ms,
  );
}

function sinceForDays(days: number): number {
  return days === 0 ? 0 : Date.now() - days * DAY;
}

/** Which range preset (index) the current sinceMs matches, or -1 for a custom date. */
function matchedRange(sinceMs: number): number {
  if (sinceMs === 0) return 0;
  const i = RANGE_PRESETS.findIndex(
    p => p.days > 0 && Math.abs(sinceForDays(p.days) - sinceMs) < 2 * DAY,
  );
  return i;
}

export function SettingsModal({
  visible,
  onClose,
  onScopeChanged,
  onReset,
  onSearchSettingsChanged,
}: {
  visible: boolean;
  onClose: () => void;
  onScopeChanged: () => void;
  onReset: () => void;
  onSearchSettingsChanged: (matchMin: number, topK: number) => void;
}) {
  const [throttle, setThrottle] = useState<number | null>(null);
  const [deviceDefault, setDeviceDefault] = useState<number | null>(null);
  const [sinceMs, setSinceMs] = useState(0);
  const [maxFiles, setMaxFiles] = useState(0);
  const [videosOn, setVideosOn] = useState(true);
  const [showPicker, setShowPicker] = useState(false);
  const [matchMin, setMatchMinState] = useState(60);
  const [topK, setTopKState] = useState(5);

  useEffect(() => {
    if (visible) {
      getSettings()
        .then(s => {
          setThrottle(nearestSpeed(s.throttleMs));
          setDeviceDefault(nearestSpeed(s.deviceDefaultMs));
          setSinceMs(s.indexSinceMs);
          setMaxFiles(s.indexMaxFiles);
          setVideosOn(s.indexVideos);
          setMatchMinState(s.matchMinPercent);
          setTopKState(s.topK);
        })
        .catch(() => {});
    }
  }, [visible]);

  const applyMatchMin = (percent: number) => {
    setMatchMinState(percent);
    setMatchMin(percent)
      .then(() => onSearchSettingsChanged(percent, topK))
      .catch(() => {});
  };

  const applyTopK = (k: number) => {
    setTopKState(k);
    setTopK(k)
      .then(() => onSearchSettingsChanged(matchMin, k))
      .catch(() => {});
  };

  const selectSpeed = (ms: number) => {
    setThrottle(ms);
    setIndexThrottle(ms).catch(() => {});
  };

  const applyScope = (since: number, max: number, videos: boolean) => {
    setSinceMs(since);
    setMaxFiles(max);
    setVideosOn(videos);
    setIndexScope(since, max, videos).then(onScopeChanged).catch(() => {});
  };

  const confirmReset = () => {
    Alert.alert(
      'Reset indexed data?',
      'This clears all indexed photos and videos, then re-indexes from scratch. Your actual photos are not touched.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => clearIndex().then(onReset).catch(() => {}),
        },
      ],
    );
  };

  const rangeIdx = matchedRange(sinceMs);
  const customActive = sinceMs > 0 && rangeIdx === -1;

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <Pressable style={styles.backdrop} onPress={onClose}>
        <Pressable style={styles.sheet} onPress={() => {}}>
          <View style={styles.header}>
            <Text style={styles.title}>Settings</Text>
            <Pressable
              onPress={onClose}
              hitSlop={12}
              accessibilityLabel="Close settings"
              accessibilityRole="button"
            >
              <Text style={styles.close}>✕</Text>
            </Pressable>
          </View>

          <ScrollView showsVerticalScrollIndicator={false}>
            <Text style={styles.section}>Minimum match</Text>
            <Text style={styles.hint}>
              Hide results below this match confidence.
            </Text>
            <View style={styles.sliderRow}>
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={95}
                step={5}
                value={matchMin}
                onSlidingComplete={applyMatchMin}
                minimumTrackTintColor={colors.accent}
                maximumTrackTintColor={colors.borderMuted}
                thumbTintColor={colors.accent}
              />
              <Text style={styles.sliderValue}>{matchMin}%</Text>
            </View>

            <Text style={[styles.section, styles.sectionSpaced]}>Result count</Text>
            <Text style={styles.hint}>Max results fetched per search.</Text>
            <View style={styles.sliderRow}>
              <Slider
                style={styles.slider}
                minimumValue={1}
                maximumValue={50}
                step={1}
                value={topK}
                onSlidingComplete={applyTopK}
                minimumTrackTintColor={colors.accent}
                maximumTrackTintColor={colors.borderMuted}
                thumbTintColor={colors.accent}
              />
              <Text style={styles.sliderValue}>{topK}</Text>
            </View>

            <Text style={[styles.section, styles.sectionSpaced]}>How far back to index</Text>
            <Text style={styles.hint}>Only index items newer than this.</Text>
            {RANGE_PRESETS.map((p, i) => {
              const selected = rangeIdx === i;
              return (
                <Pressable
                  key={p.label}
                  style={[styles.option, selected && styles.optionSelected]}
                  onPress={() => applyScope(sinceForDays(p.days), maxFiles, videosOn)}
                >
                  <Text style={styles.optionLabel}>{p.label}</Text>
                  <View style={[styles.radio, selected && styles.radioOn]}>
                    {selected && <View style={styles.radioDot} />}
                  </View>
                </Pressable>
              );
            })}
            <Pressable
              style={[styles.option, customActive && styles.optionSelected]}
              onPress={() => setShowPicker(true)}
            >
              <Text style={styles.optionLabel}>
                {customActive
                  ? `From ${new Date(sinceMs).toLocaleDateString()}`
                  : 'Choose exact date…'}
              </Text>
              <View style={[styles.radio, customActive && styles.radioOn]}>
                {customActive && <View style={styles.radioDot} />}
              </View>
            </Pressable>
            {showPicker && (
              <DateTimePicker
                value={sinceMs > 0 ? new Date(sinceMs) : new Date()}
                mode="date"
                maximumDate={new Date()}
                onValueChange={(_event, date) => {
                  setShowPicker(false);
                  if (date) applyScope(date.getTime(), maxFiles, videosOn);
                }}
                onDismiss={() => setShowPicker(false)}
              />
            )}

            <Text style={[styles.section, styles.sectionSpaced]}>Maximum files</Text>
            <Text style={styles.hint}>
              Cap how many of the most recent items get indexed.
            </Text>
            {MAXFILE_PRESETS.map(p => {
              const selected = maxFiles === p.n;
              return (
                <Pressable
                  key={p.label}
                  style={[styles.option, selected && styles.optionSelected]}
                  onPress={() => applyScope(sinceMs, p.n, videosOn)}
                >
                  <Text style={styles.optionLabel}>{p.label}</Text>
                  <View style={[styles.radio, selected && styles.radioOn]}>
                    {selected && <View style={styles.radioDot} />}
                  </View>
                </Pressable>
              );
            })}

            <Pressable
              style={styles.toggleRow}
              onPress={() => applyScope(sinceMs, maxFiles, !videosOn)}
            >
              <View style={styles.optionLeft}>
                <Text style={styles.optionLabel}>Index videos</Text>
                <Text style={styles.optionDesc}>
                  Search inside videos. Turning off removes video results.
                </Text>
              </View>
              <Switch
                value={videosOn}
                onValueChange={v => applyScope(sinceMs, maxFiles, v)}
                trackColor={{ true: colors.accent, false: colors.borderMuted }}
                thumbColor={colors.text}
              />
            </Pressable>

            <Text style={[styles.section, styles.sectionSpaced]}>Indexing speed</Text>
            <Text style={styles.hint}>
              Faster indexing warms the phone more. Set for your device.
            </Text>
            {SPEED_PRESETS.map(p => {
              const selected = throttle === p.ms;
              return (
                <Pressable
                  key={p.label}
                  style={[styles.option, selected && styles.optionSelected]}
                  onPress={() => selectSpeed(p.ms)}
                >
                  <View style={styles.optionLeft}>
                    <Text style={styles.optionLabel}>
                      {p.label}
                      {deviceDefault === p.ms ? '  · recommended' : ''}
                    </Text>
                    <Text style={styles.optionDesc}>{p.desc}</Text>
                  </View>
                  <View style={[styles.radio, selected && styles.radioOn]}>
                    {selected && <View style={styles.radioDot} />}
                  </View>
                </Pressable>
              );
            })}

            <Text style={[styles.section, styles.sectionSpaced]}>Data</Text>
            <Pressable style={styles.resetBtn} onPress={confirmReset}>
              <Text style={styles.resetText}>Reset indexed data</Text>
            </Pressable>

            <Text style={[styles.section, styles.sectionSpaced]}>About</Text>
            <Text style={styles.about}>
              Sift searches your photos and videos by meaning. Describe what you
              remember and it finds the moment, including the exact spot inside a
              video.
            </Text>
            <Text style={styles.about}>
              Everything runs entirely on your device. No internet, no cloud, no
              account. Your photos and videos never leave your phone.
            </Text>
          </ScrollView>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'flex-end' },
  sheet: {
    backgroundColor: colors.surfaceRaised,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    paddingBottom: 34,
    maxHeight: '88%',
    borderTopWidth: 1,
    borderColor: colors.border,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 18,
  },
  title: { color: colors.text, fontSize: 20, fontWeight: '700' },
  close: { color: colors.textMuted, fontSize: 18 },
  section: { color: colors.text, fontSize: 15, fontWeight: '600', marginBottom: 4 },
  sectionSpaced: { marginTop: 22 },
  hint: { color: colors.textFaint, fontSize: 12, marginBottom: 14 },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  slider: { flex: 1, height: 40 },
  sliderValue: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '600',
    width: 44,
    textAlign: 'right',
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 13,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: 10,
  },
  optionSelected: { borderColor: colors.accent, backgroundColor: colors.accentSurface },
  optionLeft: { flex: 1, paddingRight: 12 },
  optionLabel: { color: colors.text, fontSize: 15, fontWeight: '500' },
  optionDesc: { color: colors.textFaint, fontSize: 12, marginTop: 2 },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    marginTop: 4,
  },
  radio: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: colors.radioBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioOn: { borderColor: colors.accent },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: colors.accent },
  resetBtn: {
    paddingVertical: 13,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.dangerBorder,
    backgroundColor: colors.dangerSurface,
    alignItems: 'center',
  },
  resetText: { color: colors.dangerText, fontSize: 15, fontWeight: '500' },
  about: { color: colors.textMuted, fontSize: 13, lineHeight: 20, marginBottom: 10 },
});
