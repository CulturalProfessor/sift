import { useCallback, useEffect, useState } from 'react';
import {
  FlatList,
  Image,
  PermissionsAndroid,
  Platform,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  CameraRoll,
  type PhotoIdentifier,
} from '@react-native-camera-roll/camera-roll';

const NUM_COLUMNS = 3;

type PermissionState = 'checking' | 'granted' | 'denied';

async function requestMediaPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') {
    return false;
  }

  const permission =
    Platform.Version >= 33
      ? PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES
      : PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE;

  const result = await PermissionsAndroid.request(permission);
  return result === PermissionsAndroid.RESULTS.GRANTED;
}

export function PhotoGridScreen() {
  const [permissionState, setPermissionState] =
    useState<PermissionState>('checking');
  const [photos, setPhotos] = useState<PhotoIdentifier[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    requestMediaPermission().then(granted =>
      setPermissionState(granted ? 'granted' : 'denied'),
    );
  }, []);

  const loadPhotos = useCallback(async () => {
    setLoading(true);
    try {
      const page = await CameraRoll.getPhotos({
        first: 60,
        assetType: 'Photos',
      });
      setPhotos(page.edges);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (permissionState === 'granted') {
      loadPhotos();
    }
  }, [permissionState, loadPhotos]);

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
      <Text style={styles.header}>
        {loading ? 'Loading…' : `${photos.length} photos`}
      </Text>
      <FlatList
        data={photos}
        numColumns={NUM_COLUMNS}
        keyExtractor={item => item.node.id}
        renderItem={({ item }) => (
          <Image
            source={{ uri: item.node.image.uri }}
            style={styles.thumbnail}
          />
        )}
      />
    </View>
  );
}

const THUMB_SIZE = 120;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#000',
  },
  message: {
    color: '#fff',
    textAlign: 'center',
    fontSize: 16,
  },
  header: {
    color: '#fff',
    padding: 12,
    fontSize: 14,
  },
  thumbnail: {
    width: THUMB_SIZE,
    height: THUMB_SIZE,
    margin: 1,
  },
});
