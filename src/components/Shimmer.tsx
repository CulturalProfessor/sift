import { useEffect, useRef } from 'react';
import { Animated, Easing, StyleSheet, View, type ViewStyle } from 'react-native';
import { colors } from '../theme';

/**
 * A dependency-free shimmer box: a dark placeholder with a soft light band that
 * sweeps across on a loop. Used to build loading skeletons.
 */
export function Shimmer({
  width,
  height,
  radius = 8,
  style,
}: {
  width: number;
  height: number;
  radius?: number;
  style?: ViewStyle;
}) {
  const progress = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.timing(progress, {
        toValue: 1,
        duration: 1200,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      }),
    );
    loop.start();
    return () => loop.stop();
  }, [progress]);

  const translateX = progress.interpolate({
    inputRange: [0, 1],
    outputRange: [-width, width],
  });

  return (
    <View
      style={[
        { width, height, borderRadius: radius, backgroundColor: colors.skeleton },
        styles.clip,
        style,
      ]}
    >
      <Animated.View
        style={[
          styles.band,
          { width: width * 0.6, transform: [{ translateX }, { skewX: '-20deg' }] },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  clip: { overflow: 'hidden' },
  band: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
});
