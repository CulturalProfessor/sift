/**
 * SiftEmbedder is a native module with no JS-side implementation, so it's
 * undefined under Jest (no native runtime). NativeEventEmitter throws if
 * constructed with a null/undefined module, so SiftEmbedder.ts (loaded
 * transitively by App.tsx) crashes on import without this stub.
 */
const { NativeModules } = require('react-native');

NativeModules.SiftEmbedder = {
  addListener: jest.fn(),
  removeListeners: jest.fn(),
};
