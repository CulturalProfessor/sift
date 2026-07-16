// Hermes provides a global TextEncoder at runtime, but the RN tsconfig
// intentionally omits the "dom" lib (its browser globals don't apply here),
// so TypeScript doesn't know about it. Minimal ambient decl for the subset
// actually used (src/native/tokenizer.ts).
declare class TextEncoder {
  encode(input: string): Uint8Array;
}
