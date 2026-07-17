/**
 * Shared color tokens. Sift is intentionally dark-only (no light theme) —
 * this is a single source of truth for the palette, not a themeable system.
 */
export const colors = {
  background: '#0a0a0a',
  surface: '#1a1a1a',
  surfaceRaised: '#141414',
  chip: '#171717',

  border: '#262626',
  borderLight: '#2a2a2a',
  borderMuted: '#3f3f46',

  text: '#ffffff',
  textSecondary: '#e5e7eb',
  textTertiary: '#d1d5db',
  textMuted: '#9ca3af',
  textFaint: '#6b7280',

  accent: '#2563eb',
  accentStrong: '#3b82f6',
  accentSurface: '#0f1729',
  accentSurfaceAlt: '#182541',

  danger: '#dc2626',
  dangerText: '#f87171',
  dangerSurface: '#1a0f0f',
  dangerSurfaceStrong: '#3b0d0d',
  dangerBorder: '#3f1d1d',

  warning: '#f59e0b',
  success: '#22c55e',

  radioBorder: '#4b5563',
  skeleton: '#1f2937',
} as const;
