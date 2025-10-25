// Styled-components color tokens aligned with Tailwind CSS variables
// All colors are HSL-based and reference CSS custom properties for automatic light/dark support.

export const colors = {
  border: "hsl(var(--border))",
  input: "hsl(var(--input))",
  ring: "hsl(var(--ring))",
  background: "hsl(var(--background))",
  foreground: "hsl(var(--foreground))",
  primary: {
    DEFAULT: "hsl(var(--primary))",
    foreground: "hsl(var(--primary-foreground))",
    light: "hsl(var(--primary-light))",
    dark: "hsl(var(--primary-dark))",
  },
  secondary: {
    DEFAULT: "hsl(var(--secondary))",
    foreground: "hsl(var(--secondary-foreground))",
    light: "hsl(var(--secondary-light))",
  },
  destructive: {
    DEFAULT: "hsl(var(--destructive))",
    foreground: "hsl(var(--destructive-foreground))",
  },
  muted: {
    DEFAULT: "hsl(var(--muted))",
    foreground: "hsl(var(--muted-foreground))",
  },
  accent: {
    DEFAULT: "hsl(var(--accent))",
    foreground: "hsl(var(--accent-foreground))",
  },
  popover: {
    DEFAULT: "hsl(var(--popover))",
    foreground: "hsl(var(--popover-foreground))",
  },
  card: {
    DEFAULT: "hsl(var(--card))",
    foreground: "hsl(var(--card-foreground))",
  },
  sidebar: {
    DEFAULT: "hsl(var(--sidebar-background))",
    foreground: "hsl(var(--sidebar-foreground))",
    muted: "hsl(var(--sidebar-muted))",
    accent: "hsl(var(--sidebar-accent))",
    accentForeground: "hsl(var(--sidebar-accent-foreground))",
    border: "hsl(var(--sidebar-border))",
  },
} as const;

export const gradients = {
  energy: "var(--gradient-energy)",
  alert: "var(--gradient-alert)",
  card: "var(--gradient-card)",
} as const;

export const shadows = {
  sm: "var(--shadow-sm)",
  md: "var(--shadow-md)",
  lg: "var(--shadow-lg)",
  glow: "var(--shadow-glow)",
} as const;

export const transitions = {
  smooth: "var(--transition-smooth)",
  bounce: "var(--transition-bounce)",
} as const;

export const radii = {
  base: "var(--radius)",
} as const;

export type Colors = typeof colors;
export type Gradients = typeof gradients;
export type Shadows = typeof shadows;
export type Transitions = typeof transitions;
export type Radii = typeof radii;