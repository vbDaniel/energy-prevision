import { createGlobalStyle } from "styled-components";
import type { DefaultTheme } from "styled-components";
import { colors, gradients, shadows, transitions, radii } from "./colors";

// Global CSS variables and base styles aligned with the provided Tailwind tokens
export const GlobalStyle = createGlobalStyle`
  /* Design system: colors, gradients, radii, shadows, transitions */
  :root {
    /* Base palette (light mode) */
    --background: 0 0% 98%;
    --foreground: 160 10% 15%;

    --card: 0 0% 100%;
    --card-foreground: 160 10% 15%;

    --popover: 0 0% 100%;
    --popover-foreground: 160 10% 15%;

    /* Green energy primary color */
    --primary: 158 64% 52%;
    --primary-foreground: 0 0% 100%;
    --primary-light: 158 64% 65%;
    --primary-dark: 158 64% 42%;

    /* Yellow alert/accent color */
    --secondary: 43 96% 56%;
    --secondary-foreground: 160 10% 15%;
    --secondary-light: 43 96% 70%;

    --muted: 160 10% 96%;
    --muted-foreground: 160 10% 45%;

    --accent: 43 96% 56%;
    --accent-foreground: 160 10% 15%;

    --destructive: 0 84% 60%;
    --destructive-foreground: 0 0% 100%;

    --border: 160 15% 90%;
    --input: 160 15% 90%;
    --ring: 158 64% 52%;

    --radius: 0.75rem;

    /* Sidebar gradient colors */
    --sidebar-background: 158 64% 52%;
    --sidebar-foreground: 0 0% 100%;
    --sidebar-muted: 158 30% 80%;
    --sidebar-accent: 0 0% 100%;
    --sidebar-accent-foreground: 158 64% 52%;
    --sidebar-border: 158 30% 70%;

    /* Custom gradients */
    --gradient-energy: linear-gradient(135deg, hsl(158 64% 52%) 0%, hsl(0 0% 100%) 100%);
    --gradient-alert: linear-gradient(135deg, hsl(43 96% 56%) 0%, hsl(38 92% 50%) 100%);
    --gradient-card: linear-gradient(180deg, hsl(0 0% 100%) 0%, hsl(158 20% 98%) 100%);

    /* Shadows */
    --shadow-sm: 0 2px 8px hsl(158 64% 52% / 0.1);
    --shadow-md: 0 4px 16px hsl(158 64% 52% / 0.15);
    --shadow-lg: 0 8px 24px hsl(158 64% 52% / 0.2);
    --shadow-glow: 0 0 32px hsl(158 64% 52% / 0.3);

    /* Animations */
    --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-bounce: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
  }

  .dark {
    --background: 160 20% 8%;
    --foreground: 0 0% 95%;

    --card: 160 15% 12%;
    --card-foreground: 0 0% 95%;

    --popover: 160 15% 12%;
    --popover-foreground: 0 0% 95%;

    --primary: 158 64% 52%;
    --primary-foreground: 0 0% 100%;
    --primary-light: 158 64% 65%;
    --primary-dark: 158 64% 42%;

    --secondary: 43 96% 56%;
    --secondary-foreground: 160 10% 15%;
    --secondary-light: 43 96% 70%;

    --muted: 160 15% 18%;
    --muted-foreground: 160 10% 65%;

    --accent: 43 96% 56%;
    --accent-foreground: 160 10% 15%;

    --destructive: 0 84% 60%;
    --destructive-foreground: 0 0% 100%;

    --border: 160 15% 20%;
    --input: 160 15% 20%;
    --ring: 158 64% 52%;

    --sidebar-background: 160 18% 10%;
    --sidebar-foreground: 0 0% 95%;
    --sidebar-muted: 160 15% 30%;
    --sidebar-accent: 158 64% 52%;
    --sidebar-accent-foreground: 0 0% 100%;
    --sidebar-border: 160 15% 20%;

    --gradient-energy: linear-gradient(135deg, hsl(158 64% 42%) 0%, hsl(160 18% 10%) 100%);
    --gradient-alert: linear-gradient(135deg, hsl(43 96% 56%) 0%, hsl(38 92% 50%) 100%);
    --gradient-card: linear-gradient(180deg, hsl(160 15% 12%) 0%, hsl(160 18% 10%) 100%);

    --shadow-sm: 0 2px 8px hsl(0 0% 0% / 0.3);
    --shadow-md: 0 4px 16px hsl(0 0% 0% / 0.4);
    --shadow-lg: 0 8px 24px hsl(0 0% 0% / 0.5);
    --shadow-glow: 0 0 32px hsl(158 64% 52% / 0.4);
  }

  /* Base layer styles analogous to Tailwind's @layer base */
  *, *::before, *::after {
    box-sizing: border-box;
    border-color: hsl(var(--border));
  }

  html {
    color-scheme: light;
  }

  body {
    margin: 0;
    background: hsl(var(--background));
    color: hsl(var(--foreground));
  }
`;

export interface AppTheme extends DefaultTheme {
  colors: typeof colors;
  gradients: typeof gradients;
  shadows: typeof shadows;
  transitions: typeof transitions;
  radii: typeof radii;
  isDark?: boolean;
}

export const baseTheme: Omit<AppTheme, "isDark"> = {
  colors,
  gradients,
  shadows,
  transitions,
  radii,
};

// Since tokens reference CSS variables, the same object works for light/dark.
// You may toggle the `.dark` class on <html> or <body> to switch palettes.
export const lightTheme: AppTheme = {
  ...baseTheme,
  isDark: false,
};

export const darkTheme: AppTheme = {
  ...baseTheme,
  isDark: true,
};