"use client";

import { ThemeProvider } from "styled-components";
import { GlobalStyle, lightTheme } from "@/lib/theme";
import React from "react";

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider theme={lightTheme}>
      <GlobalStyle />
      {children}
    </ThemeProvider>
  );
}