import styled, { css } from "styled-components";

type Variant =
  | "default"
  | "destructive"
  | "outline"
  | "secondary"
  | "ghost"
  | "link";
type Size = "default" | "sm" | "lg" | "icon";

interface StyledButtonProps {
  $variant: Variant;
  $size: Size;
}

export const StyledButton = styled.button<StyledButtonProps>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  white-space: nowrap;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  transition: all 0.2s ease;
  cursor: pointer;
  border: none;
  outline: none;
  user-select: none;

  &:focus-visible {
    outline: 2px solid hsl(var(--ring));
    outline-offset: 2px;
  }

  &:disabled {
    opacity: 0.5;
    pointer-events: none;
  }

  /* ====== Variants ====== */
  ${({ $variant }) => {
    switch ($variant) {
      case "destructive":
        return css`
          background-color: hsl(var(--destructive));
          color: hsl(var(--destructive-foreground));
          &:hover {
            filter: brightness(0.95);
          }
        `;
      case "outline":
        return css`
          background-color: transparent;
          border: 1px solid hsl(var(--border));
          color: hsl(var(--foreground));
          &:hover {
            background-color: hsl(var(--muted));
            color: hsl(var(--foreground));
          }
        `;
      case "secondary":
        return css`
          background-color: hsl(var(--secondary));
          color: hsl(var(--secondary-foreground));
          &:hover {
            background-color: hsl(var(--secondary-light));
          }
        `;
      case "ghost":
        return css`
          background-color: transparent;
          color: hsl(var(--foreground));
          &:hover {
            background-color: hsl(var(--muted));
          }
        `;
      case "link":
        return css`
          background: none;
          color: hsl(var(--primary));
          text-decoration: none;
          &:hover {
            text-decoration: underline;
          }
        `;
      default:
        return css`
          background-color: hsl(var(--primary));
          color: hsl(var(--primary-foreground));
          &:hover {
            background-color: hsl(var(--primary-dark));
          }
        `;
    }
  }}

  /* ====== Sizes ====== */
  ${({ $size }) => {
    switch ($size) {
      case "sm":
        return css`
          height: 36px;
          padding: 0 12px;
          border-radius: 5px;
        `;
      case "lg":
        return css`
          height: 44px;
          padding: 0 24px;
          border-radius: 6px;
        `;
      case "icon":
        return css`
          width: 40px;
          height: 40px;
          padding: 0;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        `;
      default:
        return css`
          height: 40px;
          padding: 0 16px;
        `;
    }
  }}
`;
