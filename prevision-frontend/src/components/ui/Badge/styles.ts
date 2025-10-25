import styled, { css } from "styled-components";

export type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

const variantStyles = ({ $variant }: { $variant?: BadgeVariant }) => {
  switch ($variant) {
    case "secondary":
      return css`
        background: hsl(var(--secondary));
        color: hsl(var(--secondary-foreground));
        border-color: transparent;
        &:hover {
          filter: brightness(0.95);
        }
      `;
    case "destructive":
      return css`
        background: hsl(var(--destructive));
        color: hsl(var(--destructive-foreground));
        border-color: transparent;
        &:hover {
          filter: brightness(0.95);
        }
      `;
    case "outline":
      return css`
        background: transparent;
        color: hsl(var(--foreground));
        border-color: currentColor;
        &:hover {
          background: hsl(var(--muted) / 0.08);
        }
      `;
    case "default":
    default:
      return css`
        background: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
        border-color: transparent;
        &:hover {
          filter: brightness(0.95);
        }
      `;
  }
};

export const StyledBadge = styled.div<{ $variant?: BadgeVariant }>`
  display: inline-flex; /* inline-flex */
  align-items: center; /* items-center */
  gap: 0.25rem;
  border-radius: 9999px; /* rounded-full */
  border: 1px solid; /* border */
  padding: 2px 10px; /* px-2.5 py-0.5 */
  font-size: 0.75rem; /* text-xs */
  font-weight: 600; /* font-semibold */
  transition: background-color 200ms ease, color 200ms ease,
    border-color 200ms ease; /* transition-colors */
  outline: none; /* focus:outline-none */

  &:focus-visible {
    box-shadow: 0 0 0 2px hsl(var(--ring) / 0.35);
  }

  ${variantStyles}
`;
