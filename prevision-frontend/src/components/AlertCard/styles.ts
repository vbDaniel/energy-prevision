import styled, { keyframes, css } from "styled-components";

// Fade-in keyframes to replicate the original entrance animation
const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

// Pure styled-components variant mapping (no Tailwind)
const variantStyles: Record<
  "primary" | "secondary" | "muted",
  ReturnType<typeof css>
> = {
  secondary: css`
    background-color: hsl(var(--secondary) / 0.1);
    color: hsl(var(--secondary-foreground));
  `,
  primary: css`
    background-color: hsl(var(--primary) / 0.1);
    color: hsl(var(--primary));
  `,
  muted: css`
    background-color: hsl(var(--muted));
    color: hsl(var(--muted-foreground));
  `,
};

const getVariantStyles = (variant?: "primary" | "secondary" | "muted") =>
  variant ? variantStyles[variant] : variantStyles.muted;

export const StyledCard = styled.div`
  padding: 1.5rem;
  background: linear-gradient(
    180deg,
    hsl(var(--card)) 0%,
    hsl(var(--card)) 100%
  );
  border-color: hsl(var(--border) / 0.5);
  animation: ${fadeIn} 300ms ease-out both;
`;

export const Title = styled.h2`
  font-size: 1.25rem;
  line-height: 1.75rem;
  font-weight: 700;
  color: hsl(var(--foreground));
  margin-bottom: 1rem;
`;

export const AlertsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

export const AlertItem = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 1rem;
  border-radius: 0.75rem;
  background-color: hsl(var(--card));
  border: 1px solid hsl(var(--border));
  transition: all 300ms ease;

  &:hover {
    box-shadow: var(--shadow-sm);
  }
`;

export const IconWrap = styled.div<{
  $variant?: "primary" | "secondary" | "muted";
}>`
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  ${({ $variant }) => getVariantStyles($variant)};

  & > svg {
    width: 1.25rem;
    height: 1.25rem;
  }
`;

export const Content = styled.div`
  flex: 1;
`;

export const Row = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
`;

export const AlertTitle = styled.h3`
  font-weight: 600;
  color: hsl(var(--foreground));
`;

export const Message = styled.p`
  font-size: 0.875rem;
  line-height: 1.25rem;
  color: hsl(var(--muted-foreground));
`;

export const BadgeWrapper = styled.div`
  font-size: 0.75rem;
  line-height: 1rem;
`;
