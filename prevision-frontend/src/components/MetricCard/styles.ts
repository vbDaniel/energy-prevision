import styled, { keyframes, css } from "styled-components";
import CardModule from "@/components/ui/Card";

export type Trend = "up" | "down" | "neutral";

const fadeInScale = keyframes`
  from { opacity: 0; transform: translateY(4px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
`;

export const MetricCardWrap = styled(CardModule.Card)`
  padding: 1.5rem; /* p-6 */
  border: 1px solid hsl(var(--border)); /* border-border/50 */
  background: linear-gradient(180deg, hsl(var(--card)) 0%, hsl(var(--card)) 100%),
    radial-gradient(
      65% 85% at 10% 0%,
      hsl(var(--primary) / 0.12) 0%,
      hsl(var(--primary) / 0.06) 25%,
      transparent 45%
    ),
    radial-gradient(
      75% 85% at 100% 100%,
      hsl(var(--secondary) / 0.1) 0%,
      hsl(var(--secondary) / 0.04) 30%,
      transparent 50%
    );
  animation: ${fadeInScale} 300ms ease-out;
  transition: box-shadow 200ms ease;

  &:hover {
    box-shadow: var(--shadow-md); /* hover:shadow-md */
  }

  @media (prefers-reduced-motion: reduce) {
    animation: none;
    transition: none;
  }
`;

export const Row = styled.div`
  display: flex; /* flex */
  align-items: flex-start; /* items-start */
  justify-content: space-between; /* justify-between */
`;

export const Left = styled.div`
  flex: 1; /* flex-1 */
`;

export const SmallTitle = styled.p`
  font-size: 0.875rem; /* text-sm */
  color: hsl(var(--muted-foreground)); /* text-muted-foreground */
  margin: 0 0 0.5rem 0; /* mb-2 */
`;

export const Value = styled.h3`
  font-size: 1.875rem; /* text-3xl */
  font-weight: 700; /* font-bold */
  color: hsl(var(--foreground)); /* text-foreground */
  margin: 0 0 0.25rem 0; /* mb-1 */
`;

export const Change = styled.p<{ $trend: Trend }>`
  font-size: 0.875rem; /* text-sm */
  font-weight: 500; /* font-medium */
  margin: 0;
  ${({ $trend }) => {
    switch ($trend) {
      case "up":
        return css`
          color: hsl(var(--destructive)); /* destructive */
        `;
      case "down":
        return css`
          color: hsl(var(--primary)); /* primary */
        `;
      default:
        return css`
          color: hsl(var(--muted-foreground)); /* muted-foreground */
        `;
    }
  }}
`;

export const IconBox = styled.div<{ $bgColor?: string }>`
  width: 48px; /* w-12 */
  height: 48px; /* h-12 */
  border-radius: 12px; /* rounded-xl */
  display: flex; /* flex */
  align-items: center; /* items-center */
  justify-content: center; /* justify-center */
  background-color: ${({ $bgColor }) =>
    $bgColor || "hsl(var(--primary) / 0.10)"}; /* bg-primary/10 */
`;
