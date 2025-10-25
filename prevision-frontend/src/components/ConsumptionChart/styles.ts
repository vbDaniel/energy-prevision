import styled, { keyframes } from "styled-components";
import CardModule from "@/components/ui/Card";

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
`;

export const ChartCard = styled(CardModule.Card)`
  padding: 1.5rem; /* p-6 */
  border: 1px solid hsl(var(--border)); /* border-border/50 approximation */
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
  animation: ${fadeIn} 300ms ease-out;

  @media (prefers-reduced-motion: reduce) {
    animation: none;
  }
`;

export const HeaderRow = styled.div`
  display: flex; /* flex */
  align-items: center; /* items-center */
  justify-content: space-between; /* justify-between */
  margin-bottom: 1.5rem; /* mb-6 */
`;

export const Title = styled.h2`
  font-size: 1.25rem; /* text-xl */
  font-weight: 700; /* font-bold */
  color: hsl(var(--foreground)); /* text-foreground */
  margin: 0;
`;

export const Subtitle = styled.p`
  font-size: 0.875rem; /* text-sm */
  color: hsl(var(--muted-foreground)); /* text-muted-foreground */
  margin: 0;
`;

export const Controls = styled.div`
  display: flex; /* flex */
  gap: 0.5rem; /* gap-2 */
`;
