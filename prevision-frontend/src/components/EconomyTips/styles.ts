import styled, { keyframes } from "styled-components";
import CardModule from "@/components/ui/Card";

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
`;

export const TipsCard = styled(CardModule.Card)`
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

export const Title = styled.h2`
  font-size: 1.25rem; /* text-xl */
  font-weight: 700; /* font-bold */
  color: hsl(var(--foreground)); /* text-foreground */
  margin: 0 0 1rem 0; /* mb-4 */
`;

export const TipsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem; /* space-y-3 equivalent */
`;

export const TipItem = styled.div`
  display: flex; /* flex */
  align-items: flex-start; /* items-start */
  gap: 1rem; /* gap-4 */
  padding: 1rem; /* p-4 */
  border-radius: 12px; /* rounded-xl */
  background-color: hsl(var(--primary) / 0.05); /* bg-primary/5 */
  border: 1px solid hsl(var(--primary) / 0.2); /* border-primary/20 */
  cursor: pointer;
  transition: background-color 200ms ease; /* transition-all duration-300 */

  &:hover {
    background-color: hsl(var(--primary) / 0.1); /* hover:bg-primary/10 */
  }
`;

export const IconWrap = styled.div`
  width: 40px; /* w-10 */
  height: 40px; /* h-10 */
  border-radius: 10px; /* rounded-lg */
  background-color: hsl(var(--primary) / 0.1); /* bg-primary/10 */
  display: flex; /* flex */
  align-items: center; /* items-center */
  justify-content: center; /* justify-center */
`;

export const Content = styled.div`
  flex: 1; /* flex-1 */
`;

export const TipTitle = styled.h3`
  font-weight: 600; /* font-semibold */
  color: hsl(var(--foreground)); /* text-foreground */
  margin: 0 0 0.25rem 0; /* mb-1 */
`;

export const TipDescription = styled.p`
  font-size: 0.875rem; /* text-sm */
  color: hsl(var(--muted-foreground)); /* text-muted-foreground */
  margin: 0 0 0.5rem 0; /* mb-2 */
`;

export const Savings = styled.span`
  font-size: 0.75rem; /* text-xs */
  font-weight: 500; /* font-medium */
  color: hsl(var(--primary)); /* text-primary */
`;
