import styled, { keyframes } from "styled-components";

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
`;

export const PageWrap = styled.div`
  display: flex;
  min-height: 100vh;
  background: var(--background, #0f172a);
`;

export const Main = styled.main`
  flex: 1;
  padding: 2rem; /* p-8 */
  overflow: auto;
`;

export const Container = styled.div`
  max-width: 80rem; /* max-w-7xl */
  margin: 0 auto; /* mx-auto */
  display: flex; /* space-y-8 */
  flex-direction: column;
  gap: 2rem; /* space-y-8 */
`;

export const HeaderBlock = styled.div`
  animation: ${fadeIn} 300ms ease-out;
  @media (prefers-reduced-motion: reduce) {
    animation: none;
  }
`;

export const Title = styled.h1`
  font-size: 1.875rem; /* text-3xl */
  line-height: 2.25rem;
  font-weight: 700; /* font-bold */
  color: var(--foreground, #ffffff); /* text-foreground */
  margin: 0 0 0.5rem 0; /* mb-2 */
`;

export const Subtitle = styled.p`
  color: var(
    --muted-foreground,
    rgba(255, 255, 255, 0.7)
  ); /* text-muted-foreground */
  margin: 0;
`;

export const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr; /* grid-cols-1 */
  gap: 1.5rem; /* gap-6 */

  @media (min-width: 768px) {
    /* md */
    grid-template-columns: repeat(2, 1fr);
  }

  @media (min-width: 1024px) {
    /* lg */
    grid-template-columns: repeat(4, 1fr);
  }
`;

export const BottomGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr; /* grid-cols-1 */
  gap: 1.5rem; /* gap-6 */

  @media (min-width: 1024px) {
    /* lg */
    grid-template-columns: repeat(3, 1fr);
  }
`;

export const LeftArea = styled.div`
  grid-column: auto;
  @media (min-width: 1024px) {
    /* lg:col-span-2 */
    grid-column: span 2;
  }
`;
