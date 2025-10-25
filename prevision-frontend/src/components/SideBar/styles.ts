import styled, { keyframes, css } from "styled-components";

const slideInLeft = keyframes`
  from { opacity: 0; transform: translateX(-8px); }
  to { opacity: 1; transform: translateX(0); }
`;

export const SidebarWrap = styled.aside`
  width: 16rem; /* w-64 */
  min-height: 100vh; /* min-h-screen */
  position: relative;
  border-right: 1px solid hsl(var(--sidebar-border)); /* border-r border-sidebar-border */
  background: linear-gradient(180deg, hsl(var(--sidebar-background)) 0%, hsl(var(--sidebar-background)) 100%),
    radial-gradient(
      60% 70% at 0% 0%,
      hsl(var(--primary) / 0.15) 0%,
      hsl(var(--primary) / 0.05) 35%,
      transparent 60%
    ),
    radial-gradient(
      55% 75% at 100% 100%,
      hsl(var(--secondary) / 0.12) 0%,
      hsl(var(--secondary) / 0.04) 30%,
      transparent 60%
    );
  color: hsl(var(--sidebar-foreground));
  animation: ${slideInLeft} 300ms ease-out; /* animate-slide-in-left */

  @media (prefers-reduced-motion: reduce) {
    animation: none;
  }
`;

export const Inner = styled.div`
  padding: 1.5rem; /* p-6 */
`;

export const Header = styled.div`
  display: flex; /* flex */
  align-items: center; /* items-center */
  gap: 0.75rem; /* gap-3 */
  margin-bottom: 2rem; /* mb-8 */
`;

export const LogoBox = styled.div`
  width: 40px; /* w-10 */
  height: 40px; /* h-10 */
  background-color: hsl(var(--sidebar-accent)); /* bg-white */
  border-radius: 12px; /* rounded-xl */
  display: flex; /* flex */
  align-items: center; /* items-center */
  justify-content: center; /* justify-center */
  box-shadow: var(--shadow-md); /* shadow-md */

  & svg {
    width: 24px;
    height: 24px;
    color: hsl(var(--primary)); /* text-primary */
  }
`;

export const TitleWrap = styled.div``;

export const Title = styled.h1`
  font-size: 1.25rem; /* text-xl */
  font-weight: 700; /* font-bold */
  color: hsl(var(--sidebar-foreground)); /* text-white */
  margin: 0;
`;

export const SubTitle = styled.p`
  font-size: 0.75rem; /* text-xs */
  color: hsl(var(--sidebar-foreground) / 0.8); /* text-white/80 */
  margin: 0;
`;

export const Nav = styled.nav`
  display: flex; /* space-y-2 */
  flex-direction: column;
  gap: 0.5rem; /* space-y-2 */
`;

export const MenuButton = styled.button<{ $active?: boolean }>`
  width: 100%; /* w-full */
  display: flex; /* flex */
  align-items: center; /* items-center */
  gap: 0.75rem; /* gap-3 */
  padding: 0.75rem 1rem; /* px-4 py-3 */
  border-radius: 12px; /* rounded-xl */
  transition: all 300ms ease; /* transition-all duration-300 */
  border: none;
  background-color: ${({ $active }) => ($active ? "hsl(var(--sidebar-accent))" : "transparent")};
  color: ${({ $active }) => ($active ? "hsl(var(--primary))" : "hsl(var(--sidebar-foreground) / 0.8)")};
  box-shadow: ${({ $active }) => ($active ? "var(--shadow-md)" : "none")};
  cursor: pointer;

  &:hover {
    background-color: ${({ $active }) => ($active ? "hsl(var(--sidebar-accent))" : "hsl(var(--sidebar-accent) / 0.1)")};
    color: ${({ $active }) => ($active ? "hsl(var(--primary))" : "hsl(var(--sidebar-foreground))")};
  }

  & svg {
    width: 20px;
    height: 20px;
    color: currentColor;
  }

  & span {
    font-weight: 500; /* font-medium */
  }
`;

export const StatsContainer = styled.div`
  position: absolute; /* absolute */
  left: 1.5rem;
  right: 1.5rem;
  bottom: 1.5rem; /* bottom-6 left-6 right-6 */
`;

export const StatsCard = styled.div`
  background-color: hsl(var(--sidebar-accent) / 0.1); /* bg-white/10 */
  backdrop-filter: blur(6px); /* backdrop-blur-sm */
  border-radius: 12px; /* rounded-xl */
  padding: 1rem; /* p-4 */
  border: 1px solid hsl(var(--sidebar-accent) / 0.2); /* border-white/20 */
`;

export const StatTitle = styled.p`
  font-size: 0.75rem; /* text-xs */
  color: hsl(var(--sidebar-foreground) / 0.8); /* text-white/80 */
  margin: 0 0 0.5rem 0; /* mb-2 */
`;

export const StatValue = styled.p`
  font-size: 1.5rem; /* text-2xl */
  font-weight: 700; /* font-bold */
  color: hsl(var(--sidebar-foreground)); /* text-white */
  margin: 0;
`;

export const StatChange = styled.p`
  font-size: 0.75rem; /* text-xs */
  color: hsl(var(--sidebar-foreground) / 0.7); /* text-white/70 */
  margin: 0.25rem 0 0 0; /* mt-1 */
`;
