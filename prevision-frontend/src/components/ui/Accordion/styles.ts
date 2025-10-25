import styled, { keyframes } from "styled-components";

const rotateDown = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(180deg); }
`;

export const AccordionWrap = styled.div``;

export const AccordionItemWrap = styled.div`
  border-bottom: 1px solid var(--separator-color, rgba(0, 0, 0, 0.08));
`;

export const TriggerButton = styled.button<{ $open?: boolean }>`
  display: flex; /* flex */
  width: 100%; /* flex-1 */
  align-items: center; /* items-center */
  justify-content: space-between; /* justify-between */
  padding: 1rem 0; /* py-4 */
  font-weight: 500; /* font-medium */
  transition: all 200ms ease; /* transition-all */
  background: transparent;
  border: none;
  color: inherit;
  text-align: left;
  cursor: pointer;

  &:hover {
    text-decoration: underline; /* hover:underline */
  }
`;

export const Chevron = styled.span<{ $open?: boolean }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1rem;
  height: 1rem; /* h-4 w-4 */
  flex-shrink: 0; /* shrink-0 */
  transition: transform 200ms ease; /* transition-transform duration-200 */
  transform: rotate(${({ $open }) => ($open ? 180 : 0)}deg);
`;

export const ContentWrap = styled.div<{ $open?: boolean; $height: number }>`
  overflow: hidden; /* overflow-hidden */
  font-size: 0.875rem; /* text-sm */
  transition: max-height 220ms ease, opacity 220ms ease; /* transition-all */
  max-height: ${({ $open, $height }) => ($open ? `${$height}px` : "0px")};
  opacity: ${({ $open }) => ($open ? 1 : 0)};
`;

export const ContentInner = styled.div`
  padding-bottom: 1rem; /* pb-4 */
  padding-top: 0; /* pt-0 */
`;
