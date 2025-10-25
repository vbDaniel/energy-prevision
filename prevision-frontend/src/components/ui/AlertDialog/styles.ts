import styled, { keyframes } from "styled-components";
import * as AlertDialogPrimitive from "@radix-ui/react-alert-dialog";

/* ======================== */
/* ======== ANIMAÇÕES ===== */
/* ======================== */
const fadeIn = keyframes`
  from { opacity: 0; }
  to { opacity: 1; }
`;

const fadeOut = keyframes`
  from { opacity: 1; }
  to { opacity: 0; }
`;

const scaleIn = keyframes`
  from { opacity: 0; transform: translate(-50%, -48%) scale(0.95); }
  to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
`;

const scaleOut = keyframes`
  from { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  to { opacity: 0; transform: translate(-50%, -48%) scale(0.95); }
`;

/* ======================== */
/* ======== OVERLAY ======= */
/* ======================== */
export const StyledOverlay = styled(AlertDialogPrimitive.Overlay)`
  position: fixed;
  inset: 0;
  z-index: 50;
  background: rgba(0, 0, 0, 0.8);
  animation: ${fadeIn} 0.2s ease forwards;

  &[data-state="closed"] {
    animation: ${fadeOut} 0.2s ease forwards;
  }
`;

/* ======================== */
/* ======== CONTENT ======= */
/* ======================== */
export const StyledContent = styled(AlertDialogPrimitive.Content)`
  position: fixed;
  left: 50%;
  top: 50%;
  z-index: 60;
  width: 100%;
  max-width: 500px;
  transform: translate(-50%, -50%);
  border-radius: 10px;
  background: #ffffff;
  color: #0f172a;
  border: 1px solid #e2e8f0;
  padding: 24px;
  box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.25);
  animation: ${scaleIn} 0.2s ease forwards;

  &[data-state="closed"] {
    animation: ${scaleOut} 0.2s ease forwards;
  }
`;

/* ======================== */
/* ======= HEADER ========= */
/* ======================== */
export const StyledHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-align: center;

  @media (min-width: 640px) {
    text-align: left;
  }
`;

/* ======================== */
/* ======= FOOTER ========= */
/* ======================== */
export const StyledFooter = styled.div`
  display: flex;
  flex-direction: column-reverse;
  margin-top: 16px;

  @media (min-width: 640px) {
    flex-direction: row;
    justify-content: flex-end;
    gap: 8px;
  }
`;

/* ======================== */
/* ======= TITLE ========== */
/* ======================== */
export const StyledTitle = styled(AlertDialogPrimitive.Title)`
  font-size: 1.125rem;
  font-weight: 600;
  color: #0f172a;
`;

/* ======================== */
/* ===== DESCRIPTION ====== */
/* ======================== */
export const StyledDescription = styled(AlertDialogPrimitive.Description)`
  font-size: 0.875rem;
  color: #64748b;
`;

/* ======================== */
/* ======= BUTTONS ======== */
/* ======================== */
export const StyledAction = styled(AlertDialogPrimitive.Action)`
  background-color: #3b82f6;
  color: #fff;
  font-weight: 500;
  border: none;
  border-radius: 6px;
  padding: 10px 16px;
  cursor: pointer;
  transition: background 0.2s ease;

  &:hover {
    background-color: #2563eb;
  }

  &:focus {
    outline: 2px solid #2563eb;
    outline-offset: 2px;
  }
`;

export const StyledCancel = styled(AlertDialogPrimitive.Cancel)`
  background-color: transparent;
  color: #0f172a;
  font-weight: 500;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 10px 16px;
  cursor: pointer;
  transition: background 0.2s ease;

  &:hover {
    background-color: #f1f5f9;
  }

  &:focus {
    outline: 2px solid #94a3b8;
    outline-offset: 2px;
  }
`;
