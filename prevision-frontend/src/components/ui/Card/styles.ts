import styled from "styled-components";

/* CARD PRINCIPAL */
export const StyledCard = styled.div`
  border-radius: 12px;
  border: 1px solid hsl(var(--border));
  background-color: hsl(var(--card));
  color: hsl(var(--card-foreground));
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
`;

/* HEADER */
export const StyledCardHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 24px;
`;

/* TÍTULO */
export const StyledCardTitle = styled.h3`
  font-size: 1.5rem; /* text-2xl */
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: -0.01em;
  margin: 0;
`;

/* DESCRIÇÃO */
export const StyledCardDescription = styled.p`
  font-size: 0.875rem; /* text-sm */
  color: hsl(var(--muted-foreground)); /* text-muted-foreground */
  margin: 0;
`;

/* CONTEÚDO */
export const StyledCardContent = styled.div`
  padding: 24px;
  padding-top: 0;
  flex: 1;
`;

/* RODAPÉ */
export const StyledCardFooter = styled.div`
  display: flex;
  align-items: center;
  padding: 24px;
  padding-top: 0;
  border-top: 1px solid hsl(var(--border));
`;
