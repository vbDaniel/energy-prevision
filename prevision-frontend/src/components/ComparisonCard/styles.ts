import styled, { keyframes } from "styled-components";
import { Card } from "../ui";

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

export const StyledCard = styled(Card.Card)`
  padding: 1.5rem;
  background: linear-gradient(
    180deg,
    hsl(var(--card)) 0%,
    hsl(var(--card)) 100%
  );
  border-color: hsl(var(--border) / 0.5);
  animation: ${fadeIn} 300ms ease-out both;
`;

export const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 16px;
`;

export const TitleGroup = styled.div`
  h2 {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 4px;
  }

  p {
    font-size: 0.875rem;
    color: hsl(var(--muted-foreground));
  }
`;

export const IconWrapper = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 12px;
  background-color: hsl(var(--muted) / 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

export const Consumption = styled.div`
  .header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 12px;

    span:first-child {
      font-size: 0.875rem;
      font-weight: 500;
      color: hsl(var(--muted-foreground));
    }

    span:last-child {
      font-size: 1.5rem;
      font-weight: 700;
      color: hsl(var(--foreground));
    }
  }

  .progress {
    height: 12px;
  }
`;

export const ComparisonGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  padding-top: 16px;
  border-top: 1px solid hsl(var(--border));

  p {
    margin: 0;
  }

  .label {
    font-size: 0.75rem;
    color: hsl(var(--muted-foreground));
    margin-bottom: 4px;
  }

  .value {
    font-size: 1.125rem;
    font-weight: 700;
    color: hsl(var(--foreground));
  }

  .diff {
    color: hsl(var(--secondary));
  }
`;

export const TipBox = styled.div`
  background-color: hsl(var(--muted) / 0.1);
  border-radius: 12px;
  padding: 16px;

  p {
    font-size: 0.875rem;
    color: hsl(var(--muted-foreground));

    span {
      font-weight: 600;
      color: hsl(var(--foreground));
    }
  }
`;
