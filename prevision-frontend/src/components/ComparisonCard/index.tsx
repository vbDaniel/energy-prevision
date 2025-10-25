"use client";

import { Card, Progress } from "@/components/ui";
import { TrendingDown } from "lucide-react";

import {
  ComparisonGrid,
  Consumption,
  Header,
  IconWrapper,
  TipBox,
  TitleGroup,
  StyledCard,
  Section,
} from "./styles";

export const ComparisonCard = () => {
  const currentMonth = 342;
  const predicted = 330;
  const percentDiff = (((currentMonth - predicted) / predicted) * 100).toFixed(
    1
  );
  const progressValue = (currentMonth / 400) * 100;

  return (
    <StyledCard>
      <Header>
        <TitleGroup>
          <h2>Previsto vs Real</h2>
          <p>Comparação mensal</p>
        </TitleGroup>
        <IconWrapper>
          <TrendingDown size={20} color={"hsl(var(--secondary))"} />
        </IconWrapper>
      </Header>

      <Section>
        <Consumption>
          <div className="header">
            <span>Consumo Atual</span>
            <span>{currentMonth} kWh</span>
          </div>
          <Progress value={progressValue} />
        </Consumption>

        <ComparisonGrid>
          <div>
            <p className="label">Meta do Mês</p>
            <p className="value">{predicted} kWh</p>
          </div>
          <div>
            <p className="label">Diferença</p>
            <p className="value diff">+{percentDiff}%</p>
          </div>
        </ComparisonGrid>

        <TipBox>
          <p>
            💡 <span>Dica:</span> Você pode economizar até R$ 45 reduzindo o
            consumo nos próximos dias.
          </p>
        </TipBox>
      </Section>
    </StyledCard>
  );
};
