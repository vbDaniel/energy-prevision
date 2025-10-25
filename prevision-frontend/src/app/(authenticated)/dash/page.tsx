"use client";

import { Zap, TrendingDown, DollarSign, Activity } from "lucide-react";
import {
  AlertCard,
  ComparisonCard,
  ConsumptionChart,
  MetricCard,
  Sidebar,
  EconomyTips,
} from "src/components";
import {
  PageWrap,
  Main,
  Container,
  HeaderBlock,
  Title,
  Subtitle,
  MetricsGrid,
  BottomGrid,
  LeftArea,
} from "./styles";

const Index = () => {
  return (
    <PageWrap>
      <Sidebar />

      <Main>
        <Container>
          {/* Header */}
          <HeaderBlock>
            <Title>Bem-vindo ao EnergyFlow</Title>
            <Subtitle>
              Monitore e preveja seu consumo energético de forma inteligente
            </Subtitle>
          </HeaderBlock>

          {/* Metrics Grid */}
          <MetricsGrid>
            <MetricCard
              title="Consumo Atual"
              value="342 kWh"
              change="+8% vs meta"
              icon={Zap}
              trend="up"
              iconBgClass="bg-primary/10"
            />
            <MetricCard
              title="Economia Mensal"
              value="R$ 124"
              change="-12% vs mês anterior"
              icon={TrendingDown}
              trend="down"
              iconBgClass="bg-primary/10"
            />
            <MetricCard
              title="Custo Estimado"
              value="R$ 298"
              change="+R$ 24 vs previsto"
              icon={DollarSign}
              trend="up"
              iconBgClass="bg-secondary/10"
            />
            <MetricCard
              title="Eficiência"
              value="87%"
              change="Boa performance"
              icon={Activity}
              trend="neutral"
              iconBgClass="bg-primary/10"
            />
          </MetricsGrid>

          {/* Main Chart */}
          <ConsumptionChart />

          {/* Bottom Grid */}
          <BottomGrid>
            <LeftArea>
              <AlertCard />
            </LeftArea>
            <ComparisonCard />
          </BottomGrid>

          {/* Economy Tips */}
          <EconomyTips />
        </Container>
      </Main>
    </PageWrap>
  );
};

export default Index;
