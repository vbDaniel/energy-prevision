"use client";

import { useState } from "react";
import {
  Zap,
  TrendingDown,
  DollarSign,
  Activity,
  PlayCircle,
} from "lucide-react";
import {
  AlertCard,
  ComparisonCard,
  ConsumptionChart,
  MetricCard,
  Sidebar,
  EconomyTips,
} from "src/components";
import Button from "src/components/ui/Button";
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
  const [running, setRunning] = useState(false);
  const [lastCount, setLastCount] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setErrorMsg(null);
    setLastCount(null);
    try {
      const res = await fetch("/api/predict/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lookback: 24 * 7, steps: 24 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Falha ao rodar previsão");
      setLastCount(Number(data?.count ?? 0));
    } catch (e: any) {
      setErrorMsg(String(e?.message || e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <PageWrap>
      <Sidebar />

      <Main>
        <Container>
          {/* Header */}
          <HeaderBlock>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 16,
                flexWrap: "wrap",
              }}
            >
              <div>
                <Title>Bem-vindo ao EnergyFlow</Title>
                <Subtitle>
                  Monitore e preveja seu consumo energético de forma inteligente
                </Subtitle>
              </div>
              <Button onClick={handleRun} disabled={running} size="lg">
                <PlayCircle />{" "}
                {running ? "Rodando previsão..." : "Rodar previsão"}
              </Button>
            </div>
            {lastCount != null && (
              <p
                style={{ marginTop: 8, color: "hsl(var(--muted-foreground))" }}
              >
                Previsões geradas: {lastCount}
              </p>
            )}
            {errorMsg && (
              <p style={{ marginTop: 8, color: "hsl(var(--destructive))" }}>
                Erro ao rodar previsão: {errorMsg}
              </p>
            )}
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
