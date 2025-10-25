"use client";
import { useEffect, useState } from "react";
import Button from "@/components/ui/Button";
import { ChartCard, HeaderRow, Title, Subtitle, Controls } from "./styles";

import {
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  Area,
  AreaChart,
} from "recharts";

const data3Months = [
  { month: "Ago", consumo: 320, previsao: 310 },
  { month: "Set", consumo: 350, previsao: 340 },
  { month: "Out", consumo: 342, previsao: 335 },
  { month: "Nov", previsao: 330 },
  { month: "Dez", previsao: 315 },
  { month: "Jan", previsao: 325 },
];

const data6Months = [
  { month: "Ago", consumo: 320, previsao: 310 },
  { month: "Set", consumo: 350, previsao: 340 },
  { month: "Out", consumo: 342, previsao: 335 },
  { month: "Nov", previsao: 330 },
  { month: "Dez", previsao: 315 },
  { month: "Jan", previsao: 325 },
  { month: "Fev", previsao: 340 },
  { month: "Mar", previsao: 350 },
  { month: "Abr", previsao: 335 },
];

const data1Year = [
  { month: "Nov '23", consumo: 310, previsao: 305 },
  { month: "Dez '23", consumo: 295, previsao: 300 },
  { month: "Jan '24", consumo: 330, previsao: 320 },
  { month: "Fev", consumo: 340, previsao: 335 },
  { month: "Mar", consumo: 325, previsao: 330 },
  { month: "Abr", consumo: 315, previsao: 320 },
  { month: "Mai", consumo: 305, previsao: 310 },
  { month: "Jun", consumo: 320, previsao: 315 },
  { month: "Jul", consumo: 335, previsao: 325 },
  { month: "Ago", consumo: 320, previsao: 310 },
  { month: "Set", consumo: 350, previsao: 340 },
  { month: "Out", consumo: 342, previsao: 335 },
  { month: "Nov", previsao: 330 },
  { month: "Dez", previsao: 315 },
];

type Period = "3m" | "6m" | "1y";

export const ConsumptionChart = () => {
  const [period, setPeriod] = useState<Period>("3m");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const getData = () => {
    switch (period) {
      case "3m":
        return data3Months;
      case "6m":
        return data6Months;
      case "1y":
        return data1Year;
    }
  };

  return (
    <ChartCard>
      <HeaderRow>
        <div>
          <Title>Consumo Energético</Title>
          <Subtitle>Histórico e previsão de consumo</Subtitle>
        </div>
        <Controls>
          <Button
            variant={period === "3m" ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod("3m")}
          >
            3 Meses
          </Button>
          <Button
            variant={period === "6m" ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod("6m")}
          >
            6 Meses
          </Button>
          <Button
            variant={period === "1y" ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod("1y")}
          >
            1 Ano
          </Button>
        </Controls>
      </HeaderRow>

      {mounted ? (
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={getData()}>
            <defs>
              <linearGradient id="colorConsumo" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="hsl(var(--primary))"
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor="hsl(var(--primary))"
                  stopOpacity={0}
                />
              </linearGradient>
              <linearGradient id="colorPrevisao" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="hsl(var(--secondary))"
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor="hsl(var(--secondary))"
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(var(--border))"
              opacity={0.3}
            />
            <XAxis
              dataKey="month"
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickFormatter={(value) => `${value} kWh`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "var(--radius)",
                padding: "12px",
              }}
              labelStyle={{ color: "hsl(var(--foreground))", fontWeight: "bold" }}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="consumo"
              stroke="hsl(var(--primary))"
              strokeWidth={3}
              fill="url(#colorConsumo)"
              name="Consumo Real"
            />
            <Area
              type="monotone"
              dataKey="previsao"
              stroke="hsl(var(--secondary))"
              strokeWidth={3}
              strokeDasharray="5 5"
              fill="url(#colorPrevisao)"
              name="Previsão"
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div style={{ height: 350 }} />
      )}
    </ChartCard>
  );
};
