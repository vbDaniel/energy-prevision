import CardModule from "@/components/ui/Card";
import { Leaf, Sun, Wind } from "lucide-react";
import {
  TipsCard,
  Title,
  TipsList,
  TipItem,
  IconWrap,
  Content,
  TipTitle,
  TipDescription,
  Savings,
} from "./styles";

const tips = [
  {
    icon: Leaf,
    title: "Modo Eco Ativo",
    description: "Economize 25% usando horários de tarifa reduzida",
    savings: "R$ 87/mês",
  },
  {
    icon: Sun,
    title: "Energia Solar",
    description: "Considere painéis solares para sua residência",
    savings: "Até 95% de economia",
  },
  {
    icon: Wind,
    title: "Ventilação Natural",
    description: "Reduza o uso de ar-condicionado em 30%",
    savings: "R$ 62/mês",
  },
];

export const EconomyTips = () => {
  return (
    <TipsCard as={CardModule.Card}>
      <Title>Dicas de Economia</Title>
      <TipsList>
        {tips.map((tip, index) => (
          <TipItem key={index}>
            <IconWrap>
              <tip.icon size={20} color={"hsl(var(--primary))"} />
            </IconWrap>
            <Content>
              <TipTitle>{tip.title}</TipTitle>
              <TipDescription>{tip.description}</TipDescription>
              <Savings>{tip.savings}</Savings>
            </Content>
          </TipItem>
        ))}
      </TipsList>
    </TipsCard>
  );
};
